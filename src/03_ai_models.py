import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from load import load; from cox import fit_cox, M1
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
df=load().sort_values(['CustomerID','t']).reset_index(drop=True)
g=df.groupby('CustomerID')
# label: default within next 4 quarters (firm defaults in rows t+1..t+4)
evt_t=df[df.Event==1].set_index('CustomerID').t
df['def_t']=df.CustomerID.map(evt_t)
df['y4']=((df.def_t>df.t)&(df.def_t<=df.t+4)).astype(int)
df['y1']=(df.def_t==df.t+1).astype(int)
# engineered behavioral dynamics
dyn=['LimitUtilizationRatio','CashRiskRatio','QuarterlyAverageBalance','NumberOfCredits','NumberOfDebits',
     'RtrnedChecksPrctn','TotalRisk','TotalLimit','TotalRisk_IOB','TotalLimit_IOB','NumberOfBanks','LimitUtilizationRatio_IOB']
for c in dyn:
    df[c+'_d1']=g[c].diff(1); df[c+'_d4']=g[c].diff(4)
    df[c+'_m4']=g[c].transform(lambda s:s.rolling(4,min_periods=1).mean())
df['Rtrned_max4']=g['RtrnedChecksPrctn'].transform(lambda s:s.rolling(4,min_periods=1).max())
df['LUR_max4']=g['LimitUtilizationRatio'].transform(lambda s:s.rolling(4,min_periods=1).max())
df['Bal_min4']=g['QuarterlyAverageBalance'].transform(lambda s:s.rolling(4,min_periods=1).min())
df['log_limit_ratio_iob']=np.log1p(df.TotalLimit_IOB)-np.log1p(df.TotalLimit)
SEC_ALL=[c for c in df.columns if c.startswith('Sector_')]; REG_ALL=[c for c in df.columns if c.startswith('DReg')]
BASE=sorted(set(M1+SEC_ALL+REG_ALL+['Dsales_S','w_Collateral_4','Age_Record_Quarter']))
ENG=[c for c in df.columns if c.endswith(('_d1','_d4','_m4'))]+['Rtrned_max4','LUR_max4','Bal_min4','log_limit_ratio_iob']
FULL=BASE+ENG
df.to_pickle('panel.pkl')

def sel_success(score, y, sels):
    order=np.argsort(-score); n=len(score); out={}
    for s in sels:
        k=int(round(s*n)); f=np.zeros(n,bool); f[order[:k]]=True
        out[s]=(y&f).sum()/max(y.sum(),1)
    return out
res=[]; preds={}
SELS=[0.02,0.05,0.10,0.15,0.20,0.25]
for endY in [2009,2010,2011]:
    origin=(endY-2005)*4+4
    trc=df[df.Year<=endY]
    r=fit_cox(trc[M1].values, trc.Age_Record_Quarter.values, trc.Event.values)
    test=df[(df.t==origin)&(df.Event==0)].copy()
    test['cox']=test[M1].values@r['beta']
    yq=test.y1.values.astype(bool); yy=test.y4.values.astype(bool)
    # train ML on firm-quarters whose 4q label window is fully observed before origin
    tr=df[(df.t<=origin-4)&(df.Event==0)]
    tr1=df[(df.t<=origin-1)&(df.Event==0)]
    models={}
    lg=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))
    Xl=tr[M1].clip(tr[M1].quantile(.001),tr[M1].quantile(.999),axis=1)
    lg.fit(Xl,tr.y4); test['logit']=lg.decision_function(test[M1].clip(tr[M1].quantile(.001),tr[M1].quantile(.999),axis=1))
    for name,cols in [('gb_base',BASE),('gb_full',FULL)]:
        m=HistGradientBoostingClassifier(learning_rate=0.05,max_iter=400,max_leaf_nodes=15,min_samples_leaf=200,
              l2_regularization=1.0,early_stopping=False,random_state=0)
        m.fit(tr[cols],tr.y4); test[name]=m.predict_proba(test[cols])[:,1]; models[name]=m
    # 1-quarter-ahead GB trained on 1q labels
    m1=HistGradientBoostingClassifier(learning_rate=0.05,max_iter=400,max_leaf_nodes=15,min_samples_leaf=200,
              l2_regularization=1.0,early_stopping=False,random_state=0).fit(tr1[FULL],tr1.y1)
    test['gb_full_q']=m1.predict_proba(test[FULL])[:,1]
    cox_sel=(test.cox>=0).mean()
    for mod in ['cox','logit','gb_base','gb_full','gb_full_q']:
        for hz,y in [('1q',yq),('1y',yy)]:
            ss=sel_success(test[mod].values,y,SELS+[cox_sel])
            res.append(dict(window=f'[2005,{endY}]',model=mod,horizon=hz,n=len(test),defaults=int(y.sum()),
                 auc=roc_auc_score(y,test[mod]),**{f'succ@{int(s*100)}':ss[s] for s in SELS},succ_at_cox=ss[cox_sel],cox_sel=cox_sel))
    preds[endY]=test[['CustomerID','cox','logit','gb_base','gb_full','gb_full_q','y1','y4']]
    if endY==2011: import pickle; pickle.dump((models['gb_full'],FULL,tr,test),open('gb2011.pkl','wb'))
R=pd.DataFrame(res); R['gini']=2*R.auc-1
pd.set_option('display.width',250); print(R.round(3).to_string())
R.to_csv('results.csv',index=False)
pd.concat(preds,names=['endY']).to_pickle('preds.pkl')
