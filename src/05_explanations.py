import pickle, numpy as np, pandas as pd, re
from sklearn.metrics import roc_auc_score
m,FULL,tr,test=pickle.load(open('gb2011.pkl','rb'))
X=test[FULL].copy(); y=test.y4.values; base=roc_auc_score(y,m.predict_proba(X)[:,1])
def grp(c):
    c=re.sub(r'_(d1|d4|m4)$','',c)
    if c.startswith('Sector_'): return 'Sector'
    if c.startswith('DReg'): return 'Region'
    if c.startswith('Dsales'): return 'Firm size (sales)'
    if c.startswith('w_Collateral'): return 'Collateral mix'
    return {'Rtrned_max4':'RtrnedChecksPrctn','LUR_max4':'LimitUtilizationRatio','Bal_min4':'QuarterlyAverageBalance',
            'log_limit_ratio_iob':'TotalLimit_IOB'}.get(c,c)
groups={}
for c in FULL: groups.setdefault(grp(c),[]).append(c)
# use a larger, all-firm pool for permutation stability: test set at 2011Q4
rng=np.random.default_rng(0); imp=[]
for gname,cols in groups.items():
    drops=[]
    for r in range(10):
        Xp=X.copy(); perm=rng.permutation(len(X))
        Xp[cols]=X[cols].values[perm]
        drops.append(base-roc_auc_score(y,m.predict_proba(Xp)[:,1]))
    imp.append((gname,np.mean(drops),np.std(drops)))
I=pd.DataFrame(imp,columns=['group','auc_drop','sd']).sort_values('auc_drop',ascending=False)
print('base AUC',base); print(I.round(4).to_string()); I.to_csv('importance.csv',index=False)
# reason codes for top-scored firms that did default: replace each group with training median, see score drop
p=m.predict_proba(X)[:,1]; test=test.assign(p=p)
med=tr[FULL].median()
ex=test[test.y4==1].sort_values('p',ascending=False).head(3)
for _,row in ex.iterrows():
    x=row[FULL].to_frame().T.astype(float); p0=m.predict_proba(x)[0,1]; rc=[]
    for gname,cols in groups.items():
        x2=x.copy(); x2[cols]=med[cols].values; rc.append((gname,p0-m.predict_proba(x2)[0,1]))
    rc=sorted(rc,key=lambda z:-z[1])[:4]
    print(f"firm score {p0:.3f} pctile {(p<p0).mean():.3f}", [(a,round(b,3)) for a,b in rc],
          'LUR',round(row.LimitUtilizationRatio,2),'cash',round(row.CashRiskRatio,2),'rtrn',round(row.RtrnedChecksPrctn,3),'banks',row.NumberOfBanks)
print('portfolio 4q default rate', y.mean(), 'top5% rate', y[p>=np.quantile(p,.95)].mean())
