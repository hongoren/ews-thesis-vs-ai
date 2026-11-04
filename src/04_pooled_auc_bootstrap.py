import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
P=pd.read_pickle('preds.pkl').reset_index()
# rank-normalize scores within each origin so pooled AUC isn't driven by level shifts
for m in ['cox','logit','gb_base','gb_full','gb_full_q']: P[m+'_r']=P.groupby('endY')[m].rank(pct=True)
rng=np.random.default_rng(1)
out=[]
for hz,y in [('1q','y1'),('1y','y4')]:
    for m in ['cox','logit','gb_base','gb_full','gb_full_q']:
        a=roc_auc_score(P[y],P[m+'_r'])
        # bootstrap diff vs cox, resampling rows within origin
        diffs=[]
        for b in range(500):
            idx=np.concatenate([rng.choice(gi, len(gi)) for _,gi in P.groupby('endY').indices.items()])
            s=P.iloc[idx]
            if m!='cox': diffs.append(roc_auc_score(s[y],s[m+'_r'])-roc_auc_score(s[y],s['cox_r']))
        lo,hi=(np.percentile(diffs,[2.5,97.5]) if diffs else (np.nan,np.nan))
        out.append(dict(horizon=hz,model=m,defaults=int(P[y].sum()),auc=a,gini=2*a-1,diff_vs_cox_lo=lo,diff_vs_cox_hi=hi))
print(pd.DataFrame(out).round(3).to_string())
pd.DataFrame(out).to_csv('pooled.csv',index=False)
