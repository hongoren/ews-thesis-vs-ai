import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from load import load; from cox import fit_cox, M1
df=load()
r=fit_cox(df[M1].values, df.Age_Record_Quarter.values, df.Event.values)
out=pd.DataFrame({'var':M1,'HR':np.exp(r['beta']),'p':r['p'].round(3)})
print(out.to_string()); print('LL',r['ll'],'LR chi2',2*(r['ll']-r['ll0']))
out.to_csv('m1_full.csv',index=False)
