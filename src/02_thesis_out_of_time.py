import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from load import load; from cox import fit_cox, M1
df=load()
for endY in [2010,2011]:
    tr=df[df.Year<=endY]
    r=fit_cox(tr[M1].values, tr.Age_Record_Quarter.values, tr.Event.values)
    last=tr[(tr.Year==endY)&(tr.Quarter==4)&(tr.Event==0)]
    rs=np.exp(last[M1].values@r['beta']); flag=rs>=1
    fut=df[(df.Year==endY+1)&(df.Event==1)][['CustomerID','Quarter']]
    q1=last.CustomerID.isin(fut[fut.Quarter==1].CustomerID).values; y1=last.CustomerID.isin(fut.CustomerID).values
    print(endY,'active',len(last),'flag',flag.sum(),'sel %.2f'%(100*flag.mean()),
          'Q1 def',q1.sum(),'caught',(q1&flag).sum(),'yr def',y1.sum(),'caught',(y1&flag).sum())
