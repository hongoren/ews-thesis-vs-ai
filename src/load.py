import os
import pandas as pd, numpy as np
def load():
    df = pd.read_csv(os.environ.get('EWS_DATA','data/panel.txt'), sep='\t')
    df['Quarter'] = df['Quarter'].str.replace('"','').str[1].astype(int)
    df['Age_Record_Quarter'] = pd.to_numeric(df['Age_Record_Quarter'].astype(str).str.replace('"',''))
    for c in ['CustomerID','Year','Event','Age_Record_Quarter']: df[c]=df[c].astype(int)
    df['t'] = (df.Year-2005)*4 + df.Quarter  # calendar quarter index 1..32
    return df
