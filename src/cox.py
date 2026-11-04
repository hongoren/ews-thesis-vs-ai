import numpy as np
from scipy import stats
REG = ['DReg2','DReg3','DReg4','DReg5','DReg6','DReg7']
SEC = ['Sector_02','Sector_03','Sector_05','Sector_06','Sector_04','Sector_07','Sector_11','Sector_12','Sector_14','Sector_18']
M1 = REG+SEC+['Import','Export','Dsales_L','Dsales_M','NumberOfStaffQuarterly','QuarterlyAverageBalance',
 'NumberOfCredits','NumberOfDebits','TotalRisk','TotalLimit','LimitUtilizationRatio','CashRiskRatio',
 'TotalLimit_IOB','TotalRisk_IOB','LimitUtilizationRatio_IOB','Age_at_Loan','Check_Length','NumberOfBanks',
 'RtrnedChecksPrctn','CB_ReturnedChecksPrcnt','w_Collateral_1','w_Collateral_3','w_Collateral_2']

def fit_cox(X, time, event, iters=60, ridge=1e-8):
    """Breslow partial likelihood, counting-process rows: row at analysis time s is at risk only at s."""
    X = np.asarray(X, float); mu = X.mean(0); sd = X.std(0); sd[sd==0]=1
    Z = (X-mu)/sd; p = Z.shape[1]; b = np.zeros(p)
    times = np.unique(time[event==1]); idx = {s: np.where(time==s)[0] for s in times}
    def llgh(b):
        ll=0; g=np.zeros(p); H=np.zeros((p,p))
        for s in times:
            r=idx[s]; z=Z[r]; e=event[r]==1; d=e.sum()
            eta=z@b; m=eta.max(); w=np.exp(eta-m); W=w.sum()
            zbar=(w@z)/W; zz=(z*w[:,None]).T@z/W
            ll+=eta[e].sum()-d*(m+np.log(W)); g+=z[e].sum(0)-d*zbar; H-=d*(zz-np.outer(zbar,zbar))
        return ll,g,H
    ll,g,H=llgh(b); ll0=ll
    for it in range(iters):
        step=np.linalg.solve(H-ridge*np.eye(p), -g); t=1.0
        while True:
            nb=b+t*step; nll,ng,nH=llgh(nb)
            if nll>=ll-1e-9 or t<1e-4: break
            t/=2
        conv=abs(nll-ll)<1e-8; b,ll,g,H=nb,nll,ng,nH
        if conv: break
    cov=np.linalg.inv(-H)
    beta=b/sd; se=np.sqrt(np.diag(cov))/sd
    return dict(beta=beta, se=se, p=2*stats.norm.sf(np.abs(beta/se)), ll=ll, ll0=ll0, mu=mu, sd=sd, bz=b)
