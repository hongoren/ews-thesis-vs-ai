# Credit Early Warning: Thesis Cox Model vs AI

Can AI improve a bank's early warning system for SME and corporate borrowers?

This project revisits my 2016 PhD thesis, *An Empirical Study on Credit Early Warning Systems* (Kadir Has University). The thesis predicted borrower default with a Cox proportional hazards model using only core banking data, with no financial statements: credit line usage, account balances and activity, bounced cheques, borrowing at other banks, and collateral.

Here I reproduce the thesis model exactly, then compare it with gradient boosting on the same data.

**[View the visual report](https://hongoren.github.io/ews-thesis-vs-ai/results/report.html)**

## Data

- 202,615 firm-quarters, 15,593 SME and corporate borrowers, 1,307 defaults
- Quarterly panel, 2005 Q1 to 2012 Q4, from a Turkish participation bank
- Customer IDs are transformed and no names are included

**The raw data is not included in this repository.** It belongs to the bank and stays private. The code is published so the method can be reviewed and applied to other lenders' data.

## Results

**1. The thesis model reproduces exactly.** All 39 hazard ratios and p-values match the thesis (log likelihood −10,636.664, LR χ² 2,673.37), as do the out-of-time flag counts.

**2. AI catches far more defaulters.** Flagging the same 16.8% of active firms the thesis model flagged at the end of 2011:

| Model | 2012 defaulters caught |
|---|---|
| Thesis Cox model | 37.8% (59 of 156) |
| Gradient boosting with behavior trends | **67.9%** (106 of 156) |

**3. Pooled over three test years (2010, 2011, 2012):**

| Horizon | Thesis Cox AUC | AI AUC | 95% bootstrap interval for the gain |
|---|---|---|---|
| 1 year (478 defaults) | 0.699 | **0.781** | +0.050 to +0.112 |
| 1 quarter (118 defaults) | 0.806 | **0.858** | +0.005 to +0.100 |

The AI model does not win every time. Trained through 2009 and tested on 2010, right after the 2008–09 crisis, the Cox model ranked slightly better for 1-year prediction (AUC 0.726 vs 0.708). Single-quarter tests have only 25 to 55 defaults, so the pooled results are the fair summary.

**4. Main drivers:** account activity (number of debit transactions), cash share of exposure, average balance, number of lending banks, and staff count. Firms with zero staff recorded default about 10 times more often than the rest, which most likely reflects missing data acting as a warning sign.

Full tables: [`results/`](results/). Visual report: [online](https://hongoren.github.io/ews-thesis-vs-ai/results/report.html) or [`results/report.html`](results/report.html).

## Method

- **Thesis Cox model:** Model 1 from the thesis, 39 covariates, time-varying covariates in counting-process form, Breslow ties, analysis time in quarters since credit was granted. Implemented directly with NumPy/SciPy (`src/cox.py`).
- **AI model:** scikit-learn histogram gradient boosting. Target: default within the next 4 quarters (or next quarter). Features: the thesis variables plus 1- and 4-quarter changes, rolling means and rolling extremes of usage, balances and transactions.
- **Out-of-time testing:** models are trained only on information available before each test year, then scored at the end of 2009, 2010 and 2011 on active, non-defaulted firms. Fixed settings, no tuning on test years.
- **Metrics:** AUC/Gini with 500-sample bootstrap intervals, plus the thesis's own measures, *selectivity ratio* (share of firms flagged) and *success rate for defaulters* (share of defaulters flagged).
- **Explanations:** permutation importance by variable group, and per-firm reason codes.

## Run it

```bash
pip install -r requirements.txt
export EWS_DATA=path/to/panel.txt   # tab-separated panel, same columns as the thesis data
python src/01_reproduce_thesis_model1.py
python src/02_thesis_out_of_time.py
python src/03_ai_models.py
python src/04_pooled_auc_bootstrap.py
python src/05_explanations.py
```

Run from the repository root. To use your own loan book, prepare a quarterly panel with the same column names.

## About

Haluk Öngören, PhD in Finance and Banking. I build credit early warning and default prediction models for lenders. Open to consulting projects.

## License

Code: MIT. Data: not included and not licensed.
