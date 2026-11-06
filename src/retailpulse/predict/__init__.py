"""P3b prediction: out-of-time return classifiers, spend regression and the backtest.

dataset.py   train snapshot (cutoff - offset) / test snapshot (real cutoff) + preprocessing
classify.py  logistic, LDA, QDA, naive Bayes, KNN; CV on train, threshold on train, test once
regress.py   log1p(next_spend) of returners: simple LR, multiple LR, KNN regression
backtest.py  "we predicted N would return; X% did vs base rate Y%" + decile table
stage.py     I/O only: reads artifacts, writes predictions/, models/, metrics.json
"""
