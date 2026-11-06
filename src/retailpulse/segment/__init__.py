"""P3a customer segmentation: K-Means vs an RFM-quartile baseline, validated statistically.

cluster.py   preprocessing pipeline, k scan + choice rule, ids by revenue, naming, end-to-end
baseline.py  RFM quartile scores / groups, cluster quality, CV AUC on returned_flag
validate.py  subsample stability (ARI), PCA, chi-square segment x UK
stage.py     I/O only: read features_cutoff + labels, write segment artifacts + metrics
"""
