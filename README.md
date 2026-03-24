projekt_genomforschung
Building and training multi-modal machine learning models on paired cancer genomics and drug chemical structure to predict and understand tumour sensitivity profiles. 

# Phase 1: Computational Project (Months 1–6)
## Data integration and preprocessing (Month 1)
Download and harmonise drug response data from GDSC2 and gene expression profiles from DepMap \
Restrict genomic features to curated L1000 landmark gene set \
Use RDKit to compute Morgan fingerprints and standard physicochemical descriptors
## Baseline modelling and exploratory analysis (Months 2–3)
Train single-modality baseline models:\
Genomic features only\
Chemical descriptors only\
Cross-validate and test on held-out cell lines and drugs\
Dimensionality reduction analysis for spaces exploration\
Clustering for identifying co-sensitive cell lines and structurally related drug groups
## Fusion strategy comparison (Months 4–5)
Implement and benchmark 3 fusion strategies:\
Early fusion: direct concatenation of genomic and chemical feature vectors\
Late fusion: independent models and ensembled predictions\
Attention-based fusion: weights genomic features conditionally on chemical input and vice versa\
Evaluate against known baselines 
## Writing report (Month 6)
Write report\
Introduce GNNs as natural future direction
