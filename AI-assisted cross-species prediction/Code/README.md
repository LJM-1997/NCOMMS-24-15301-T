﻿# Introduction of code and data
This is the implementation for cross-substrate and cross-photocatalyst prediction, which includes two main parts: Data preprocessing and Model building.

## Data preprocessing
* We provide a step-by-step workflow in ``$ data_preprocessing.ipynb`` to show the preprocessing of the data. 
* This section includes environment set up, mordred fingerprint generation and dimension-reduction module, which takes Photocatalysis_data.csv as original data input.
* We also provide the data after preprocessing in data_after_preprocessing.csv. 

## Model building
* We provide a step-by-step workflow in ``$ cross_validation.ipynb`` to show the model training and experimental results.
* The code uses data_after_preprocessing.csv as input.
* ``$ cross_validation.ipynb`` also includes the cross-validation code and results for cross-substrate and cross-photocatalyst prediction.


