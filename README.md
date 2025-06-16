# Master Thesis – Kriging and Data Valuation in Mobile Network Coverage Prediction

**Author**: Matthew Samuel  
**Institution**: Universidad Politécnica de Madrid , University of Ghent 
**Date**: June 2025  
**Degree**: MSc in Electronics and ICT Engineering Technology
**Thesis Supervisors**: Prof. Santiago Andrés Azcoitia (UPM), Prof. Dr. Zoraida Frías (UPM)

## Overview

This repository contains the full codebase developed for a master's thesis focusing on spatial coverage prediction in mobile networks. The core methodology combines **Kriging interpolation** with **data valuation via Shapley values**. 
The objective is to enhance the efficiency and accuracy of signal strength estimation (RSRP) by selecting the most valuable data points from large, crowdsourced measurement datasets.
The code supports both prediction and valuation for **multiple spatial targets in parallel** using **Dask** for geohash-level and event-level.
All experiments and simulations in the TSSAnalysis file are executed at the **Geohash aggregation level** (Scenario 1), as event-level evaluation was computationally infeasible. 



## Features

- Kriging-based spatial interpolation for RSRP prediction  
- Shapley value estimation using **Truncated Structured Sampling (TSS)**  
- Robustness analysis across multiple truncation and sampling parameters  
- CSV export of prediction and valuation results  
- Integrated plotting of relevant metrics and visual diagnostics

## File Structure

```text
.
├── eventLevel/                              # Event-level implementation (Scenario 2)
│   ├── input/                               # Input files for event-level analysis
│   ├── FrameworkEvent.py                    # Evaluation framework for event-level Kriging and Shapley
│   └── mainEvent.ipynb                      # Notebook to run event-level predictions and analysis
│
├── geohashLevel/                            # GeoHash-level implementation (Scenario 1)
│   ├── input/                               # Input files for geohash-level analysis
│   ├── FrameworkGEO.py                      # Evaluation framework for geohash-level predictions
│   ├── TSSAnalysis.ipynb                    # Robustness analysis of Shapley estimation parameters
│   └── mainGEO.ipynb                        # Notebook to run geohash-level predictions and analysis
│
├── Output/                                  # Needs to be added by user to store output csv's
├── README.md                                # Project description and usage guide
└── LICENSE                                   # Project license

```

Requirements
	•	Python 3.9+
	•	Dask (for parallel computation)
	•	NumPy, pandas, matplotlib
	•	Jupyter Notebook


Usage
	1.	Open one of the Jupyter Notebook files, depending on the chosen scenario:
	  •	geohashLevel/mainGEO.ipynb → Scenario 1: GeoHash-level prediction and valuation
	  •	eventLevel/mainEvent.ipynb → Scenario 2: Event-level prediction and valuation
	2.	Set the number of Dask workers according to the available computing resources.
	3.	Check and update input paths to ensure the correct location of datasets and configuration files.
	4.	Configure key parameters in the notebook:
	  •	empty_geohashes_to_predict: number of empty GeoHashes for which predictions will be made
	  •	target_index: index of the target GeoHash within the sorted list (based on neighbor count)
	5.	(Optional) If output results are to be saved:
	  •	Create an Output/ directory manually if it does not yet exist
	  •	The script will automatically save prediction results and Shapley valuations in CSV format in that folder

Once these steps are completed, the notebook will execute Kriging-based predictions and Shapley value computations in parallel using Dask. Optionally, it will generate plots for result interpretation.



Citation
If this repository is used for academic or industrial research, please cite the corresponding thesis once published. For now, attribution can be given as:

Matthew Samuel, Kriging and Data Valuation in Mobile Network Coverage Prediction, Master’s Thesis, Universidad Politécnica de Madrid, 2025.
GitHub: https://github.com/matthewsamuel/MasterThesis


