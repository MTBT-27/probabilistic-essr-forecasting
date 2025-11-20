# Probabilistic Forecasting of Household Energy and Evaluation of Energy Self-Sufficiency Rate Using Pre-trained Time Series Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code and analysis notebooks for the research paper titled **"Probabilistic Forecasting of Household Energy and Evaluation of Energy Self-Sufficiency Rate Using Pre-trained Time Series Models"**.

## 📌 Overview

This study proposes a framework to quantify the uncertainty in household energy autonomy. By integrating **Chronos** (a pre-trained time-series foundation model) with **Monte Carlo Simulation**, we probabilistically forecast the **Energy Self-Sufficiency Ratio (ESSR)** over the medium term.

The code covers the entire workflow from data preprocessing and Exploratory Data Analysis (EDA) to probabilistic forecasting and ESSR risk evaluation.

### Key Features
* **Probabilistic Forecasting**: Implementation of Chronos model finetuned on household energy data.
* **ESSR Evaluation**: Monte Carlo Simulation to derive probability distributions of annual and monthly ESSR.
* **Baseline Comparison**: Performance comparison with Seasonal Naive (SN) models.

---

## 📄 Paper Information

If you use this code or ideas in your research, please cite our paper:

> **Title**: Probabilistic Forecasting of Household Energy and Evaluation of Energy Self-Sufficiency Rate Using Pre-trained Time Series Models  
> **Authors**: Hiroki Yamasaki, Libei Wu, and Masaaki Nagahara  
> **Journal**: *Energies* (Submitted/Published, 2025)  
> **DOI**: [INSERT DOI HERE if available]

---

## ⚠️ Data Privacy & Mock Data

**Important Note on Data Availability:**
The actual dataset used in the paper (HEMS data from 39 households in Kitakyushu City) contains private information and **cannot be made publicly available** due to privacy restrictions and non-disclosure agreements.

Therefore, this repository provides **mock data (dummy data)** in the `data/` directory to demonstrate the functionality of the code.
* The structure (column names, data types) is identical to the original data.
* The values are randomly generated and do not reflect actual household behaviors.
* Please verify the code logic using this sample data.

---

## 📂 Repository Structure

```text
.
├── data/
│   └── sample_data.csv      # Dummy dataset for demonstration
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Data cleaning and preprocessing logic
│   ├── 02_eda.ipynb                # Exploratory Data Analysis (ACF/PACF plots, etc.)
│   ├── 03_forecasting_chronos.ipynb # Chronos model execution (recommended on Colab)
│   ├── 04_forecasting_baseline.ipynb # Seasonal Naive baseline model
│   ├── 05_accuracy_comparison.ipynb  # Comparison of MAE, RMSE, and MASE
│   └── 06_essr_simulation.ipynb    # Monte Carlo Simulation for ESSR distributions
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT License
