# Probabilistic Forecasting of Household Energy and Evaluation of Energy Self-Sufficiency Rate Using Pre-trained Time Series Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code and analysis notebooks for the research paper titled **"Probabilistic Forecasting of Household Energy and Evaluation of Energy Self-Sufficiency Rate Using Pre-trained Time Series Models"**.

## 📌 Overview

This study proposes a framework to quantify the uncertainty in household energy autonomy. By integrating **Chronos** (a pre-trained time-series foundation model) with **Monte Carlo Simulation**, we probabilistically forecast the **Energy Self-Sufficiency Ratio (ESSR)** over the medium term.

### Key Features
* **Probabilistic Forecasting**: Implementation of Chronos model fine-tuned on household energy data.
* **ESSR Evaluation**: Monte Carlo Simulation to derive probability distributions of annual and monthly ESSR.
* **Baseline Comparison**: Performance comparison with Seasonal Naive (SN) models.

---

## 📄 Paper Information

If you use this code or ideas in your research, please cite our paper:

> **Title**: Probabilistic Forecasting of Household Energy and Evaluation of Energy Self-Sufficiency Rate Using Pre-trained Time Series Models  
> **Authors**:  
> **Journal**: *Energies* (Submitted, 2025)  
> **DOI**: 

---

## ⚠️ Data Privacy & Mock Data

**Important Note on Data Availability:**
The actual dataset used in the paper (HEMS data from 39 households in Kitakyushu City) contains private information and **cannot be made publicly available** due to privacy restrictions and non-disclosure agreements.

Therefore, this repository provides **mock data (dummy data)** in the `data/` directory to demonstrate the functionality of the code.
* The structure (column names, data types) is identical to the original cleaned data.
* The values are randomly generated and do not reflect actual household behaviors.
* Please verify the code logic using this sample data.

---

## 📂 Repository Structure

The analysis is divided into 5 steps.

```text
.
├── sample_cleaned_data.csv # Dummy dataset (Cleaned version) for demonstration
├── 01_eda.py                 # Step 1: Exploratory Data Analysis (ACF/PACF, etc.)
├── 02_forecasting_chronos.py # Step 2: Chronos model execution (Requires GPU)
├── 03_forecasting_baseline.py # Step 3: Seasonal Naive baseline model
├── 04_accuracy_comparison.py  # Step 4: Comparison of MAE, RMSE, and MASE
├── 05_essr_simulation.py     # Step 5: Monte Carlo Simulation for ESSR
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT License
```

---

## 🚀 Getting Started & Usage

This project requires different environments depending on the notebook.

### 1. Environment Setup

#### A. Local Environment (CPU)
Recommended for notebooks: `01`, `03`, `04`, `05`
Use your local machine (PC) for lightweight analysis and simulations.

1.  Clone the repo:
    ```sh
    git clone [https://github.com/YourUsername/repo-name.git](https://github.com/YourUsername/repo-name.git)
    ```
2.  Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

#### B. GPU Environment (Google Colab)
**Required** for notebook: `02_forecasting_chronos.ipynb`
The Chronos model requires a GPU for efficient fine-tuning and inference. We highly recommend running this notebook on **Google Colab**.

* Upload `02_forecasting_chronos.ipynb` and the data file to your Google Drive or Colab environment.
* Install necessary libraries within the notebook (commands are included in the notebook).

---

### 2. Running the Notebooks

Please execute the notebooks in the following order:

**Step 1: Exploratory Data Analysis**
* **File**: `01_eda.ipynb` (Local CPU)
* **Description**: Visualizes time series plots and seasonality (ACF/PACF) using the cleaned dataset.

**Step 2: Forecasting with Chronos**
* **File**: `02_forecasting_chronos.ipynb` (**GPU Required**)
* **Description**: Fine-tunes the Chronos model and generates probabilistic forecasts (quantiles).
* **Output**: Saves the forecast results (e.g., `chronos_forecasts.csv`) for later comparison.

**Step 3: Forecasting with Baseline**
* **File**: `03_forecasting_baseline.ipynb` (Local CPU)
* **Description**: Generates forecasts using the Seasonal Naive (SN) approach.
* **Output**: Saves the baseline forecast results.

**Step 4: Accuracy Comparison**
* **File**: `04_accuracy_comparison.ipynb` (Local CPU)
* **Description**: Loads results from Step 2 and Step 3. Calculates metrics (MAE, RMSE, MASE) and visualizes the comparison (Box plots).

**Step 5: ESSR Simulation**
* **File**: `05_essr_simulation.ipynb` (Local CPU)
* **Description**: Performs Monte Carlo Simulation using the probabilistic forecasts from Step 2 to calculate and visualize ESSR probability distributions.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
