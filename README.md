#  Used Car Auction Agent

An intelligent machine learning system that predicts used-car prices and automatically bids in simulated wholesale auctions.

---

## Overview

This project consists of two components:

- A LightGBM regression model trained on **447,048 historical auction records** to estimate vehicle prices.
- An automated bidding agent that predicts a vehicle's market value and places bids using a deterministic, budget-aware bidding strategy.

---

## Dataset

- 447,048 auction records
- 16 model features
- Extensive preprocessing and feature engineering

---

## Features

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- LightGBM Regression
- Hyperparameter Tuning
- Automated Auction Bidding
- Budget Management Strategy

---

## Model Performance

| Metric | Score |
|--------|-------|
| R² | **0.9524** |
| RMSE | **$2,117** |

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- Matplotlib

---

## Repository Structure

```
analysis.ipynb
agent.py
report.pdf
model.pkl
encoders.pkl
feature_constants.pkl
imputation_dicts.pkl
```

---

## Future Improvements

- Reinforcement Learning based bidding
- Live auction API integration
- Deep Learning models
