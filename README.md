# 🏠 Property Price Prediction Models

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

This project provides **polygon-specific property CMU price prediction** using **LightGBM** and **KMeans clustering**. It handles polygons of varying sizes and includes a fallback mechanism for very small datasets.

---

## **✨ Features**

- Read property data from Excel with **retry logic**
- Encode categorical features automatically
- Train polygon-specific models:
  - **Full polygon models** (≥30 rows) with **KMeans + LightGBM**
  - **Small polygon models** (10–29 rows) with **LightGBM only**
  - **Fallback for very small polygons** using **nearest neighbors**
- Save and load trained models via `joblib`
- Predict CMU price for a given property and location

---

## **📦 Installation**

### Requirements

```bash
Python 3.9+
pandas
numpy
scikit-learn
lightgbm
joblib
