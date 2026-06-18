 PropIQ — AI House Price Prediction App

> An end-to-end Machine Learning application predicting Pakistani real estate prices using 53,000+ Zameen.com listings.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

 Project Overview

PropIQ solves a real problem — Pakistan's property market has zero price transparency. Buyers rely on agents, agents guess, nobody wins.

This app trains a **Random Forest Regressor** on real Zameen.com listings and deploys it as a professional web app where users can instantly get AI-powered price estimates for any property across Pakistan.

---

 Features

-  **AI Price Estimation** — Random Forest model with R² Score of 0.73
-  **53,000+ Listings** — Trained on real Zameen.com data
-  **100+ Cities** — Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad & more
-  **Price Range** — Low / AI Estimate / High confidence range
-  **Professional UI** — Dark gradient Streamlit app with Roboto typography
-  **Pakistani Format** — Prices displayed in Lakh and Crore

---

 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.x |
| Data | Pandas, NumPy |
| ML Model | Scikit-learn — Random Forest Regressor |
| Deployment | Streamlit |
| Dataset | Zameen.com (53,255 listings) |

---

 Project Structure

```
propiq-house-price-prediction/
│
├── app.py                  ← Streamlit web application (PropIQ UI)
├── train.py                ← Data cleaning + model training script
├── requirements.txt        ← Python dependencies
├── .gitignore              ← Excludes large files
└── README.md               ← You are here
```

> **Note:** `model.pkl`, `model_features.pkl` and `raw_data_zameen.csv` are excluded from this repo due to file size. Run `train.py` to regenerate them.


 🚀 Getting Started

 1. Clone the repo
```bash
git clone https://github.com/Basugondal/-propiq-house-price-prediction.git
cd -propiq-house-price-prediction
```

 2. Install dependencies
```bash
pip install -r requirements.txt
```

 3. Add dataset
Download `raw_data_zameen.csv` from [Kaggle](https://www.kaggle.com) and place it in the root folder.

 4. Train the model
```bash
python train.py
```
This generates `model.pkl` and `model_features.pkl`.

 5. Run the app
```bash
streamlit run app.py
```

ML Pipeline

```
Raw Data (53,255 listings)
        ↓
Custom Cleaning
  • Regex price parser (PKR 1.2 Crore → 12,000,000)
  • Area converter (Kanal/Marla → Marla)
  • Outlier removal (99th percentile)
        ↓
Feature Engineering
  • One-hot encoding (city, province, purpose)
  • 160 total features
        ↓
Train/Test Split (80/20)
        ↓
Random Forest Regressor
  • n_estimators = 300
  • n_jobs = -1
        ↓
Evaluation
  • R² Score: 0.73
  • MAE: ~9.4M PKR
        ↓
Streamlit Deployment
```



 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.7277 |
| MAE | ~9.38M PKR |
| Training Samples | 11,028 |
| Test Samples | 2,758 |
| Total Features | 160 |
<img width="720" height="1403" alt="WhatsApp Image 2026-06-18 at 3 40 58 PM" src="https://github.com/user-attachments/assets/32712048-0b5b-496f-8f0b-e66a57414f5d" />
<img width="720" height="1126" alt="WhatsApp Image 2026-06-18 at 3 40 59 PM" src="https://github.com/user-attachments/assets/032d412f-9dd9-404b-a326-607e82fd530f" />
<img width="720" height="1430" alt="WhatsApp Image 2026-06-18 at 3 41 02 PM" src="https://github.com/user-attachments/assets/151bca51-64a3-4738-9a5b-08fab1a290fd" />
<img width="720" height="1426" alt="WhatsApp Image 2026-06-18 at 3 41 02 PM (1)" src="https://github.com/user-attachments/assets/8fe14d0e-22d1-49a0-8591-57164e90011f" />
Built by **Abdul Basit**
BSc Software Engineering · University of Sargodha · 2026
Flutter Developer & AI/ML Enthusiast · Pakistan
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/Basugondal)



