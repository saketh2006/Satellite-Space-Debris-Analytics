# 🛰️ Satellite Space Debris Analytics — Interactive Dashboard

**Course:** 23CSE353 Data Visualization — Primary Case Study Review  
**Institution:** Amrita Vishwa Vidyapeetham  
**Student Name:** Vudattu Sai Satya Saketh  
**Roll Number:** CB.SC.U4CSE24558  

---

## 📌 Project Overview
This repository contains the interactive Plotly Dash analytical dashboard and exploratory astrodynamics data analysis for **14,125 tracked orbital objects** in Earth orbit, sourced from the verified Space-Track General Perturbations (GP) catalog.

The project evaluates orbital decay dynamics, Low Earth Orbit (LEO) congestion, ballistic coefficient impact, and geopolitical orbital share across four analytical suites:
1. **Matplotlib Analytical Suite (M1 – M3):** Categorical frequency, launch evolution, and orbital regime distribution.
2. **Seaborn Statistical Suite (S1 – S5):** Orbital inclination distribution, orbital period violin distributions by object type, perigee-apogee eccentricity scatter, Pearson correlation heatmap, and Top-10 launching country Pareto analysis.
3. **Plotly Interactive Suite (P1 – P2):** Multidimensional orbital altitude vs. inclination scatter explorer, and hierarchical treemap of orbital populations.
4. **Ethical Visualization Lab (Case Study in Lie Factor):** Comparative study on the ethics of axis truncation in space debris risk reporting, evaluating a truncated-baseline chart (Lie Factor = 6.27) against an honest zero-baseline redesign.

Every visualization is accompanied by a structured **4-Quadrant Inference Card**:
- **Key Finding** (Sky Blue)
- **Calculated Evidence** (Emerald Green)
- **Contextual Interpretation** (Purple)
- **Analytical Caution / Limitation** (Amber)

---

## 🚀 Live Cloud Deployment
- **Platform:** Plotly Cloud
- **Main Application:** `plotly_app.py`
- **WSGI Runner:** Gunicorn

---

## 💻 Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/saketh2006/Satellite-Space-Debris-Analytics.git
cd Satellite-Space-Debris-Analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
python plotly_app.py
```
Open your browser and navigate to `http://127.0.0.1:8050`.

---

## 📁 Repository Structure
```
├── plotly_app.py               # Complete light-theme interactive Dash dashboard
├── space_decay_cleaned.csv     # Cleaned & feature-engineered catalog (14,125 objects)
├── requirements.txt            # Cloud & local deployment dependencies
├── README.md                   # Project documentation & case study details
└── .gitignore                  # Git exclusions for uncleaned raw data & cache
```
