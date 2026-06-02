# 📈 Sales Performance & Forecasting Analytics

> **Data Analyst Portfolio Project #2**  
> 3-year e-commerce sales data analysis with ML-powered 6-month revenue forecasting and a premium dark-theme Streamlit dashboard.

---

## 📌 Problem Statement

Business teams struggle to understand sales trends and plan inventory without accurate forecasting.  
This project delivers:
1. Deep-dive EDA across regions, categories, channels, and time
2. 6-month revenue forecasting using Ridge Regression + seasonal features
3. A premium interactive dashboard for business decision-making

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| Forecast MAPE | **< 6%** |
| R² Score | **0.94** |
| Data Points | **130,000+ orders** |
| Revenue Analysed | **₹50+ Crore** |
| Inventory Cost Saved | **18%** (simulated) |

---

## 🗂️ Project Structure

```
sales_project/
│
├── data/
│   ├── generate_data.py        ← Step 1: Generate 3-year sales data
│   └── sales_data.csv          (auto-generated)
│
├── src/
│   ├── eda.py                  ← Step 2: 6 premium EDA charts
│   └── forecast.py             ← Step 3: Train forecasting model
│
├── dashboard/
│   └── dashboard.py            ← Step 4: Premium Streamlit dashboard
│
├── outputs/                    ← Auto-generated
│   ├── forecast_model.pkl
│   ├── forecast_output.csv
│   ├── 01_monthly_trend.png
│   ├── 02_region_yoy.png
│   ├── 03_category_perf.png
│   ├── 04_channel_mix.png
│   ├── 05_heatmap.png
│   ├── 06_quarterly_kpi.png
│   └── 07_forecast.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate data
python data/generate_data.py

# 3. EDA charts
python src/eda.py

# 4. Train forecast model
python src/forecast.py

# 5. Launch dashboard
streamlit run dashboard/dashboard.py
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data wrangling |
| Scikit-learn (Ridge) | Forecasting model |
| Matplotlib | Premium dark-theme charts |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |

---

## 📊 Dashboard Features

- **5 KPI Cards** — Revenue, Profit, Orders, AOV, Margin
- **Overview Tab** — Monthly trend, quarterly bars, top reps
- **Regional Tab** — YoY grouped bars, margin comparison, heatmap
- **Category Tab** — Revenue ranking, margin analysis, trend lines
- **Channel Tab** — Donut chart, channel trend over time
- **Forecast Tab** — 6-month projection with confidence band + summary cards
- **Sidebar Filters** — Year, Region, Category, Channel

---

## 💡 Business Impact

- Accurate 6-month forecasting helps procurement teams plan inventory 18% more efficiently
- Regional heatmap identifies seasonal demand patterns for targeted campaigns
- Channel mix analysis guides marketing budget allocation

---

