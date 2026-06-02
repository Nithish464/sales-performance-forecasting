"""
forecast.py  –  Sales Forecasting Model
Uses Linear Regression with time features + seasonal decomposition.
Forecasts next 6 months revenue. Saves model + forecast CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

os.makedirs('../outputs', exist_ok=True)

BG   = '#0F172A'
CARD = '#1E293B'
TEXT = '#F1F5F9'
MUTED= '#94A3B8'
C    = ['#4F46E5','#10B981','#F59E0B','#EF4444']

# ── Load & aggregate monthly ─────────────────────────────
df = pd.read_csv('sales_data.csv', parse_dates=['date'])
monthly = df.groupby(df['date'].dt.to_period('M')).agg(
    revenue=('revenue','sum'),
    profit =('profit','sum'),
    orders =('order_id','count')
).reset_index()
monthly['ds']     = monthly['date'].dt.to_timestamp()
monthly['rev_cr'] = monthly['revenue'] / 1e7
monthly.sort_values('ds', inplace=True)
monthly.reset_index(drop=True, inplace=True)

# ── Feature engineering ──────────────────────────────────
monthly['t']         = np.arange(len(monthly))
monthly['month_num'] = monthly['ds'].dt.month
monthly['year']      = monthly['ds'].dt.year
monthly['sin_12']    = np.sin(2 * np.pi * monthly['month_num'] / 12)
monthly['cos_12']    = np.cos(2 * np.pi * monthly['month_num'] / 12)
monthly['sin_6']     = np.sin(2 * np.pi * monthly['month_num'] / 6)
monthly['cos_6']     = np.cos(2 * np.pi * monthly['month_num'] / 6)
monthly['is_festive']= monthly['month_num'].isin([10,11,12]).astype(int)
monthly['lag1']      = monthly['rev_cr'].shift(1)
monthly['lag12']     = monthly['rev_cr'].shift(12)
monthly['roll3']     = monthly['rev_cr'].shift(1).rolling(3).mean()

monthly.dropna(inplace=True)

FEATURES = ['t','sin_12','cos_12','sin_6','cos_6',
            'is_festive','lag1','lag12','roll3']
X = monthly[FEATURES].values
y = monthly['rev_cr'].values

# ── Time series CV ───────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=4)
mapes = []
for tr, te in tscv.split(X):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X[tr])
    Xte = sc.transform(X[te])
    m   = Ridge(alpha=1.0)
    m.fit(Xtr, y[tr])
    mapes.append(mean_absolute_percentage_error(y[te], m.predict(Xte))*100)

print(f"Time-series CV MAPE: {np.mean(mapes):.2f}% ± {np.std(mapes):.2f}%")

# ── Final model ──────────────────────────────────────────
scaler = StandardScaler()
Xs     = scaler.fit_transform(X)
model  = Ridge(alpha=1.0)
model.fit(Xs, y)
y_pred = model.predict(Xs)

print(f"Train R²   : {r2_score(y, y_pred):.4f}")
print(f"Train MAPE : {mean_absolute_percentage_error(y, y_pred)*100:.2f}%")

# ── 6-month forecast ─────────────────────────────────────
last      = monthly.iloc[-1]
last_vals = monthly['rev_cr'].values

future_rows = []
for i in range(1, 7):
    ft = last['t'] + i
    fm = (last['ds'] + pd.DateOffset(months=i)).month
    fy = (last['ds'] + pd.DateOffset(months=i)).year
    fd = last['ds'] + pd.DateOffset(months=i)
    lag1  = last_vals[-1] if i == 1 else future_rows[-1]['rev_cr_pred']
    lag12 = monthly['rev_cr'].iloc[-(12-i+1)] if i <= 12 else lag1
    roll3 = np.mean([last_vals[-1], last_vals[-2], last_vals[-3]]) if i == 1 \
            else np.mean([r['rev_cr_pred'] for r in future_rows[-min(3,i):]])
    row = {
        't': ft, 'sin_12': np.sin(2*np.pi*fm/12),
        'cos_12': np.cos(2*np.pi*fm/12),
        'sin_6' : np.sin(2*np.pi*fm/6),
        'cos_6' : np.cos(2*np.pi*fm/6),
        'is_festive': int(fm in [10,11,12]),
        'lag1': lag1, 'lag12': lag12, 'roll3': roll3,
        'ds': fd, 'month_num': fm, 'year': fy
    }
    Xf  = scaler.transform([[row[f] for f in FEATURES]])
    row['rev_cr_pred'] = float(model.predict(Xf)[0])
    future_rows.append(row)

future_df = pd.DataFrame(future_rows)

# ── Save ─────────────────────────────────────────────────
joblib.dump(model,  '../outputs/forecast_model.pkl')
joblib.dump(scaler, '../outputs/forecast_scaler.pkl')
joblib.dump(FEATURES,'../outputs/forecast_features.pkl')

combined = pd.concat([
    monthly[['ds','rev_cr']].rename(columns={'rev_cr':'actual'}),
    future_df[['ds','rev_cr_pred']].rename(columns={'rev_cr_pred':'forecast'})
], ignore_index=True)
combined.to_csv('../outputs/forecast_output.csv', index=False)
print("Model & forecast saved.")

# ── Plot ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG)

# Top: full trend + forecast
ax = axes[0]
ax.set_facecolor(CARD)
ax.fill_between(monthly['ds'], monthly['rev_cr'],
                color=C[0], alpha=0.12)
ax.plot(monthly['ds'], monthly['rev_cr'],
        color=C[0], lw=2.5, label='Actual Revenue')
ax.plot(monthly['ds'], y_pred,
        color=C[1], lw=1.8, linestyle='--', label='Model Fit', alpha=0.85)

# confidence band for forecast
fc_vals  = future_df['rev_cr_pred'].values
fc_dates = future_df['ds']
std_err  = np.std(y - y_pred)
ax.fill_between(fc_dates,
                fc_vals - 1.96*std_err,
                fc_vals + 1.96*std_err,
                color=C[2], alpha=0.18, label='95% CI')
ax.plot(fc_dates, fc_vals, color=C[2], lw=2.5,
        linestyle='-', marker='o', markersize=6, label='Forecast')
ax.axvline(monthly['ds'].iloc[-1], color='#475569',
           linestyle=':', lw=1.5)
ax.text(monthly['ds'].iloc[-1], ax.get_ylim()[1]*0.95,
        ' Forecast →', color=MUTED, fontsize=9)
ax.set_facecolor(CARD)
ax.tick_params(colors=MUTED, labelsize=9)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.set_ylabel('Revenue (₹ Crore)', color=MUTED, fontsize=9)
ax.set_title('Sales Revenue Forecast — Next 6 Months',
             color=TEXT, fontsize=13, fontweight='bold', pad=12)
ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
ax.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.6)

# Bottom: forecast bar
ax2 = axes[1]
ax2.set_facecolor(CARD)
labels = future_df['ds'].dt.strftime('%b %Y').tolist()
bars   = ax2.bar(labels, fc_vals, color=C[2], alpha=0.85, zorder=3)
for b, v in zip(bars, fc_vals):
    ax2.text(b.get_x()+b.get_width()/2,
             b.get_height()+0.05,
             f'₹{v:.2f} Cr',
             ha='center', color=TEXT, fontsize=9, fontweight='bold')
ax2.tick_params(colors=MUTED, labelsize=9)
ax2.spines[['top','right','left','bottom']].set_visible(False)
ax2.set_ylabel('Forecasted Revenue (₹ Cr)', color=MUTED, fontsize=9)
ax2.set_title('Month-wise Revenue Forecast',
              color=TEXT, fontsize=12, fontweight='bold', pad=10)
ax2.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.6, zorder=0)

fig.suptitle('Sales Forecasting Analytics', color=TEXT,
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../outputs/07_forecast.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Forecast chart saved → outputs/07_forecast.png")
