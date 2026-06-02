"""
eda.py  –  Sales Performance EDA
Generates 8 premium publication-quality charts saved to outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)

# ── Palette ─────────────────────────────────────────────
C = ['#4F46E5','#06B6D4','#10B981','#F59E0B','#EF4444','#8B5CF6']
BG    = '#0F172A'
CARD  = '#1E293B'
TEXT  = '#F1F5F9'
MUTED = '#94A3B8'

def dark_fig(rows, cols, figsize, title=''):
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             facecolor=BG)
    fig.patch.set_facecolor(BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=16,
                     fontweight='bold', y=1.01)
    return fig, axes

def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.6)
    ax.grid(axis='x', visible=False)

# ── Load ────────────────────────────────────────────────
df = pd.read_csv('sales_data.csv', parse_dates=['date'])
df['month']   = df['date'].dt.to_period('M')
df['quarter'] = df['date'].dt.to_period('Q')
df['year']    = df['date'].dt.year
df['month_num']= df['date'].dt.month
df['week']    = df['date'].dt.isocalendar().week

print(f"Loaded {len(df):,} orders | {df['date'].min()} → {df['date'].max()}")

# ════════════════════════════════════════════════════════
# CHART 1 — Monthly Revenue Trend (3 years)
# ════════════════════════════════════════════════════════
monthly = df.groupby('month').agg(
    revenue=('revenue','sum'),
    profit =('profit','sum'),
    orders =('order_id','count')
).reset_index()
monthly['month_dt'] = monthly['month'].dt.to_timestamp()
monthly['rev_cr']   = monthly['revenue'] / 1e7
monthly['pro_cr']   = monthly['profit']  / 1e7

fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
ax.set_facecolor(CARD)
ax.fill_between(monthly['month_dt'], monthly['rev_cr'],
                color=C[0], alpha=0.15)
ax.plot(monthly['month_dt'], monthly['rev_cr'],
        color=C[0], lw=2.5, label='Revenue (₹ Cr)')
ax.fill_between(monthly['month_dt'], monthly['pro_cr'],
                color=C[2], alpha=0.15)
ax.plot(monthly['month_dt'], monthly['pro_cr'],
        color=C[2], lw=2, linestyle='--', label='Profit (₹ Cr)')
style_ax(ax, ylabel='₹ Crore', title='Monthly Revenue & Profit Trend (2022–2024)')
ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9, framealpha=0.8)
plt.tight_layout()
plt.savefig('../outputs/01_monthly_trend.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("✅ Chart 1 saved")

# ════════════════════════════════════════════════════════
# CHART 2 — Revenue by Region (grouped bar, YoY)
# ════════════════════════════════════════════════════════
reg_year = df.groupby(['year','region'])['revenue'].sum().unstack()
years    = reg_year.index.tolist()
regions  = reg_year.columns.tolist()
x        = np.arange(len(regions))
width    = 0.25

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(CARD)
for i, yr in enumerate(years):
    vals = reg_year.loc[yr].values / 1e7
    bars = ax.bar(x + i*width, vals, width=width*0.85,
                  color=C[i], label=str(yr), zorder=3)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height()+0.2, f'{b.get_height():.1f}',
                ha='center', va='bottom', color=MUTED, fontsize=7.5)
ax.set_xticks(x + width)
ax.set_xticklabels(regions, color=TEXT)
style_ax(ax, ylabel='Revenue (₹ Cr)',
         title='Year-on-Year Revenue by Region')
ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
ax.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig('../outputs/02_region_yoy.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("✅ Chart 2 saved")

# ════════════════════════════════════════════════════════
# CHART 3 — Category Performance (Revenue + Profit Margin)
# ════════════════════════════════════════════════════════
cat_perf = df.groupby('category').agg(
    revenue=('revenue','sum'),
    profit =('profit','sum')
).reset_index()
cat_perf['margin_pct'] = cat_perf['profit'] / cat_perf['revenue'] * 100
cat_perf = cat_perf.sort_values('revenue', ascending=True)

fig, ax1 = plt.subplots(figsize=(11, 5), facecolor=BG)
ax1.set_facecolor(CARD)
bars = ax1.barh(cat_perf['category'],
                cat_perf['revenue']/1e7,
                color=C[:len(cat_perf)], alpha=0.85, zorder=3)
for b in bars:
    ax1.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
             f'₹{b.get_width():.1f} Cr',
             va='center', color=MUTED, fontsize=8.5)
ax2 = ax1.twiny()
ax2.set_facecolor(CARD)
ax2.plot(cat_perf['margin_pct'], cat_perf['category'],
         'o--', color=C[3], lw=2, markersize=8, label='Margin %')
ax2.tick_params(colors=MUTED)
ax2.spines[['top','right','left','bottom']].set_visible(False)
ax2.set_xlabel('Profit Margin (%)', color=MUTED, fontsize=9)
style_ax(ax1, xlabel='Revenue (₹ Cr)',
         title='Category Revenue vs Profit Margin')
ax1.tick_params(axis='y', colors=TEXT)
ax1.grid(axis='x', color='#334155', linewidth=0.5, alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig('../outputs/03_category_perf.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("✅ Chart 3 saved")

# ════════════════════════════════════════════════════════
# CHART 4 — Channel Mix (Donut) + Monthly Channel Trend
# ════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
ch_rev = df.groupby('channel')['revenue'].sum()
wedges, texts, autotexts = ax1.pie(
    ch_rev.values, labels=ch_rev.index, autopct='%1.1f%%',
    colors=C[:len(ch_rev)], startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
    textprops=dict(color=TEXT, fontsize=9))
for at in autotexts:
    at.set_color(BG); at.set_fontsize(8.5); at.set_fontweight('bold')
ax1.set_facecolor(BG)
ax1.set_title('Channel Revenue Mix', color=TEXT,
              fontsize=11, fontweight='bold')

ch_month = df.groupby(['month','channel'])['revenue'].sum().unstack().fillna(0)
ch_month.index = ch_month.index.to_timestamp()
ax2.set_facecolor(CARD)
for i, ch in enumerate(ch_month.columns):
    ax2.plot(ch_month.index, ch_month[ch]/1e7,
             color=C[i], lw=2, label=ch)
style_ax(ax2, ylabel='Revenue (₹ Cr)', title='Monthly Revenue by Channel')
ax2.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig('../outputs/04_channel_mix.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("✅ Chart 4 saved")

# ════════════════════════════════════════════════════════
# CHART 5 — Heatmap: Revenue by Month × Region
# ════════════════════════════════════════════════════════
pivot = df.groupby(['month_num','region'])['revenue'].sum().unstack()
pivot.index = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
ax.set_facecolor(CARD)
im = ax.imshow(pivot.values / 1e7, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, color=TEXT, fontsize=9)
ax.set_yticks(range(12))
ax.set_yticklabels(pivot.index, color=TEXT, fontsize=9)
for i in range(12):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f'₹{pivot.values[i,j]/1e7:.1f}',
                ha='center', va='center', color='black',
                fontsize=7.5, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors=MUTED)
cbar.set_label('Revenue (₹ Cr)', color=MUTED, fontsize=9)
ax.set_title('Revenue Heatmap — Month × Region (₹ Cr)',
             color=TEXT, fontsize=11, fontweight='bold', pad=12)
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/05_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("✅ Chart 5 saved")

# ════════════════════════════════════════════════════════
# CHART 6 — Quarterly KPI Summary (4-panel)
# ════════════════════════════════════════════════════════
q_data = df.groupby('quarter').agg(
    revenue =('revenue','sum'),
    profit  =('profit','sum'),
    orders  =('order_id','count'),
    avg_ord =('revenue','mean')
).reset_index()
q_data['q_str']   = q_data['quarter'].astype(str)
q_data['margin']  = q_data['profit'] / q_data['revenue'] * 100

fig = plt.figure(figsize=(14, 8), facecolor=BG)
gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
panels = [
    (0,0, 'revenue',  'Revenue (₹ Cr)',    C[0], lambda v: v/1e7),
    (0,1, 'profit',   'Profit (₹ Cr)',     C[2], lambda v: v/1e7),
    (1,0, 'orders',   'Total Orders',      C[1], lambda v: v/1000),
    (1,1, 'margin',   'Profit Margin (%)', C[3], lambda v: v),
]
ylabels = ['₹ Crore','₹ Crore','Orders (K)','Margin %']
for idx,(r,c,col,lbl,clr,fn) in enumerate(panels):
    ax = fig.add_subplot(gs[r, c])
    ax.set_facecolor(CARD)
    vals = fn(q_data[col].values)
    bars = ax.bar(q_data['q_str'], vals, color=clr, alpha=0.85, zorder=3,
                  width=0.6)
    ax.plot(q_data['q_str'], vals, 'o-', color='white', lw=1.5,
            markersize=5, zorder=4)
    style_ax(ax, ylabel=ylabels[idx], title=lbl)
    ax.tick_params(axis='x', rotation=45, labelsize=7, colors=MUTED)
    ax.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.6, zorder=0)
fig.suptitle('Quarterly KPI Dashboard', color=TEXT,
             fontsize=15, fontweight='bold')
plt.savefig('../outputs/06_quarterly_kpi.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("✅ Chart 6 saved")

print("\n🎉 All 6 EDA charts saved to outputs/")
