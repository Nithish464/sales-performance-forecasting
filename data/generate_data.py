"""
generate_data.py
Generates 3 years of realistic e-commerce sales data.
Run this first before anything else.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ── Config ──────────────────────────────────────────────
START = datetime(2022, 1, 1)
END   = datetime(2024, 12, 31)
DAYS  = (END - START).days + 1

regions   = ['North', 'South', 'East', 'West', 'Central']
products  = {
    'Electronics' : {'base': 18000, 'margin': 0.18},
    'Fashion'     : {'base':  2500, 'margin': 0.42},
    'Grocery'     : {'base':   650, 'margin': 0.12},
    'Beauty'      : {'base':  1800, 'margin': 0.55},
    'Sports'      : {'base':  3200, 'margin': 0.30},
    'Home & Kitchen':{'base': 4500, 'margin': 0.28},
}
channels  = ['Online', 'Retail Store', 'Mobile App', 'Marketplace']
salesreps = [f'SR{str(i).zfill(3)}' for i in range(1, 31)]

rows = []
date = START
while date <= END:
    month     = date.month
    weekday   = date.weekday()
    # seasonal factor
    seasonal  = 1 + 0.3 * np.sin((month - 3) * np.pi / 6)
    # weekend bump
    day_factor= 1.25 if weekday >= 5 else 1.0
    # festive spike (Oct-Dec)
    festive   = 1.4 if month in [10, 11, 12] else 1.0
    # year-on-year growth
    year_growth = 1 + 0.18 * (date.year - 2022)

    n_orders = int(np.random.poisson(120) * seasonal * day_factor * festive)

    for _ in range(n_orders):
        cat    = np.random.choice(list(products.keys()),
                                  p=[0.22,0.20,0.18,0.15,0.13,0.12])
        info   = products[cat]
        region = np.random.choice(regions, p=[0.25,0.22,0.20,0.18,0.15])
        ch     = np.random.choice(channels, p=[0.38,0.25,0.27,0.10])
        qty    = np.random.randint(1, 6)
        price  = info['base'] * np.random.uniform(0.85, 1.15) * year_growth
        revenue= round(price * qty, 2)
        cost   = round(revenue * (1 - info['margin']), 2)
        profit = round(revenue - cost, 2)
        rows.append({
            'date'        : date.strftime('%Y-%m-%d'),
            'region'      : region,
            'category'    : cat,
            'channel'     : ch,
            'sales_rep'   : np.random.choice(salesreps),
            'quantity'    : qty,
            'unit_price'  : round(price, 2),
            'revenue'     : revenue,
            'cost'        : cost,
            'profit'      : profit,
            'order_id'    : f'ORD{np.random.randint(100000,999999)}',
        })
    date += timedelta(days=1)

df = pd.DataFrame(rows)
df.to_csv('sales_data.csv', index=False)
print(f"✅ Dataset: {len(df):,} orders | Revenue: ₹{df['revenue'].sum()/1e7:.1f} Cr")
