import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import local package
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

import multielo

def build_website_views():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    website_dir = os.path.join(pkg_root, 'website')
    data_dir = os.path.join(script_dir, 'data')
    
    os.makedirs(website_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print("=== Pre-building Website Views and Interactive Visualizations ===")
    
    # Ensure ratings_3eloC_all_norm.csv and ratings_3eloC_spain.csv exist
    norm_csv_path = os.path.join(data_dir, 'ratings_3eloC_all_norm.csv')
    spain_csv_path = os.path.join(data_dir, 'ratings_3eloC_spain.csv')
    
    if not os.path.exists(norm_csv_path) or not os.path.exists(spain_csv_path):
        from run_compute_team import run_compute_team
        print("Pre-computing rating trajectories...")
        run_compute_team(team='spain', system='3eloC', normalize=False)
        run_compute_team(team='all', system='3eloC', normalize=True)

    # 1. Build index.qmd (Landing Page)
    df_norm = pd.read_csv(norm_csv_path)
    df_norm['date'] = pd.to_datetime(df_norm['date'])
    df_norm = df_norm[df_norm['date'] >= '1950-01-01']
    
    idx_max = df_norm.groupby('date')['elo'].idxmax()
    df_no1 = df_norm.loc[idx_max].sort_values('date').reset_index(drop=True)
    df_no1 = df_no1[df_no1['team'] != 'Tahiti']
    
    fig_no1 = px.scatter(
        df_no1,
        x='norm_def',
        y='norm_off',
        color='team',
        hover_data=['date', 'elo', 'norm_elo'],
        labels={'norm_def': 'Defensive Score (R^d / R^d_10th)', 'norm_off': 'Offensive Score (R^o / R^o_10th)', 'team': 'World #1 Nation'},
        title="World #1 Tactical Style Positions (1950–2026)"
    )
    fig_no1.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig_no1.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig_no1.update_layout(template="plotly_dark", height=580)
    
    plotly_html_no1 = fig_no1.to_html(full_html=False, include_plotlyjs='cdn')
    
    index_qmd = f"""---
title: "Multi-Dimensional Elo Ratings & Forecasting"
subtitle: "Interactive Data Platform for International Football Strength & Tactical Style"
format:
  html:
    page-layout: full
---

::: {{.hero-banner}}
# MultiElo Football

Multi-dimensional Elo rating architectures and 32 Poisson Generalized Linear Models ($M_{{01}}$–$M_{{32}}$) for national team football forecasting (1872–2026).

[View on GitHub](https://github.com/cesarrennocosta/multielo-football){{.btn .btn-primary .btn-lg role="button"}}
[Read Methodology](models.html){{.btn .btn-outline-light .btn-lg role="button"}}
:::

::: {{.row}}
::: {{.col-md-3}}
::: {{.card-metric}}
::: {{.metric-value}}
49,518
:::
::: {{.metric-label}}
Match Records Analyzed
:::
:::
:::

::: {{.col-md-3}}
::: {{.card-metric}}
::: {{.metric-value}}
32 GLMs
:::
::: {{.metric-label}}
Model Architecture Grid
:::
:::
:::

::: {{.col-md-3}}
::: {{.card-metric}}
::: {{.metric-value}}
3-Elo Complete
:::
::: {{.metric-label}}
Top Performing Architecture
:::
:::
:::

::: {{.col-md-3}}
::: {{.card-metric}}
::: {{.metric-value}}
-4.5%
:::
::: {{.metric-label}}
Loss Reduction vs Benchmark
:::
:::
:::
:::

---

## ⚡ Quickstart Python Package

Install the `multielo-football` library directly via `pip`:

```bash
pip install multielo-football
```

Compute ratings and predict match outcomes in Python:

```python
import multielo

# 1. Load Match Dataset
df = multielo.load_dataset()

# 2. Compute 3-Elo Complete Ratings
df_rated = multielo.compute_ratings(df, system='3elo-complete')

# 3. Predict Spain vs England (Neutral Venue)
spain_ratings = {{'elo': 2279.0, 'off': 2276.6, 'def': 2412.4}}
england_ratings = {{'elo': 2117.7, 'off': 2341.5, 'def': 2226.9}}

pred = multielo.predict(spain_ratings, england_ratings, model_specs='M32', is_neutral=True)

print(f"P(Spain Win)   : {{pred['p_win_a']*100:.1f}}%")
print(f"P(Draw)        : {{pred['p_draw']*100:.1f}}%")
print(f"P(England Win) : {{pred['p_win_b']*100:.1f}}%")
print(f"Most Likely Score: {{pred['most_likely_score'][0]}} - {{pred['most_likely_score'][1]}}")
```

---

## 📊 World #1 Tactical Style Space (Interactive)

{plotly_html_no1}
"""
    with open(os.path.join(website_dir, 'index.qmd'), 'w') as f:
        f.write(index_qmd)

    # 2. Build ratings.qmd
    df_spain = pd.read_csv(spain_csv_path)
    df_spain['date'] = pd.to_datetime(df_spain['date'])
    df_spain = df_spain[df_spain['date'] >= '2000-01-01'].sort_values('date')
    
    fig_spain = make_subplots(specs=[[{"secondary_y": True}]])
    fig_spain.add_trace(go.Scatter(x=df_spain['date'], y=df_spain['elo'], name="Overall Elo (R^e)", line=dict(color='#ef4444', width=3)), secondary_y=False)
    fig_spain.add_trace(go.Scatter(x=df_spain['date'], y=df_spain['elo_off'], name="Offensive Elo (R^o)", line=dict(color='#f59e0b', width=2, dash='dash')), secondary_y=True)
    fig_spain.add_trace(go.Scatter(x=df_spain['date'], y=df_spain['elo_def'], name="Defensive Elo (R^d)", line=dict(color='#6366f1', width=2, dash='dot')), secondary_y=True)
    
    fig_spain.update_layout(title="Spain Historical Elo Rating Dynamics (2000–2026)", template="plotly_dark", height=580, xaxis_title="Match Calendar Date")
    fig_spain.update_yaxes(title_text="Overall Rating Points (R^e)", secondary_y=False)
    fig_spain.update_yaxes(title_text="Style Rating Points (R^o, R^d)", secondary_y=True)
    
    plotly_html_spain = f"\n\n{fig_spain.to_html(full_html=False, include_plotlyjs='cdn')}\n\n"
    
    ratings_qmd = f"""---
title: "Interactive Team Trajectories Explorer"
subtitle: "Explore Historical Overall, Offensive, and Defensive Elo Evolution (2000–2026)"
format:
  html:
    page-layout: full
---

{plotly_html_spain}
"""
    with open(os.path.join(website_dir, 'ratings.qmd'), 'w') as f:
        f.write(ratings_qmd)

    # 3. Build style_space.qmd
    df_avg = df_norm.groupby('team')[['norm_def', 'norm_off', 'elo']].mean().reset_index()
    df_avg = df_avg[df_avg['elo'] > 1400].sort_values('elo', ascending=False).head(30)
    
    fig_style = px.scatter(
        df_avg,
        x='norm_def',
        y='norm_off',
        text='team',
        size='elo',
        color='elo',
        color_continuous_scale='Viridis',
        labels={'norm_def': 'Defensive Score (R^d / R^d_10th)', 'norm_off': 'Offensive Score (R^o / R^o_10th)'},
        title="Multi-Decade Average Style Profiles for Top 30 National Teams (1950–2026)"
    )
    fig_style.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig_style.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig_style.update_traces(textposition='top center')
    fig_style.update_layout(template="plotly_dark", height=620)
    
    plotly_html_style = fig_style.to_html(full_html=False, include_plotlyjs='cdn')
    
    style_space_qmd = f"""---
title: "Normalized Tactical Style Space"
subtitle: "Evaluating National Playing Philosophies in Non-Dimensional Coordinates"
format:
  html:
    page-layout: full
---

{plotly_html_style}
"""
    with open(os.path.join(website_dir, 'style_space.qmd'), 'w') as f:
        f.write(style_space_qmd)

    # 4. Build models.qmd
    models_rows = []
    for code, specs in multielo.GLM_TAXONOMY.items():
        models_rows.append(f"| **{code}** | {specs['dist']} | {specs['coupling']} | {specs['response']} | {'Yes' if specs['decay'] else 'No'} | {'Yes' if specs['competition'] else 'No'} |")
        
    models_table_md = "\n".join(models_rows)
    
    models_qmd = f"""---
title: "Poisson GLM Model Hierarchy (M01–M32)"
subtitle: "Architectural Design Grid & Parameter Complexity Specifications"
format:
  html:
    page-layout: full
---

We systematically evaluate a 5-dimensional binary feature grid of 32 GLM specifications ($M_{{01}}$ through $M_{{32}}$):

| Model Code | Distribution | Parameter Coupling | Rating Response | Temporal Decay (T) | Competition Weight (C) |
| :--- | :--- | :--- | :--- | :---: | :---: |
{models_table_md}
"""
    with open(os.path.join(website_dir, 'models.qmd'), 'w') as f:
        f.write(models_qmd)
        
    print("Successfully built pre-rendered website views and interactive graphics!")

if __name__ == '__main__':
    build_website_views()
