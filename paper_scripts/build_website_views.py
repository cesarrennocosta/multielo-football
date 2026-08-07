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
    
    results_csv_path = os.path.join(data_dir, 'results.csv')
    spain_csv_path = os.path.join(data_dir, 'ratings_3eloC_spain.csv')
    norm_csv_path = os.path.join(data_dir, 'ratings_3eloC_all_norm.csv')
    
    if not os.path.exists(results_csv_path):
        from run_download_dataset import run_download_dataset
        print("Downloading match dataset...")
        run_download_dataset()

    if not os.path.exists(spain_csv_path):
        from run_compute_team import run_compute_team
        print("Pre-computing Spain rating trajectory...")
        run_compute_team(team='spain', system='3eloC', normalize=False)

    if not os.path.exists(norm_csv_path):
        from run_compute_team import run_compute_team
        print("Pre-computing normalized rating trajectories...")
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
    
    plotly_html_no1 = f"\n\n```{{=html}}\n{fig_no1.to_html(full_html=False, include_plotlyjs='cdn')}\n```\n\n"
    
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

    # 2. Build ratings.qmd (Multi-Team Trajectory Explorer)
    teams_to_compute = ['Spain', 'Brazil', 'Germany', 'Argentina', 'Italy', 'France', 'England', 'Netherlands', 'Uruguay', 'Portugal']
    team_colors = {
        'Spain': '#ef4444',
        'Brazil': '#eab308',
        'Germany': '#94a3b8',
        'Argentina': '#06b6d4',
        'France': '#3b82f6',
        'Italy': '#0284c7',
        'England': '#f43f5e',
        'Netherlands': '#f97316',
        'Uruguay': '#38bdf8',
        'Portugal': '#10b981'
    }
    
    from run_compute_team import run_compute_team
    for t in teams_to_compute:
        t_csv = os.path.join(data_dir, f'ratings_3eloC_{t.lower()}.csv')
        if not os.path.exists(t_csv):
            print(f"Pre-computing {t} rating trajectory...")
            run_compute_team(team=t, system='3eloC', normalize=False)

    fig_ratings = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Top Panel: Overall Rating Points (R^e)", "Bottom Panel: Tactical Style Ratings — Offensive (R^o) & Defensive (R^d)")
    )
    
    # Add traces for all 10 teams
    # Spain and Brazil checked by default
    default_checked = ['Spain', 'Brazil']
    
    for i, t in enumerate(teams_to_compute):
        t_csv = os.path.join(data_dir, f'ratings_3eloC_{t.lower()}.csv')
        df_t = pd.read_csv(t_csv)
        df_t['date'] = pd.to_datetime(df_t['date'])
        df_t = df_t[df_t['date'] >= '1950-01-01'].sort_values('date')
        
        c = team_colors[t]
        is_vis = (t in default_checked)
        
        # Trace 0: Overall Elo (Solid thick line)
        fig_ratings.add_trace(
            go.Scatter(x=df_t['date'], y=df_t['elo'], name=f"{t} (R^e)", line=dict(color=c, width=3, dash='solid'), visible=is_vis),
            row=1, col=1
        )
        # Trace 1: Offensive Elo (Straight/Solid line)
        fig_ratings.add_trace(
            go.Scatter(x=df_t['date'], y=df_t['elo_off'], name=f"{t} Offense (R^o)", line=dict(color=c, width=2.2, dash='solid'), visible=is_vis),
            row=2, col=1
        )
        # Trace 2: Defensive Elo (Dashed line)
        fig_ratings.add_trace(
            go.Scatter(x=df_t['date'], y=df_t['elo_def'], name=f"{t} Defense (R^d)", line=dict(color=c, width=2.2, dash='dash'), visible=is_vis),
            row=2, col=1
        )

    # Add FIFA World Cup Tournament Markers (1950–2026)
    world_cup_years = [
        (1950, "WC '50"), (1954, "WC '54"), (1958, "WC '58"), (1962, "WC '62"),
        (1966, "WC '66"), (1970, "WC '70"), (1974, "WC '74"), (1978, "WC '78"),
        (1982, "WC '82"), (1986, "WC '86"), (1990, "WC '90"), (1994, "WC '94"),
        (1998, "WC '98"), (2002, "WC '02"), (2006, "WC '06"), (2010, "WC '10"),
        (2014, "WC '14"), (2018, "WC '18"), (2022, "WC '22"), (2026, "WC '26")
    ]
    
    for yr, label in world_cup_years:
        wc_date = f"{yr}-06-15"
        # Add subtle vertical marker line on both subplots
        fig_ratings.add_vline(
            x=wc_date,
            line_dash="dot",
            line_color="rgba(148, 163, 184, 0.35)",
            line_width=1.2
        )
        # Add label annotation on top row
        fig_ratings.add_annotation(
            x=wc_date, y=1.02, yref="y domain",
            text=label, showarrow=False,
            font=dict(size=9, color="#94a3b8"),
            row=1, col=1
        )

    chart_div_id = "ratings-plotly-chart"
    
    fig_ratings.update_layout(
        template="plotly_dark",
        height=740,
        showlegend=False,  # Legend removed to prevent overlaying the figure
        margin=dict(t=50, b=40, l=60, r=40),
        xaxis2=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    # Generate Plotly HTML string
    plotly_inner_html = fig_ratings.to_html(full_html=False, include_plotlyjs='cdn', div_id=chart_div_id)
    
    # Build Checkbox Controls HTML + JS
    team_checkboxes_html = ""
    for i, t in enumerate(teams_to_compute):
        c = team_colors[t]
        is_chk = "checked" if t in default_checked else ""
        team_checkboxes_html += f"""
        <label style="color: {c}; font-weight: 600; cursor: pointer; background: #0f172a; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155;">
          <input type="checkbox" id="chk-team-{i}" {is_chk} onchange="updateRatingsChart()"> {t}
        </label>"""

    control_panel_html = f"""
<div class="team-selector-box" style="background: #1e293b; padding: 18px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #334155;">
  <div style="font-weight: 600; color: #f8fafc; margin-bottom: 10px; font-size: 1.05rem;">
    ⚽ Select Teams to Compare:
  </div>
  <div class="team-checkbox-grid" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px;">
    {team_checkboxes_html}
  </div>
  <hr style="border-color: #334155; margin: 12px 0;">
  <div style="font-weight: 600; color: #f8fafc; margin-bottom: 10px; font-size: 1.05rem;">
    📊 Subplot Component Controls:
  </div>
  <div class="metric-checkbox-grid" style="display: flex; flex-wrap: wrap; gap: 18px;">
    <label style="color: #f8fafc; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-elo" checked onchange="updateRatingsChart()"> Top Panel: Overall Rating ($R^e$)
    </label>
    <label style="color: #f59e0b; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-off" checked onchange="updateRatingsChart()"> Bottom Panel: Offensive Rating ($R^o$)
    </label>
    <label style="color: #3b82f6; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-def" checked onchange="updateRatingsChart()"> Bottom Panel: Defensive Rating ($R^d$)
    </label>
  </div>
</div>

<script type="text/javascript">
function updateRatingsChart() {{
    var gd = document.getElementById('{chart_div_id}');
    if (!gd) return;
    
    var showElo = document.getElementById('chk-elo').checked;
    var showOff = document.getElementById('chk-off').checked;
    var showDef = document.getElementById('chk-def').checked;
    
    var visArray = [];
    var numTeams = {len(teams_to_compute)};
    
    for (var i = 0; i < numTeams; i++) {{
        var teamChk = document.getElementById('chk-team-' + i).checked;
        visArray.push(teamChk && showElo);
        visArray.push(teamChk && showOff);
        visArray.push(teamChk && showDef);
    }}
    
    Plotly.restyle(gd, {{visible: visArray}});
}}
</script>
"""

    plotly_full_block = f"\n\n```{{=html}}\n{control_panel_html}\n{plotly_inner_html}\n```\n\n"
    
    ratings_qmd = f"""---
title: "Interactive Team Trajectories Explorer"
subtitle: "Multi-Team Side-by-Side Comparison Suite & Time Range Filter (1950–2026)"
format:
  html:
    page-layout: full
---

Use the **Team Checkboxes** below to add or remove national teams dynamically. Overall Elo ($R^e$) and Tactical Style ($R^o, R^d$) are displayed on decoupled subplot panels to prevent visual overlay.

{plotly_full_block}
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
    
    plotly_html_style = f"\n\n```{{=html}}\n{fig_style.to_html(full_html=False, include_plotlyjs='cdn')}\n```\n\n"
    
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
