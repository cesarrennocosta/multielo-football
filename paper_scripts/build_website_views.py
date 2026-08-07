import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import local package
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

import multielo

# Exact World Cup Victory Final Dates & Champions (1950-2026)
WC_VICTORIES = [
    ('Uruguay', '1950-07-16', "Uruguay '50"),
    ('Germany', '1954-07-04', "Germany '54"),
    ('Brazil', '1958-06-29', "Brazil '58"),
    ('Brazil', '1962-06-17', "Brazil '62"),
    ('England', '1966-07-30', "England '66"),
    ('Brazil', '1970-06-21', "Brazil '70"),
    ('Germany', '1974-07-07', "Germany '74"),
    ('Argentina', '1978-06-25', "Argentina '78"),
    ('Italy', '1982-07-11', "Italy '82"),
    ('Argentina', '1986-06-29', "Argentina '86"),
    ('Germany', '1990-07-08', "Germany '90"),
    ('Brazil', '1994-07-17', "Brazil '94"),
    ('France', '1998-07-12', "France '98"),
    ('Brazil', '2002-06-30', "Brazil '02"),
    ('Italy', '2006-07-09', "Italy '06"),
    ('Spain', '2010-07-11', "Spain '10"),
    ('Germany', '2014-07-13', "Germany '14"),
    ('France', '2018-07-15', "France '18"),
    ('Argentina', '2022-12-18', "Argentina '22")
]

TEAM_COLORS = {
    'Spain': '#dc2626',
    'Brazil': '#eab308',
    'Germany': '#475569',
    'Argentina': '#0891b2',
    'France': '#2563eb',
    'Italy': '#0284c7',
    'England': '#e11d48',
    'Netherlands': '#ea580c',
    'Uruguay': '#0284c7',
    'Portugal': '#059669',
    'Hungary': '#059669',
    'Netherlands': '#ea580c',
    'Belgium': '#b91c1c'
}

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

    df_norm = pd.read_csv(norm_csv_path)
    df_norm['date'] = pd.to_datetime(df_norm['date'])
    df_norm = df_norm[df_norm['date'] >= '1950-01-01']

    # 1. Build index.qmd (Landing Page)
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
[Explore World #1 Style Space](world_no1.html){{.btn .btn-outline-light .btn-lg role="button"}}
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
"""
    with open(os.path.join(website_dir, 'index.qmd'), 'w') as f:
        f.write(index_qmd)

    # 2. Build world_no1.qmd (World #1 Style Space - Downsampled, >= 6 Months Stints, Golden Stars)
    idx_max = df_norm.groupby('date')['elo'].idxmax()
    df_no1 = df_norm.loc[idx_max].sort_values('date').reset_index(drop=True)
    df_no1 = df_no1[df_no1['team'] != 'Tahiti']
    
    # Identify contiguous World #1 stints
    df_no1['team_change'] = (df_no1['team'] != df_no1['team'].shift(1)).astype(int)
    df_no1['stint_id'] = df_no1['team_change'].cumsum()
    
    stints = []
    for stint_id, group in df_no1.groupby('stint_id'):
        team = group['team'].iloc[0]
        start_date = group['date'].min()
        end_date = group['date'].max()
        duration_days = (end_date - start_date).days
        
        # Only keep nations holding #1 for at least 6 months (180 days)
        if duration_days >= 180:
            group = group.copy()
            group['year'] = group['date'].dt.year
            group['half'] = np.where(group['date'].dt.month <= 6, 1, 2)
            # Sample max 2 points per year (one per half-year)
            for (yr, hf), subg in group.groupby(['year', 'half']):
                stints.append(subg.iloc[len(subg)//2])
                
    df_no1_sampled = pd.DataFrame(stints).reset_index(drop=True)
    unique_teams = sorted(df_no1_sampled['team'].unique())
    
    fig_no1 = go.Figure()
    
    # Track traces per team for JS toggling
    team_trace_map = {}
    
    for i, t in enumerate(unique_teams):
        df_t = df_no1_sampled[df_no1_sampled['team'] == t]
        c = TEAM_COLORS.get(t, '#64748b')
        
        # Team scatter points
        fig_no1.add_trace(go.Scatter(
            x=df_t['norm_def'],
            y=df_t['norm_off'],
            mode='markers',
            name=t,
            marker=dict(size=10, color=c, opacity=0.9, line=dict(width=1, color='#1e293b')),
            customdata=np.stack((df_t['date'].dt.strftime('%Y-%m-%d'), df_t['elo'].round(1), df_t['norm_elo'].round(3)), axis=-1),
            hovertemplate="<b>" + t + "</b><br>Date: %{customdata[0]}<br>Defensive Score: %{x:.3f}<br>Offensive Score: %{y:.3f}<br>Elo Rating: %{customdata[1]}<extra></extra>"
        ))
        team_trace_map[t] = i

    # Identify World Cup Victory exact dates in df_norm
    wc_stars_x = []
    wc_stars_y = []
    wc_stars_text = []
    wc_stars_hover = []
    
    for tm, dt_str, label in WC_VICTORIES:
        dt_val = pd.to_datetime(dt_str)
        # Find closest match date for team
        df_tm = df_norm[(df_norm['team'] == tm) & (df_norm['date'] >= dt_val - pd.Timedelta(days=14)) & (df_norm['date'] <= dt_val + pd.Timedelta(days=14))]
        if not df_tm.empty:
            row_wc = df_tm.iloc[0]
            wc_stars_x.append(row_wc['norm_def'])
            wc_stars_y.append(row_wc['norm_off'])
            wc_stars_text.append(label)
            wc_stars_hover.append(f"⭐ <b>{label}</b><br>Date: {row_wc['date'].strftime('%Y-%m-%d')}<br>Defensive Score: {row_wc['norm_def']:.3f}<br>Offensive Score: {row_wc['norm_off']:.3f}")

    # Trace for World Cup Golden Stars (Always Top Layer)
    star_trace_idx = len(unique_teams)
    fig_no1.add_trace(go.Scatter(
        x=wc_stars_x,
        y=wc_stars_y,
        mode='markers+text',
        name="World Cup Winners (⭐)",
        text=wc_stars_text,
        textposition="top right",
        textfont=dict(size=11, color="#b45309", family="Inter, sans-serif"),
        marker=dict(symbol="star", size=17, color="#f59e0b", line=dict(width=1.5, color="#78350f")),
        hoverinfo="text",
        hovertext=wc_stars_hover
    ))

    fig_no1.add_hline(y=1.0, line_dash="dash", line_color="#cbd5e1")
    fig_no1.add_vline(x=1.0, line_dash="dash", line_color="#cbd5e1")

    fig_no1.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=720,
        showlegend=False,
        font=dict(color="#0f172a", family="Inter, sans-serif"),
        xaxis=dict(title="Defensive Score (R^d / R^d_10th)", gridcolor="#f1f5f9", zerolinecolor="#cbd5e1"),
        yaxis=dict(title="Offensive Score (R^o / R^o_10th)", gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    )

    no1_div_id = "world-no1-chart"
    plotly_inner_no1 = fig_no1.to_html(full_html=False, include_plotlyjs='cdn', div_id=no1_div_id)

    # Build Team Checkboxes (Soft clear transparent gray when off)
    no1_team_checkboxes = ""
    for i, t in enumerate(unique_teams):
        c = TEAM_COLORS.get(t, '#475569')
        no1_team_checkboxes += f"""
        <label style="color: {c}; font-weight: 600; cursor: pointer; background: #f8fafc; padding: 5px 10px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 0.9rem;">
          <input type="checkbox" id="chk-no1-{i}" checked onchange="updateNo1Chart()"> {t}
        </label>"""

    control_panel_no1 = f"""
<div class="team-selector-box" style="background: #f8fafc; padding: 16px; border-radius: 10px; margin-bottom: 18px; border: 1px solid #cbd5e1;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <div style="font-weight: 700; color: #0f172a; font-size: 1.05rem;">
      ⚽ Filter World #1 Nations (Off = Soft Clear Gray Cloud):
    </div>
    <label style="color: #b45309; font-weight: 700; cursor: pointer; background: #fef3c7; padding: 6px 14px; border-radius: 20px; border: 1px solid #fde68a;">
      <input type="checkbox" id="chk-wc-stars" checked onchange="updateNo1Chart()"> ⭐ Highlight World Cup Champions
    </label>
  </div>
  <div class="team-checkbox-grid" style="display: flex; flex-wrap: wrap; gap: 8px;">
    {no1_team_checkboxes}
  </div>
</div>

<script type="text/javascript">
var defaultTeamColors = {json.dumps([TEAM_COLORS.get(t, '#64748b') for t in unique_teams])};

function updateNo1Chart() {{
    var gd = document.getElementById('{no1_div_id}');
    if (!gd) return;
    
    var numTeams = {len(unique_teams)};
    var colorUpdates = [];
    var opacityUpdates = [];
    var sizeUpdates = [];
    
    for (var i = 0; i < numTeams; i++) {{
        var chk = document.getElementById('chk-no1-' + i);
        if (chk && chk.checked) {{
            colorUpdates.push(defaultTeamColors[i]);
            opacityUpdates.push(0.9);
            sizeUpdates.push(10);
        }} else {{
            // Soft clear transparent gray when off
            colorUpdates.push('rgba(226, 232, 240, 0.22)');
            opacityUpdates.push(0.12);
            sizeUpdates.push(6);
        }}
    }}
    
    // Update team trace colors & opacities
    for (var i = 0; i < numTeams; i++) {{
        Plotly.restyle(gd, {{
            'marker.color': colorUpdates[i],
            'marker.opacity': opacityUpdates[i],
            'marker.size': sizeUpdates[i]
        }}, [i]);
    }}
    
    // Toggle World Cup Star Markers
    var chkStars = document.getElementById('chk-wc-stars');
    var showStars = chkStars ? chkStars.checked : true;
    Plotly.restyle(gd, {{visible: showStars}}, [{star_trace_idx}]);
}}
</script>
"""

    plotly_full_no1 = f"\n\n```{{=html}}\n{control_panel_no1}\n{plotly_inner_no1}\n```\n\n"

    world_no1_qmd = f"""---
title: "World #1 Tactical Style Space"
subtitle: "Evaluating Historical #1 Ranked Teams & World Cup Champions (1950–2026)"
format:
  html:
    page-layout: full
---

This visualization plots the relative offensive ($R^o / R^o_{{10th}}$) and defensive ($R^d / R^d_{{10th}}$) coordinates of nations holding the **World #1 Elo Ranking** for at least 6 months (sampled at most 2 points per year for maximum responsiveness). 

Unchecking a team fades its points into a **soft clear transparent gray background cloud** so active teams stand out vividly. Exact **World Cup Victories** are marked with Golden Stars (⭐) and text labels.

{plotly_full_no1}
"""
    with open(os.path.join(website_dir, 'world_no1.qmd'), 'w') as f:
        f.write(world_no1_qmd)

    # 3. Build ratings.qmd (Multi-Team Trajectory Explorer - White Theme, Faded Gray Off-State, Straight Offense / Dashed Defense)
    teams_to_compute = ['Spain', 'Brazil', 'Germany', 'Argentina', 'Italy', 'France', 'England', 'Netherlands', 'Uruguay', 'Portugal']
    
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
        subplot_titles=("Top Panel: Overall Rating Points (R^e)", "Bottom Panel: Tactical Style Ratings — Offensive (R^o, Solid) & Defensive (R^d, Dashed)")
    )
    
    default_checked = ['Spain', 'Brazil']
    
    for i, t in enumerate(teams_to_compute):
        t_csv = os.path.join(data_dir, f'ratings_3eloC_{t.lower()}.csv')
        df_t = pd.read_csv(t_csv)
        df_t['date'] = pd.to_datetime(df_t['date'])
        df_t = df_t[df_t['date'] >= '1950-01-01'].sort_values('date')
        
        c = TEAM_COLORS.get(t, '#334155')
        is_vis = (t in default_checked)
        
        # Trace 0: Overall Elo (Solid line)
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
        fig_ratings.add_vline(
            x=wc_date,
            line_dash="dot",
            line_color="rgba(148, 163, 184, 0.45)",
            line_width=1.2
        )
        fig_ratings.add_annotation(
            x=wc_date, y=1.02, yref="y domain",
            text=label, showarrow=False,
            font=dict(size=9, color="#64748b"),
            row=1, col=1
        )

    ratings_div_id = "ratings-plotly-chart"
    
    fig_ratings.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=740,
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=40),
        font=dict(color="#0f172a", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
        xaxis2=dict(
            gridcolor="#e2e8f0", zerolinecolor="#cbd5e1",
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis2=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    )
    
    plotly_inner_ratings = fig_ratings.to_html(full_html=False, include_plotlyjs='cdn', div_id=ratings_div_id)
    
    team_checkboxes_html = ""
    for i, t in enumerate(teams_to_compute):
        c = TEAM_COLORS.get(t, '#334155')
        is_chk = "checked" if t in default_checked else ""
        team_checkboxes_html += f"""
        <label style="color: {c}; font-weight: 600; cursor: pointer; background: #f8fafc; padding: 6px 12px; border-radius: 6px; border: 1px solid #cbd5e1;">
          <input type="checkbox" id="chk-team-{i}" {is_chk} onchange="updateRatingsChart()"> {t}
        </label>"""

    control_panel_ratings = f"""
<div class="team-selector-box" style="background: #f1f5f9; padding: 18px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #cbd5e1;">
  <div style="font-weight: 600; color: #0f172a; margin-bottom: 10px; font-size: 1.05rem;">
    ⚽ Select Teams to Compare:
  </div>
  <div class="team-checkbox-grid" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px;">
    {team_checkboxes_html}
  </div>
  <hr style="border-color: #cbd5e1; margin: 12px 0;">
  <div style="font-weight: 600; color: #0f172a; margin-bottom: 10px; font-size: 1.05rem;">
    📊 Subplot Component Controls:
  </div>
  <div class="metric-checkbox-grid" style="display: flex; flex-wrap: wrap; gap: 18px;">
    <label style="color: #0f172a; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-elo" checked onchange="updateRatingsChart()"> Top Panel: Overall Rating ($R^e$)
    </label>
    <label style="color: #d97706; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-off" checked onchange="updateRatingsChart()"> Bottom Panel: Offensive Rating ($R^o$, Solid)
    </label>
    <label style="color: #2563eb; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-def" checked onchange="updateRatingsChart()"> Bottom Panel: Defensive Rating ($R^d$, Dashed)
    </label>
  </div>
</div>

<script type="text/javascript">
function updateRatingsChart() {{
    var gd = document.getElementById('{ratings_div_id}');
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

    plotly_full_ratings = f"\n\n```{{=html}}\n{control_panel_ratings}\n{plotly_inner_ratings}\n```\n\n"
    
    ratings_qmd = f"""---
title: "Interactive Team Trajectories Explorer"
subtitle: "Multi-Team Side-by-Side Comparison Suite & Time Range Filter (1950–2026)"
format:
  html:
    page-layout: full
---

Use the **Team Checkboxes** below to add or remove national teams dynamically. Overall Elo ($R^e$) and Tactical Style ($R^o, R^d$) are displayed on decoupled subplot panels with FIFA World Cup tournament markers (1950–2026).

{plotly_full_ratings}
"""
    with open(os.path.join(website_dir, 'ratings.qmd'), 'w') as f:
        f.write(ratings_qmd)

    # 4. Build style_space.qmd (White Theme)
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
    fig_style.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8")
    fig_style.add_vline(x=1.0, line_dash="dash", line_color="#94a3b8")
    fig_style.update_traces(textposition='top center')
    fig_style.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=640,
        font=dict(color="#0f172a", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#cbd5e1"),
        yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    )
    
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

    # 5. Build models.qmd
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
