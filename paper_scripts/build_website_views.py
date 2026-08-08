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
    'Brazil': '#d97706',
    'Germany': '#475569',
    'Argentina': '#0891b2',
    'France': '#2563eb',
    'Italy': '#0284c7',
    'England': '#e11d48',
    'Netherlands': '#ea580c',
    'Uruguay': '#0284c7',
    'Portugal': '#059669'
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
    
    # Calculate dataset metadata
    df_results = pd.read_csv(results_csv_path, low_memory=False)
    latest_match_date = pd.to_datetime(df_results['date']).max().strftime('%B %d, %Y')
    total_matches_count = len(df_results)

    # Calculate MoM Rating & Rank changes
    latest_dt = df_norm['date'].max()
    mom_dt = latest_dt - pd.Timedelta(days=30)

    df_latest = df_norm[df_norm['date'] == latest_dt].copy()
    df_mom_dates = df_norm[df_norm['date'] <= mom_dt]
    closest_mom_dt = df_mom_dates['date'].max()
    df_mom = df_norm[df_norm['date'] == closest_mom_dt].copy()

    df_latest['rank'] = df_latest['elo'].rank(ascending=False, method='min').astype(int)
    df_mom['rank_mom'] = df_mom['elo'].rank(ascending=False, method='min').astype(int)

    df_merged = pd.merge(
        df_latest[['team', 'elo', 'elo_off', 'elo_def', 'rank']],
        df_mom[['team', 'elo', 'elo_off', 'elo_def', 'rank_mom']],
        on='team',
        suffixes=('', '_mom')
    )

    df_merged['rank_change'] = df_merged['rank_mom'] - df_merged['rank']
    df_merged['elo_change'] = df_merged['elo'] - df_merged['elo_mom']
    df_merged['off_change'] = df_merged['elo_off'] - df_merged['elo_off_mom']
    df_merged['def_change'] = df_merged['elo_def'] - df_merged['elo_def_mom']

    df_merged = df_merged.sort_values('rank').reset_index(drop=True)

    # 1. Build index.qmd (Landing Page with 3 Top-10 Leaderboard Columns & Last Update metadata)
    top10_overall = df_merged.sort_values('rank').head(10)
    top10_offense = df_merged.sort_values('elo_off', ascending=False).head(10).reset_index(drop=True)
    top10_defense = df_merged.sort_values('elo_def', ascending=False).head(10).reset_index(drop=True)

    def format_rank_change(val):
        if val > 0:
            return f'<span style="color: #16a34a; font-weight: 600;">▲ {val}</span>'
        elif val < 0:
            return f'<span style="color: #dc2626; font-weight: 600;">▼ {abs(val)}</span>'
        else:
            return '<span style="color: #94a3b8;">-</span>'

    def format_pts_change(val):
        if val > 0:
            return f'<span style="color: #16a34a; font-size: 0.85rem;">(+{val:.1f})</span>'
        elif val < 0:
            return f'<span style="color: #dc2626; font-size: 0.85rem;">({val:.1f})</span>'
        else:
            return '<span style="color: #94a3b8; font-size: 0.85rem;">(0.0)</span>'

    # Build Top 10 Overall Table Rows
    rows_overall = ""
    for idx, r in top10_overall.iterrows():
        medal = "🥇 " if r['rank'] == 1 else ("🥈 " if r['rank'] == 2 else ("🥉 " if r['rank'] == 3 else f"{r['rank']}. "))
        rows_overall += f"""
        <tr>
          <td><strong>{medal}{r['team']}</strong></td>
          <td style="text-align: right;"><strong>{r['elo']:.1f}</strong> {format_pts_change(r['elo_change'])}</td>
          <td style="text-align: center;">{format_rank_change(r['rank_change'])}</td>
        </tr>"""

    # Build Top 10 Offensive Table Rows
    rows_offense = ""
    for idx, r in top10_offense.iterrows():
        rank_no = idx + 1
        medal = "🥇 " if rank_no == 1 else ("🥈 " if rank_no == 2 else ("🥉 " if rank_no == 3 else f"{rank_no}. "))
        rows_offense += f"""
        <tr>
          <td><strong>{medal}{r['team']}</strong></td>
          <td style="text-align: right;"><strong>{r['elo_off']:.1f}</strong> {format_pts_change(r['off_change'])}</td>
        </tr>"""

    # Build Top 10 Defensive Table Rows
    rows_defense = ""
    for idx, r in top10_defense.iterrows():
        rank_no = idx + 1
        medal = "🥇 " if rank_no == 1 else ("🥈 " if rank_no == 2 else ("🥉 " if rank_no == 3 else f"{rank_no}. "))
        rows_defense += f"""
        <tr>
          <td><strong>{medal}{r['team']}</strong></td>
          <td style="text-align: right;"><strong>{r['elo_def']:.1f}</strong> {format_pts_change(r['def_change'])}</td>
        </tr>"""

    leaderboards_html = f"""
<div style="background: #f8fafc; padding: 8px 16px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 24px; font-weight: 600; color: #475569; display: inline-block;">
  📅 <strong>Data Last Updated:</strong> {latest_match_date} &nbsp;|&nbsp; ⚽ <strong>Total Match Records:</strong> {total_matches_count:,}
</div>

::: {{.row}}
::: {{.col-md-4}}
<div class="card-metric" style="background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 18px;">
  <h4 style="color: #2563eb; font-weight: 700; border-bottom: 2px solid #2563eb; padding-bottom: 8px; margin-bottom: 12px;">
    🏆 Top 10 Overall Elo ($R^e$)
  </h4>
  <table class="table table-sm table-hover" style="font-size: 0.92rem; margin-bottom: 0;">
    <thead>
      <tr style="color: #64748b;">
        <th>Team</th>
        <th style="text-align: right;">Elo Pts</th>
        <th style="text-align: center;">MoM</th>
      </tr>
    </thead>
    <tbody>
      {rows_overall}
    </tbody>
  </table>
</div>
:::

::: {{.col-md-4}}
<div class="card-metric" style="background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 18px;">
  <h4 style="color: #d97706; font-weight: 700; border-bottom: 2px solid #d97706; padding-bottom: 8px; margin-bottom: 12px;">
    ⚔️ Top 10 Offensive ($R^o$)
  </h4>
  <table class="table table-sm table-hover" style="font-size: 0.92rem; margin-bottom: 0;">
    <thead>
      <tr style="color: #64748b;">
        <th>Team</th>
        <th style="text-align: right;">Offense Pts</th>
      </tr>
    </thead>
    <tbody>
      {rows_offense}
    </tbody>
  </table>
</div>
:::

::: {{.col-md-4}}
<div class="card-metric" style="background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 18px;">
  <h4 style="color: #059669; font-weight: 700; border-bottom: 2px solid #059669; padding-bottom: 8px; margin-bottom: 12px;">
    🛡️ Top 10 Defensive ($R^d$)
  </h4>
  <table class="table table-sm table-hover" style="font-size: 0.92rem; margin-bottom: 0;">
    <thead>
      <tr style="color: #64748b;">
        <th>Team</th>
        <th style="text-align: right;">Defense Pts</th>
      </tr>
    </thead>
    <tbody>
      {rows_defense}
    </tbody>
  </table>
</div>
:::
:::
"""

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

[View Global Rankings](rankings.html){{.btn .btn-primary .btn-lg role="button"}}
[Explore World #1 Style Space](world_no1.html){{.btn .btn-outline-light .btn-lg role="button"}}
:::

```{{=html}}
{leaderboards_html}
```

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

    # 2. Build rankings.qmd (Global National Team Rankings & MoM Changes for Overall, Offense, and Defense)
    df_latest['rank_elo'] = df_latest['elo'].rank(ascending=False, method='min').astype(int)
    df_latest['rank_off'] = df_latest['elo_off'].rank(ascending=False, method='min').astype(int)
    df_latest['rank_def'] = df_latest['elo_def'].rank(ascending=False, method='min').astype(int)

    df_mom['rank_elo_mom'] = df_mom['elo'].rank(ascending=False, method='min').astype(int)
    df_mom['rank_off_mom'] = df_mom['elo_off'].rank(ascending=False, method='min').astype(int)
    df_mom['rank_def_mom'] = df_mom['elo_def'].rank(ascending=False, method='min').astype(int)

    df_merged = pd.merge(
        df_latest[['team', 'elo', 'elo_off', 'elo_def', 'rank_elo', 'rank_off', 'rank_def']],
        df_mom[['team', 'elo', 'elo_off', 'elo_def', 'rank_elo_mom', 'rank_off_mom', 'rank_def_mom']],
        on='team',
        suffixes=('', '_mom')
    )

    df_merged['rank_change_elo'] = df_merged['rank_elo_mom'] - df_merged['rank_elo']
    df_merged['rank_change_off'] = df_merged['rank_off_mom'] - df_merged['rank_off']
    df_merged['rank_change_def'] = df_merged['rank_def_mom'] - df_merged['rank_def']

    df_merged['elo_change'] = df_merged['elo'] - df_merged['elo_mom']
    df_merged['off_change'] = df_merged['elo_off'] - df_merged['elo_off_mom']
    df_merged['def_change'] = df_merged['elo_def'] - df_merged['elo_def_mom']

    df_merged = df_merged.sort_values('rank_elo').reset_index(drop=True)

    full_rankings_rows = ""
    for idx, r in df_merged.iterrows():
        medal = "🥇 " if r['rank_elo'] == 1 else ("🥈 " if r['rank_elo'] == 2 else ("🥉 " if r['rank_elo'] == 3 else f"{r['rank_elo']}"))
        full_rankings_rows += f"""
        <tr>
          <td><strong>{medal}</strong></td>
          <td><strong>{r['team']}</strong></td>
          <td style="text-align: right;" data-sort="{r['elo']:.2f}"><strong>{r['elo']:.1f}</strong> {format_pts_change(r['elo_change'])}</td>
          <td style="text-align: center;" data-sort="{r['rank_change_elo']}">{format_rank_change(r['rank_change_elo'])}</td>
          <td style="text-align: right;" data-sort="{r['elo_off']:.2f}"><strong>{r['elo_off']:.1f}</strong> {format_pts_change(r['off_change'])}</td>
          <td style="text-align: center;" data-sort="{r['rank_change_off']}">{format_rank_change(r['rank_change_off'])}</td>
          <td style="text-align: right;" data-sort="{r['elo_def']:.2f}"><strong>{r['elo_def']:.1f}</strong> {format_pts_change(r['def_change'])}</td>
          <td style="text-align: center;" data-sort="{r['rank_change_def']}">{format_rank_change(r['rank_change_def'])}</td>
        </tr>"""

    rankings_qmd = f"""---
title: "Global National Team Rankings & MoM Dynamics"
subtitle: "Interactive Sorting by Overall ($R^e$), Offensive ($R^o$), or Defensive ($R^d$) Ratings with Rank Movements"
format:
  html:
    page-layout: full
---

This leaderboard ranks all active national teams evaluated under the top-performing **3-Elo Complete ($M_{{32}}$)** model architecture. Click the buttons below or click any table header to reorder teams by Overall, Offensive, or Defensive strength!

```{{=html}}
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
  <div style="background: #f8fafc; padding: 10px 18px; border-radius: 8px; border: 1px solid #cbd5e1; font-weight: 600; color: #475569;">
    📅 <strong>Data Last Updated:</strong> {latest_match_date} &nbsp;|&nbsp; ⚽ <strong>Total Match Records:</strong> {total_matches_count:,}
  </div>
  
  <div style="display: flex; gap: 10px;">
    <button onclick="sortTableByCol(2)" class="btn btn-sm btn-primary" style="font-weight: 600;">🏆 Sort by Overall ($R^e$)</button>
    <button onclick="sortTableByCol(4)" class="btn btn-sm btn-warning" style="font-weight: 600; color: #78350f;">⚔️ Sort by Offense ($R^o$)</button>
    <button onclick="sortTableByCol(6)" class="btn btn-sm btn-success" style="font-weight: 600;">🛡️ Sort by Defense ($R^d$)</button>
  </div>
</div>

<table id="rankingsTable" class="table table-striped table-hover" style="font-size: 0.95rem;">
  <thead>
    <tr style="background: #f1f5f9; color: #0f172a; cursor: pointer;">
      <th style="width: 70px;" onclick="sortTableByCol(0)">Rank</th>
      <th onclick="sortTableByCol(1)">National Team ⇕</th>
      <th style="text-align: right;" onclick="sortTableByCol(2)">Overall Elo ($R^e$) ⇕</th>
      <th style="text-align: center;" onclick="sortTableByCol(3)">$\Delta\text{{Rank}}_{{MoM}}$ ⇕</th>
      <th style="text-align: right;" onclick="sortTableByCol(4)">Offensive Elo ($R^o$) ⇕</th>
      <th style="text-align: center;" onclick="sortTableByCol(5)">$\Delta\text{{Rank}}_{{Off}}$ ⇕</th>
      <th style="text-align: right;" onclick="sortTableByCol(6)">Defensive Elo ($R^d$) ⇕</th>
      <th style="text-align: center;" onclick="sortTableByCol(7)">$\Delta\text{{Rank}}_{{Def}}$ ⇕</th>
    </tr>
  </thead>
  <tbody>
    {full_rankings_rows}
  </tbody>
</table>

<script type="text/javascript">
var currentSortCol = -1;
var sortAscending = false;

function sortTableByCol(colIdx) {{
    var table = document.getElementById("rankingsTable");
    if (!table) return;
    var tbody = table.getElementsByTagName("tbody")[0];
    var rows = Array.from(tbody.getElementsByTagName("tr"));
    
    if (currentSortCol === colIdx) {{
        sortAscending = !sortAscending;
    }} else {{
        currentSortCol = colIdx;
        sortAscending = (colIdx === 1); // Alphabetical asc for team name, numeric desc for ratings
    }}
    
    rows.sort(function(a, b) {{
        var cellA = a.getElementsByTagName("td")[colIdx];
        var cellB = b.getElementsByTagName("td")[colIdx];
        
        var valA = cellA.getAttribute("data-sort") || cellA.innerText.trim();
        var valB = cellB.getAttribute("data-sort") || cellB.innerText.trim();
        
        var numA = parseFloat(valA);
        var numB = parseFloat(valB);
        
        if (!isNaN(numA) && !isNaN(numB)) {{
            return sortAscending ? numA - numB : numB - numA;
        }} else {{
            return sortAscending ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }}
    }});
    
    for (var i = 0; i < rows.length; i++) {{
        tbody.appendChild(rows[i]);
    }}
}}
</script>
```
"""
    with open(os.path.join(website_dir, 'rankings.qmd'), 'w') as f:
        f.write(rankings_qmd)

    # 3. Build world_no1.qmd
    df_norm = df_norm[df_norm['date'] >= '1950-01-01']
    idx_max = df_norm.groupby('date')['elo'].idxmax()
    df_no1 = df_norm.loc[idx_max].sort_values('date').reset_index(drop=True)
    df_no1 = df_no1[df_no1['team'] != 'Tahiti']
    
    df_no1['team_change'] = (df_no1['team'] != df_no1['team'].shift(1)).astype(int)
    df_no1['stint_id'] = df_no1['team_change'].cumsum()
    
    stints = []
    for stint_id, group in df_no1.groupby('stint_id'):
        team = group['team'].iloc[0]
        start_date = group['date'].min()
        end_date = group['date'].max()
        duration_days = (end_date - start_date).days
        
        if duration_days >= 180:
            group = group.copy()
            group['year'] = group['date'].dt.year
            group['half'] = np.where(group['date'].dt.month <= 6, 1, 2)
            for (yr, hf), subg in group.groupby(['year', 'half']):
                stints.append(subg.iloc[len(subg)//2])
                
    df_no1_sampled = pd.DataFrame(stints).reset_index(drop=True)
    unique_teams = sorted(df_no1_sampled['team'].unique())
    
    fig_no1 = go.Figure()
    
    for i, t in enumerate(unique_teams):
        df_t = df_no1_sampled[df_no1_sampled['team'] == t]
        c = TEAM_COLORS.get(t, '#64748b')
        
        fig_no1.add_trace(go.Scatter(
            x=df_t['norm_def'],
            y=df_t['norm_off'],
            mode='markers',
            name=t,
            marker=dict(size=10, color=c, opacity=0.9, line=dict(width=1, color='#1e293b')),
            customdata=np.stack((df_t['date'].dt.strftime('%Y-%m-%d'), df_t['elo'].round(1), df_t['norm_elo'].round(3)), axis=-1),
            hovertemplate="<b>" + t + "</b><br>Date: %{customdata[0]}<br>Defensive Score: %{x:.3f}<br>Offensive Score: %{y:.3f}<br>Elo Rating: %{customdata[1]}<extra></extra>"
        ))

    wc_stars_x = []
    wc_stars_y = []
    wc_stars_text = []
    wc_stars_hover = []
    
    for tm, dt_str, label in WC_VICTORIES:
        dt_val = pd.to_datetime(dt_str)
        df_tm = df_norm[(df_norm['team'] == tm) & (df_norm['date'] >= dt_val - pd.Timedelta(days=14)) & (df_norm['date'] <= dt_val + pd.Timedelta(days=14))]
        if not df_tm.empty:
            row_wc = df_tm.iloc[0]
            wc_stars_x.append(row_wc['norm_def'])
            wc_stars_y.append(row_wc['norm_off'])
            wc_stars_text.append(label)
            wc_stars_hover.append(f"⭐ <b>{label}</b><br>Date: {row_wc['date'].strftime('%Y-%m-%d')}<br>Defensive Score: {row_wc['norm_def']:.3f}<br>Offensive Score: {row_wc['norm_off']:.3f}")

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
            colorUpdates.push('rgba(226, 232, 240, 0.22)');
            opacityUpdates.push(0.12);
            sizeUpdates.push(6);
        }}
    }}
    
    for (var i = 0; i < numTeams; i++) {{
        Plotly.restyle(gd, {{
            'marker.color': colorUpdates[i],
            'marker.opacity': opacityUpdates[i],
            'marker.size': sizeUpdates[i]
        }}, [i]);
    }}
    
    var chkStars = document.getElementById('chk-wc-stars');
    var showStars = chkStars ? chkStars.checked : true;
    
    if (showStars) {{
        Plotly.restyle(gd, {{
            'marker.color': '#f59e0b',
            'marker.line.color': '#78350f',
            'textfont.color': '#b45309',
            'marker.size': 17,
            'opacity': 1.0
        }}, [{star_trace_idx}]);
    }} else {{
        Plotly.restyle(gd, {{
            'marker.color': 'rgba(226, 232, 240, 0.25)',
            'marker.line.color': 'rgba(148, 163, 184, 0.3)',
            'textfont.color': 'rgba(148, 163, 184, 0.3)',
            'marker.size': 12,
            'opacity': 0.35
        }}, [{star_trace_idx}]);
    }}
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

    # 4. Build ratings.qmd & ratings_norm.qmd
    teams_to_compute = ['Spain', 'Brazil', 'Germany', 'Argentina', 'Italy', 'France', 'England', 'Netherlands', 'Uruguay', 'Portugal']
    
    from run_compute_team import run_compute_team
    for t in teams_to_compute:
        t_csv = os.path.join(data_dir, f'ratings_3eloC_{t.lower()}.csv')
        if not os.path.exists(t_csv):
            print(f"Pre-computing {t} rating trajectory...")
            run_compute_team(team=t, system='3eloC', normalize=False)

    def generate_trajectory_page(is_normalized=False):
        fig_ratings = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=("Top Panel: Overall Rating Points (R^e)" if not is_normalized else "Top Panel: Normalized Overall Rating (R^e / R^e_10th)",
                            "Bottom Panel: Tactical Style Ratings — Offensive (R^o, Solid) & Defensive (R^d, Dashed)" if not is_normalized else "Bottom Panel: Normalized Tactical Style Ratios — Offensive (R^o / R^o_10th) & Defensive (R^d / R^d_10th)")
        )
        
        default_checked = ['Spain', 'Brazil']
        
        for i, t in enumerate(teams_to_compute):
            if is_normalized:
                t_csv = os.path.join(data_dir, f'ratings_3eloC_{t.lower()}_norm.csv')
                if not os.path.exists(t_csv):
                    print(f"Pre-computing {t} normalized trajectory...")
                    run_compute_team(team=t, system='3eloC', normalize=True)
            else:
                t_csv = os.path.join(data_dir, f'ratings_3eloC_{t.lower()}.csv')
                if not os.path.exists(t_csv):
                    print(f"Pre-computing {t} raw trajectory...")
                    run_compute_team(team=t, system='3eloC', normalize=False)
                    
            df_t = pd.read_csv(t_csv)
            df_t['date'] = pd.to_datetime(df_t['date'])
            df_t = df_t[df_t['date'] >= '1950-01-01'].sort_values('date').reset_index(drop=True)
            
            # Balanced Downsampling: Keep ALL actual match dates + 1 anchor point every 7 days during idle periods
            if 'played_match_today' in df_t.columns:
                df_t = df_t[(df_t['played_match_today'] == True) | (df_t['date'].dt.day % 7 == 0)].reset_index(drop=True)
            else:
                df_t = df_t[df_t['date'].dt.day % 7 == 0].reset_index(drop=True)
                
            c = TEAM_COLORS.get(t, '#334155')
            is_vis = (t in default_checked)
            
            y_elo = df_t['norm_elo'] if (is_normalized and 'norm_elo' in df_t.columns) else df_t['elo']
            y_off = df_t['norm_off'] if (is_normalized and 'norm_off' in df_t.columns) else df_t['elo_off']
            y_def = df_t['norm_def'] if (is_normalized and 'norm_def' in df_t.columns) else df_t['elo_def']
            
            fig_ratings.add_trace(
                go.Scatter(x=df_t['date'], y=y_elo, name=f"{t} (R^e)", line=dict(color=c, width=3, dash='solid'), visible=is_vis),
                row=1, col=1
            )
            fig_ratings.add_trace(
                go.Scatter(x=df_t['date'], y=y_off, name=f"{t} Offense (R^o)", line=dict(color=c, width=2.2, dash='solid'), visible=is_vis),
                row=2, col=1
            )
            fig_ratings.add_trace(
                go.Scatter(x=df_t['date'], y=y_def, name=f"{t} Defense (R^d)", line=dict(color=c, width=2.2, dash='dash'), visible=is_vis),
                row=2, col=1
            )

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

        ratings_div_id = "ratings-norm-chart" if is_normalized else "ratings-plotly-chart"
        
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
              <input type="checkbox" id="chk-team-{i}{'-norm' if is_normalized else ''}" {is_chk} onchange="updateRatingsChart{'_norm' if is_normalized else ''}()"> {t}
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
      <input type="checkbox" id="chk-elo{'-norm' if is_normalized else ''}" checked onchange="updateRatingsChart{'_norm' if is_normalized else ''}()"> Top Panel: {'Normalized Rating ($R^e / R^e_{{10th}}$)' if is_normalized else 'Overall Rating ($R^e$)'}
    </label>
    <label style="color: #d97706; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-off{'-norm' if is_normalized else ''}" checked onchange="updateRatingsChart{'_norm' if is_normalized else ''}()"> Bottom Panel: {'Normalized Offense ($R^o / R^o_{{10th}}$, Solid)' if is_normalized else 'Offensive Rating ($R^o$, Solid)'}
    </label>
    <label style="color: #2563eb; font-weight: 600; cursor: pointer;">
      <input type="checkbox" id="chk-def{'-norm' if is_normalized else ''}" checked onchange="updateRatingsChart{'_norm' if is_normalized else ''}()"> Bottom Panel: {'Normalized Defense ($R^d / R^d_{{10th}}$, Dashed)' if is_normalized else 'Defensive Rating ($R^d$, Dashed)'}
    </label>
  </div>
</div>

<script type="text/javascript">
function updateRatingsChart{'_norm' if is_normalized else ''}() {{
    var gd = document.getElementById('{ratings_div_id}');
    if (!gd) return;
    
    var showElo = document.getElementById('chk-elo{'-norm' if is_normalized else ''}').checked;
    var showOff = document.getElementById('chk-off{'-norm' if is_normalized else ''}').checked;
    var showDef = document.getElementById('chk-def{'-norm' if is_normalized else ''}').checked;
    
    var visArray = [];
    var numTeams = {len(teams_to_compute)};
    
    for (var i = 0; i < numTeams; i++) {{
        var teamChk = document.getElementById('chk-team-' + i + '{'-norm' if is_normalized else ''}').checked;
        visArray.push(teamChk && showElo);
        visArray.push(teamChk && showOff);
        visArray.push(teamChk && showDef);
    }}
    
    Plotly.restyle(gd, {{visible: visArray}});
}}
</script>
"""

        plotly_full_ratings = f"\n\n```{{=html}}\n{control_panel_ratings}\n{plotly_inner_ratings}\n```\n\n"
        return plotly_full_ratings

    plotly_ratings_raw = generate_trajectory_page(is_normalized=False)
    ratings_qmd = f"""---
title: "Interactive Team Trajectories Explorer"
subtitle: "Multi-Team Side-by-Side Comparison Suite & Time Range Filter (1950–2026)"
format:
  html:
    page-layout: full
---

Use the **Team Checkboxes** below to add or remove national teams dynamically. Overall Elo ($R^e$) and Tactical Style ($R^o, R^d$) are displayed on decoupled subplot panels with FIFA World Cup tournament markers (1950–2026).

{plotly_ratings_raw}
"""
    with open(os.path.join(website_dir, 'ratings.qmd'), 'w') as f:
        f.write(ratings_qmd)

    plotly_ratings_norm = generate_trajectory_page(is_normalized=True)
    ratings_norm_qmd = f"""---
title: "Normalized Team Trajectories Explorer"
subtitle: "Non-Dimensional Rating Coordinates Relative to 10th-Place World Baseline (1950–2026)"
format:
  html:
    page-layout: full
---

This tab evaluates national team trajectories in **non-dimensional normalized coordinates** relative to the 10th-place World baseline ($R^e / R^e_{{10th}}$, $R^o / R^o_{{10th}}$, $R^d / R^d_{{10th}}$) on match dates.

{plotly_ratings_norm}
"""
    with open(os.path.join(website_dir, 'ratings_norm.qmd'), 'w') as f:
        f.write(ratings_norm_qmd)

    # 5. Build style_space.qmd
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

    # 6. Build models.qmd (Methods: 3eloC & M32 Equations, Parameters & GLM Grid)
    models_rows = []
    for code, specs in multielo.GLM_TAXONOMY.items():
        models_rows.append(f"| **{code}** | {specs['dist']} | {specs['coupling']} | {specs['response']} | {'Yes' if specs['decay'] else 'No'} | {'Yes' if specs['competition'] else 'No'} |")
        
    models_table_md = "\n".join(models_rows)
    
    models_qmd = f"""---
title: "Methods & Model Architecture Framework"
subtitle: "Mathematical Formulations, 3-Elo Complete Dynamics, Model M32, and Parameter Specifications"
format:
  html:
    page-layout: full
---

This section details the mathematical formulations and parameter estimates for the **3-Elo Complete** rating architecture and the **$M_{{32}}$ Bivariate Poisson GLM** forecasting model.

---

## 1. The 3-Elo Complete Rating Architecture (`3eloC`)

The **3-Elo Complete** system decomposes national team strength into three dynamic state variables evaluated chronologically across all match dates:
1. **Overall Rating ($R^e$)**: Aggregate winning capacity.
2. **Offensive Rating ($R^o$)**: Goal-scoring potency.
3. **Defensive Rating ($R^d$)**: Defensive resistance.

### Mathematical Rating Update Equations
Following match $k$ between Home Team $i$ and Away Team $j$ with match goal outcome $(g_i, g_j)$:

#### Overall Rating Update ($R^e$):
$$R^e_{{i, k+1}} = R^e_{{i, k}} + K_{{\\text{{base}}}} \\cdot W_c \\cdot \\gamma(m) \\cdot (S_i - E_i)$$
$$R^e_{{j, k+1}} = R^e_{{j, k}} + K_{{\\text{{base}}}} \\cdot W_c \\cdot \\gamma(m) \\cdot (S_j - E_j)$$

where expected win probability $E_i$ is computed via logistic sigmoid:
$$E_i = \\frac{{1}}{{1 + 10^{{-\\frac{{(R^e_{{i}} - R^e_{{j}}) + H_{{\\text{{overall}}}}}}{{D_{{\\text{{overall}}}}}}}}}}$$

#### Tactical Style Updates (Offensive $R^o$ and Defensive $R^d$):
$$R^o_{{i, k+1}} = R^o_{{i, k}} + K_{{\\text{{scale}}}} \\cdot K_{{\\text{{base}}}} \\cdot W_c \\cdot \\gamma(m) \\cdot (S^o_i - E^o_i)$$
$$R^d_{{i, k+1}} = R^d_{{i, k}} + K_{{\\text{{scale}}}} \\cdot K_{{\\text{{base}}}} \\cdot W_c \\cdot \\gamma(m) \\cdot (S^d_i - E^d_i)$$

#### Goal Difference Margin Scaling Factor $\\gamma(m)$:
$$\\gamma(m) = M_{{\\text{{overall}}}} \\cdot \\ln(1 + |g_i - g_j|) \\cdot \\left(\\frac{{a_{{\\text{{margin}}}}}}{{a_{{\\text{{margin}}}} + |g_i - g_j|}}\\right)$$

#### Tuned 3-Elo Complete Parameter Estimates:

| Parameter | Description | Value |
| :--- | :--- | :---: |
| $K_{{\\text{{base}}}}$ | Base rating volatility factor | `32.0537` |
| $H_{{\\text{{overall}}}}$ | Home venue advantage constant | `218.2097` |
| $D_{{\\text{{overall}}}}$ | Logistic scale divisor | `1267.5829` |
| $M_{{\\text{{overall}}}}$ | Goal difference margin multiplier | `2.5010` |
| $a_{{\\text{{margin}}}}$ | Margin saturation parameter | `4.3270` |
| $b_{{\\text{{margin}}}}$ | Secondary margin parameter | `3.6032` |
| $M_{{\\text{{style}}}}$ | Tactical style update factor | `1.0729` |
| $K_{{\\text{{scale}}}}$ | Style-to-overall $K$-ratio | `0.5521` |

---

## 2. Model $M_{{32}}$ Bivariate Poisson GLM & Score Distribution

Model **$M_{{32}}$** is the top-performing forecasting architecture. It calculates expected match scorelines using the tactical interaction between Team A's Offensive rating ($R^o_A$) and Team B's Defensive rating ($R^d_B$).

### Expected Goal Poisson Intensities ($\\lambda_A, \\lambda_B$):
For a match between Team A and Team B:

$$\\lambda_A = \\mu_{{\\text{{base}}}} \\cdot 10^{{\\frac{{(R^o_A - R^d_B) + H_{{\\text{{style}}}}}}{{D_{{\\text{{style}}}}}}}}$$

$$\\lambda_B = \\mu_{{\\text{{base}}}} \\cdot 10^{{\\frac{{(R^o_B - R^d_A) - H_{{\\text{{style}}}}}}{{D_{{\\text{{style}}}}}}}}$$

*(For neutral-venue matches, $H_{{\\text{{style}}}} = 0$).*

### Bivariate Dixon-Coles Goal Probability Matrix:
The joint probability of scoreline $(x, y)$ (Team A scores $x$ goals, Team B scores $y$ goals) is given by:

$$P(G_A = x, G_B = y) = \\tau(x, y) \\cdot \\left( \\frac{{\\lambda_A^x e^{{-\\lambda_A}}}}{{x!}} \\right) \\cdot \\left( \\frac{{\\lambda_B^y e^{{-\\lambda_B}}}}{{y!}} \\right)$$

where $\\tau(x, y)$ is the Dixon-Coles low-score dependency adjustment matrix:

$$\\tau(x, y) = \\begin{{cases}} 
1 - \\lambda_A \\lambda_B \\rho & \\text{{if }} x = 0, y = 0 \\\\ 
1 + \\lambda_A \\rho & \\text{{if }} x = 1, y = 0 \\\\ 
1 + \\lambda_B \\rho & \\text{{if }} x = 0, y = 1 \\\\ 
1 - \\rho & \\text{{if }} x = 1, y = 1 \\\\ 
1 & \\text{{otherwise}} 
\\end{{cases}}$$

### Outcome Probabilities:

$$P(\\text{{Win}}_A) = \\sum_{{x > y}} P(G_A = x, G_B = y)$$

$$P(\\text{{Draw}}) = \\sum_{{x = y}} P(G_A = x, G_B = y)$$

$$P(\\text{{Win}}_B) = \\sum_{{x < y}} P(G_A = x, G_B = y)$$

### Tuned Model $M_{{32}}$ Parameter Estimates:

| Parameter | Description | Value |
| :--- | :--- | :---: |
| $\\mu_{{\\text{{base}}}}$ | Baseline expected goal rate | `1.3500` |
| $D_{{\\text{{style}}}}$ | Tactical rating differential scale divisor | `974.5535` |
| $H_{{\\text{{style}}}}$ | Tactical home venue advantage | `60.9455` |
| $\\rho$ | Dixon-Coles low-score dependency factor | `-0.0821` |

---

## 3. Systematic 32 Poisson GLM Feature Grid ($M_{{01}}$–$M_{{32}}$)

We systematically evaluate a 5-dimensional binary feature grid of 32 GLM specifications ($M_{{01}}$ through $M_{{32}}$):

| Model Code | Distribution | Parameter Coupling | Rating Response | Temporal Decay (T) | Competition Weight (C) |
| :--- | :--- | :--- | :--- | :---: | :---: |
{models_table_md}
"""
    with open(os.path.join(website_dir, 'models.qmd'), 'w') as f:
        f.write(models_qmd)
        
    # 7. Copy Jupyter Notebook demo
    import shutil
    demo_nb_src = os.path.join(pkg_root, 'examples', 'demo_group_simulation.ipynb')
    demo_nb_dst = os.path.join(website_dir, 'demo_simulation.ipynb')
    if os.path.exists(demo_nb_src):
        shutil.copy(demo_nb_src, demo_nb_dst)
        print(f"Copied simulation demo notebook to: {demo_nb_dst}")
        
    print("Successfully built pre-rendered website views and interactive graphics!")

if __name__ == '__main__':
    build_website_views()
