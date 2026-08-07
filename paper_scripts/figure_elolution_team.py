import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects

# Configure Matplotlib vector text formatting
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Import local runner functions
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from run_compute_team import run_compute_team
import multielo

# Confederation Lookup Dictionary
CONFEDERATIONS = {
    'UEFA': ['Spain', 'Germany', 'Italy', 'France', 'England', 'Portugal', 'Netherlands', 'Belgium', 'Croatia', 'Denmark', 'Switzerland', 'Austria', 'Hungary', 'Czech Republic', 'Czechia', 'Sweden', 'Norway', 'Poland', 'Scotland', 'Wales', 'Serbia', 'Turkey', 'Greece', 'Romania'],
    'CONMEBOL': ['Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Ecuador', 'Chile', 'Peru', 'Paraguay', 'Venezuela', 'Bolivia'],
    'CAF': ['Nigeria', 'Morocco', 'Senegal', 'Egypt', 'Algeria', 'Cameroon', 'Ivory Coast', 'Ghana', 'Tunisia', 'Mali', 'South Africa'],
    'CONCACAF': ['Mexico', 'United States', 'USA', 'Canada', 'Costa Rica', 'Panama', 'Jamaica', 'Honduras'],
    'AFC': ['Japan', 'South Korea', 'Iran', 'Australia', 'Saudi Arabia', 'Qatar', 'Iraq', 'Uzbekistan']
}

# Tournament Ranges (Start Date, End Date, Title, Winner)
TOURNAMENTS = [
    # World Cups
    {'name': 'WC', 'title': '1950 World Cup', 'start': '1950-06-24', 'end': '1950-07-16', 'winner': 'Uruguay'},
    {'name': 'WC', 'title': '1954 World Cup', 'start': '1954-06-16', 'end': '1954-07-04', 'winner': 'Germany'},
    {'name': 'WC', 'title': '1958 World Cup', 'start': '1958-06-08', 'end': '1958-06-29', 'winner': 'Brazil'},
    {'name': 'WC', 'title': '1962 World Cup', 'start': '1962-05-30', 'end': '1962-06-17', 'winner': 'Brazil'},
    {'name': 'WC', 'title': '1966 World Cup', 'start': '1966-07-11', 'end': '1966-07-30', 'winner': 'England'},
    {'name': 'WC', 'title': '1970 World Cup', 'start': '1970-05-31', 'end': '1970-06-21', 'winner': 'Brazil'},
    {'name': 'WC', 'title': '1974 World Cup', 'start': '1974-06-13', 'end': '1974-07-07', 'winner': 'Germany'},
    {'name': 'WC', 'title': '1978 World Cup', 'start': '1978-06-01', 'end': '1978-06-25', 'winner': 'Argentina'},
    {'name': 'WC', 'title': '1982 World Cup', 'start': '1982-06-13', 'end': '1982-07-11', 'winner': 'Italy'},
    {'name': 'WC', 'title': '1986 World Cup', 'start': '1986-05-31', 'end': '1986-06-29', 'winner': 'Argentina'},
    {'name': 'WC', 'title': '1990 World Cup', 'start': '1990-06-08', 'end': '1990-07-08', 'winner': 'Germany'},
    {'name': 'WC', 'title': '1994 World Cup', 'start': '1994-06-17', 'end': '1994-07-17', 'winner': 'Brazil'},
    {'name': 'WC', 'title': '1998 World Cup', 'start': '1998-06-10', 'end': '1998-07-12', 'winner': 'France'},
    {'name': 'WC', 'title': '2002 World Cup', 'start': '2002-05-31', 'end': '2002-06-30', 'winner': 'Brazil'},
    {'name': 'WC', 'title': '2006 World Cup', 'start': '2006-06-09', 'end': '2006-07-09', 'winner': 'Italy'},
    {'name': 'WC', 'title': '2010 World Cup', 'start': '2010-06-11', 'end': '2010-07-11', 'winner': 'Spain'},
    {'name': 'WC', 'title': '2014 World Cup', 'start': '2014-06-12', 'end': '2014-07-13', 'winner': 'Germany'},
    {'name': 'WC', 'title': '2018 World Cup', 'start': '2018-06-14', 'end': '2018-07-15', 'winner': 'France'},
    {'name': 'WC', 'title': '2022 World Cup', 'start': '2022-11-20', 'end': '2022-12-18', 'winner': 'Argentina'},
    {'name': 'WC', 'title': '2026 World Cup', 'start': '2026-06-11', 'end': '2026-07-19', 'winner': 'Spain'},

    # Euro Cups
    {'name': 'EURO', 'title': 'Euro 1996', 'start': '1996-06-08', 'end': '1996-06-30', 'winner': 'Germany'},
    {'name': 'EURO', 'title': 'Euro 2000', 'start': '2000-06-10', 'end': '2000-07-02', 'winner': 'France'},
    {'name': 'EURO', 'title': 'Euro 2004', 'start': '2004-06-12', 'end': '2004-07-04', 'winner': 'Greece'},
    {'name': 'EURO', 'title': 'Euro 2008', 'start': '2008-06-07', 'end': '2008-06-29', 'winner': 'Spain'},
    {'name': 'EURO', 'title': 'Euro 2012', 'start': '2012-06-08', 'end': '2012-07-01', 'winner': 'Spain'},
    {'name': 'EURO', 'title': 'Euro 2016', 'start': '2016-06-10', 'end': '2016-07-10', 'winner': 'Portugal'},
    {'name': 'EURO', 'title': 'Euro 2020', 'start': '2021-06-11', 'end': '2021-07-11', 'winner': 'Italy'},
    {'name': 'EURO', 'title': 'Euro 2024', 'start': '2024-06-14', 'end': '2024-07-14', 'winner': 'Spain'},

    # Copa América
    {'name': 'COPA', 'title': 'Copa América 2004', 'start': '2004-07-06', 'end': '2004-07-25', 'winner': 'Brazil'},
    {'name': 'COPA', 'title': 'Copa América 2007', 'start': '2007-06-26', 'end': '2007-07-15', 'winner': 'Brazil'},
    {'name': 'COPA', 'title': 'Copa América 2011', 'start': '2011-07-01', 'end': '2011-07-24', 'winner': 'Uruguay'},
    {'name': 'COPA', 'title': 'Copa América 2015', 'start': '2015-06-11', 'end': '2015-07-04', 'winner': 'Chile'},
    {'name': 'COPA', 'title': 'Copa América 2016', 'start': '2016-06-03', 'end': '2016-06-26', 'winner': 'Chile'},
    {'name': 'COPA', 'title': 'Copa América 2019', 'start': '2019-06-14', 'end': '2019-07-07', 'winner': 'Brazil'},
    {'name': 'COPA', 'title': 'Copa América 2021', 'start': '2021-06-13', 'end': '2021-07-10', 'winner': 'Argentina'},
    {'name': 'COPA', 'title': 'Copa América 2024', 'start': '2024-06-20', 'end': '2024-07-14', 'winner': 'Argentina'},

    # AFCON
    {'name': 'AFCON', 'title': 'AFCON 2019', 'start': '2019-06-21', 'end': '2019-07-19', 'winner': 'Algeria'},
    {'name': 'AFCON', 'title': 'AFCON 2021', 'start': '2022-01-09', 'end': '2022-02-06', 'winner': 'Senegal'},
    {'name': 'AFCON', 'title': 'AFCON 2023', 'start': '2024-01-13', 'end': '2024-02-11', 'winner': 'Ivory Coast'},

    # Gold Cup
    {'name': 'GOLD', 'title': 'Gold Cup 2019', 'start': '2019-06-15', 'end': '2019-07-07', 'winner': 'Mexico'},
    {'name': 'GOLD', 'title': 'Gold Cup 2021', 'start': '2021-07-10', 'end': '2021-08-01', 'winner': 'United States'},
    {'name': 'GOLD', 'title': 'Gold Cup 2023', 'start': '2023-06-24', 'end': '2023-07-16', 'winner': 'Mexico'},

    # Asian Cup
    {'name': 'ASIA', 'title': 'Asian Cup 2019', 'start': '2019-01-05', 'end': '2019-02-01', 'winner': 'Qatar'},
    {'name': 'ASIA', 'title': 'Asian Cup 2023', 'start': '2024-01-12', 'end': '2024-02-10', 'winner': 'Qatar'},
]


def parse_date_string(date_str, is_end=False):
    if not date_str:
        return None
    s = str(date_str).strip()
    parts = s.split('-')
    
    # Case 1: Year only (e.g. 2006)
    if len(parts) == 1 and parts[0].isdigit():
        yr = int(parts[0])
        return pd.to_datetime(f"{yr}-12-31" if is_end else f"{yr}-01-01")
        
    # Case 2: Month-Year or Year-Month (e.g. 06-2006 or 2006-06)
    elif len(parts) == 2:
        p1, p2 = parts[0], parts[1]
        if len(p1) == 4 and p1.isdigit():
            yr, mo = int(p1), int(p2)
        else:
            mo, yr = int(p1), int(p2)
        if is_end:
            last_day = pd.Period(f"{yr}-{mo:02d}").days_in_month
            return pd.to_datetime(f"{yr}-{mo:02d}-{last_day:02d}")
        else:
            return pd.to_datetime(f"{yr}-{mo:02d}-01")
            
    # Case 3: Day-Month-Year or Year-Month-Day (e.g. 01-06-2006 or 2006-06-01)
    elif len(parts) == 3:
        if len(parts[0]) == 4:
            return pd.to_datetime(f"{parts[0]}-{parts[1]}-{parts[2]}")
        else:
            return pd.to_datetime(f"{parts[2]}-{parts[1]}-{parts[0]}")
            
    return pd.to_datetime(s)


def get_confederation(team_name):
    t_lower = str(team_name).lower().strip()
    for conf, c_teams in CONFEDERATIONS.items():
        if any(c.lower() == t_lower for c in c_teams):
            return conf
    return 'UEFA' # Default fallback


def figure_evolution_team(team, system='3eloC', start_date=None, stop_date=None, normalized=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Parse start and stop dates
    dt_start = parse_date_string(start_date, is_end=False) if start_date else pd.to_datetime('2006-01-01')
    dt_stop = parse_date_string(stop_date, is_end=True) if stop_date else pd.to_datetime('2026-07-31')
    
    # Ensure computed CSV file exists
    csv_filename = f"ratings_{system}_{team.lower()}{'_norm' if normalized else ''}.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"Computed ratings file {csv_filename} not found. Triggering run_compute_team.py...")
        run_compute_team(team=team, system=system, normalize=normalized)
        
    df_team = pd.read_csv(csv_path)
    df_team['date'] = pd.to_datetime(df_team['date'])
    
    # Filter date range
    df_team = df_team[(df_team['date'] >= dt_start) & (df_team['date'] <= dt_stop)].sort_values('date').reset_index(drop=True)
    
    if len(df_team) == 0:
        raise ValueError(f"No rating data found for team '{team}' in date range {dt_start.strftime('%Y-%m-%d')} to {dt_stop.strftime('%Y-%m-%d')}.")

    exact_team_name = df_team['team'].iloc[0]
    confed = get_confederation(exact_team_name)
    
    print(f"=== Plotting Rating Evolution for {exact_team_name} ===")
    print(f"Confederation: {confed} | Model: {system} | Normalized: {normalized}")
    print(f"Date Window  : {dt_start.strftime('%Y-%m-%d')} to {dt_stop.strftime('%Y-%m-%d')}")

    # Setup Plot Figure
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    
    has_style = ('elo_off' in df_team.columns) or ('norm_off' in df_team.columns)
    
    if normalized:
        # Single Y-axis for normalized ratings
        col_elo = 'norm_elo' if 'norm_elo' in df_team.columns else 'elo'
        col_off = 'norm_off' if 'norm_off' in df_team.columns else 'elo_off'
        col_def = 'norm_def' if 'norm_def' in df_team.columns else 'elo_def'
        
        ax1.plot(df_team['date'], df_team[col_elo], label='Overall (Relative)', color='#d95f02', linewidth=2.5, drawstyle='steps-post')
        if has_style:
            ax1.plot(df_team['date'], df_team[col_off], label='Offensive (Relative)', color='#e6ab02', linewidth=2.0, linestyle='--', drawstyle='steps-post')
            ax1.plot(df_team['date'], df_team[col_def], label='Defensive (Relative)', color='#7570b3', linewidth=2.0, linestyle=':', drawstyle='steps-post')
            
        ax1.axhline(1.0, color='#666666', linestyle='--', linewidth=1.2, label='Top 10 Baseline (1.0)')
        ax1.set_ylabel('Normalized Rating Score ($R / R_{10\\mathrm{th}}$)', fontsize=11, fontweight='bold', labelpad=8)
        
    else:
        # Un-normalized raw rating scale
        if has_style:
            ax2 = ax1.twinx()
            
            l1 = ax1.plot(df_team['date'], df_team['elo'], label='Overall Elo ($R^e$)', color='#d95f02', linewidth=2.8, drawstyle='steps-post')
            l2 = ax2.plot(df_team['date'], df_team['elo_off'], label='Offensive Elo ($R^o$)', color='#e6ab02', linewidth=2.0, linestyle='--', drawstyle='steps-post')
            l3 = ax2.plot(df_team['date'], df_team['elo_def'], label='Defensive Elo ($R^d$)', color='#7570b3', linewidth=2.0, linestyle=':', drawstyle='steps-post')
            
            ax1.set_ylabel('Overall Rating Points ($R^e$)', fontsize=11, fontweight='bold', color='#d95f02', labelpad=8)
            ax2.set_ylabel('Style Rating Points ($R^o$, $R^d$)', fontsize=11, fontweight='bold', color='#333333', labelpad=8)
            ax1.tick_params(axis='y', labelcolor='#d95f02')
            
            # Combine legends
            lines = l1 + l2 + l3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        else:
            ax1.plot(df_team['date'], df_team['elo'], label='Overall Elo', color='#d95f02', linewidth=2.5, drawstyle='steps-post')
            ax1.set_ylabel('Rating Points', fontsize=11, fontweight='bold', labelpad=8)
            ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    # Add Tournament Shading & Trophy Markers
    confed_match_keys = {'UEFA': ['EURO'], 'CONMEBOL': ['COPA'], 'CAF': ['AFCON'], 'CONCACAF': ['GOLD'], 'AFC': ['ASIA']}[confed]
    allowed_tourn_keys = ['WC'] + confed_match_keys
    
    for t_info in TOURNAMENTS:
        if t_info['name'] not in allowed_tourn_keys:
            continue
            
        t_start = pd.to_datetime(t_info['start'])
        t_end = pd.to_datetime(t_info['end'])
        
        if t_start >= dt_start and t_end <= dt_stop:
            is_wc = (t_info['name'] == 'WC')
            band_color = '#fee08b' if is_wc else '#e0f3f8'
            alpha_val = 0.40 if is_wc else 0.35
            
            ax1.axvspan(t_start, t_end, color=band_color, alpha=alpha_val, zorder=1)
            
            # Check if team won the championship
            if t_info['winner'].lower() == exact_team_name.lower():
                # Add Trophy Winner Star Marker
                mid_dt = t_start + (t_end - t_start) / 2
                y_max = ax1.get_ylim()[1]
                y_pos = y_max * 0.96 if normalized else y_max * 0.97
                
                ax1.scatter([mid_dt], [y_pos], color='#d73027', marker='*', s=220, zorder=10, edgecolor='black', linewidth=0.8)
                txt = ax1.text(mid_dt, y_pos * 0.985, f"{t_info['title']}\n(Winner ★)", ha='center', va='top', fontsize=8, fontweight='bold', color='#a50026', zorder=11)
                txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    ax1.set_title(f"Historical Rating Evolution for {exact_team_name} ({dt_start.strftime('%Y')}–{dt_stop.strftime('%Y')})", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Match Calendar Date', fontsize=11, fontweight='bold', labelpad=8)
    ax1.grid(True, linestyle=':', alpha=0.5)

    if normalized:
        ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    
    # Save outputs
    out_name = f"figure_eloevo_{exact_team_name.lower()}{'_norm' if normalized else ''}"
    pdf_path = os.path.join(figures_dir, f"{out_name}.pdf")
    png_path = os.path.join(figures_dir, f"{out_name}.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Successfully generated figure:")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}\n")
    return pdf_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate team Elo evolution figure equivalent to figure_eloevo_spain.pdf.")
    parser.add_argument('team', type=str, help="Country name (Required, e.g. spain, brazil, germany).")
    parser.add_argument('system', nargs='?', default='3eloC', help="Rating model (Default: 3eloC).")
    parser.add_argument('startdate', nargs='?', default='2006', help="Start year / date (e.g. 2006, 06-2006, 2006-06-01). Default: 2006.")
    parser.add_argument('stopdate', nargs='?', default='2026', help="End year / date (e.g. 2026, 07-2026, 2026-07-19). Default: 2026.")
    parser.add_argument('-n', '--normalized', action='store_true', help="Plot normalized ratings relative to Top 10 world baseline (single Y-axis).")
    
    args = parser.parse_args()
    
    figure_evolution_team(
        team=args.team,
        system=args.system,
        start_date=args.startdate,
        stop_date=args.stopdate,
        normalized=args.normalized
    )
