import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

from run_compute_team import run_compute_team, SYSTEM_CONFIGS, SYSTEM_ALIASES
from figure_elolution_team import TOURNAMENTS, parse_date_string
import multielo


def parse_remarkable_arg(arg_str):
    """
    Parse remarkable team string format: 'team-year' or 'team_year' (e.g., 'brazil-1982', 'netherlands-1974').
    """
    s = str(arg_str).strip()
    if '-' in s:
        parts = s.rsplit('-', 1)
    elif '_' in s:
        parts = s.rsplit('_', 1)
    else:
        raise ValueError(f"Invalid remarkable format '{arg_str}'. Expected 'team-year' (e.g. 'brazil-1982').")
        
    team_name = parts[0].strip()
    year_val = int(parts[1].strip())
    return team_name, year_val


def figure_point_teams(rating='3eloC', startdate='1950', enddate='2026', remarkable_list=None, nowc=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Resolve system
    sys_str = str(rating).lower().strip()
    canonical_sys = SYSTEM_ALIASES.get(sys_str, '3eloC')
    config = SYSTEM_CONFIGS.get(canonical_sys, SYSTEM_CONFIGS['3eloC'])
    
    dt_start = parse_date_string(startdate, is_end=False)
    dt_stop = parse_date_string(enddate, is_end=True)
    
    print(f"=== Plotting World #1 Style Scatter Space ===")
    print(f"Rating Model: '{canonical_sys}' ({config['system']})")
    print(f"Date Window : {dt_start.strftime('%Y-%m-%d')} to {dt_stop.strftime('%Y-%m-%d')}")
    print(f"Include WC  : {not nowc}")
    print(f"Remarkable  : {remarkable_list}")
    
    # Ensure normalized daily rating dataset exists
    norm_csv_path = os.path.join(data_dir, f"ratings_{canonical_sys}_all_norm.csv")
    if not os.path.exists(norm_csv_path):
        print(f"Normalized ratings dataset not found at {norm_csv_path}. Triggering run_compute_team.py...")
        run_compute_team(team='all', system=canonical_sys, normalize=True)
        
    df_all_norm = pd.read_csv(norm_csv_path)
    df_all_norm['date'] = pd.to_datetime(df_all_norm['date'])
    
    # Filter date range
    df_all_norm = df_all_norm[(df_all_norm['date'] >= dt_start) & (df_all_norm['date'] <= dt_stop)].reset_index(drop=True)
    
    if 'norm_off' not in df_all_norm.columns or 'norm_def' not in df_all_norm.columns:
        raise ValueError(f"Rating system '{canonical_sys}' does not maintain separate Offensive and Defensive style ratings.")

    # 1. Identify World #1 ranked team on every match date
    # On each match date, extract the team with maximum overall rating (elo)
    idx_max = df_all_norm.groupby('date')['elo'].idxmax()
    df_no1_daily = df_all_norm.loc[idx_max].sort_values('date').reset_index(drop=True)
    
    # Filter out isolated regional inflation artifacts (e.g. Tahiti 1980s Pacific games)
    df_no1_daily = df_no1_daily[df_no1_daily['team'] != 'Tahiti'].reset_index(drop=True)
    
    # 2. Perform 90-day (quarterly) temporal sampling for World #1 tenures
    df_no1_daily['spell_id'] = (df_no1_daily['team'] != df_no1_daily['team'].shift(1)).cumsum()
    
    sampled_no1_rows = []
    for _, spell in df_no1_daily.groupby('spell_id'):
        duration_days = (spell['date'].max() - spell['date'].min()).days
        if duration_days >= 90:
            # Sample quarterly (every 90 days)
            q_dates = pd.date_range(spell['date'].min(), spell['date'].max(), freq='90D')
            for q_dt in q_dates:
                idx_near = (spell['date'] - q_dt).abs().idxmin()
                sampled_no1_rows.append(spell.loc[idx_near])
        else:
            # Sample median date for short tenure
            mid_idx = len(spell) // 2
            sampled_no1_rows.append(spell.iloc[mid_idx])
            
    df_no1_sampled = pd.DataFrame(sampled_no1_rows).drop_duplicates(subset=['date', 'team']).reset_index(drop=True)
    
    print(f"Extracted {len(df_no1_sampled)} sampled World #1 points across {len(df_no1_sampled['team'].unique())} distinct leaders.")

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    
    # Reference Lines
    ax.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2, zorder=1)
    ax.axvline(1.0, color='#999999', linestyle='--', linewidth=1.2, zorder=1)
    ax.plot([0.5, 1.5], [0.5, 1.5], color='#cccccc', linestyle=':', linewidth=1.0, zorder=1)

    # Plot World #1 Scatter Points (colored by team)
    teams_no1 = sorted(df_no1_sampled['team'].unique())
    cmap = plt.get_cmap('tab10', max(10, len(teams_no1)))
    color_map = {t: cmap(i) for i, t in enumerate(teams_no1)}
    
    for t_name, group in df_no1_sampled.groupby('team'):
        ax.scatter(group['norm_def'], group['norm_off'], label=t_name, color=color_map[t_name],
                   alpha=0.65, s=40, edgecolor='white', linewidth=0.4, zorder=3)
                   
    # 4. Overlay World Cup Champions (Gold Stars) if not --nowc
    if not nowc:
        print("Overlaying World Cup Champions (Gold Stars ★)...")
        for t_info in TOURNAMENTS:
            if t_info['name'] == 'WC':
                wc_start = pd.to_datetime(t_info['start'])
                winner = t_info['winner']
                
                if wc_start >= dt_start and wc_start <= dt_stop:
                    # Find winner's rating point just prior to tournament
                    sub_winner = df_all_norm[(df_all_norm['team'].str.lower() == winner.lower()) & 
                                             (df_all_norm['date'] <= wc_start)].sort_values('date')
                    if len(sub_winner) > 0:
                        champ_pt = sub_winner.iloc[-1]
                        c_def, c_off = champ_pt['norm_def'], champ_pt['norm_off']
                        
                        ax.scatter([c_def], [c_off], color='#ffd700', marker='*', s=240,
                                   edgecolor='#b8860b', linewidth=0.9, zorder=10)
                        
                        yr_txt = f"{winner} '{wc_start.strftime('%y')}"
                        txt = ax.text(c_def + 0.003, c_off + 0.003, yr_txt, fontsize=8, fontweight='bold', color='#8b0000', zorder=11)
                        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # 5. Overlay Remarkable Teams (Silver Circles) if requested via --remarkable
    if remarkable_list:
        print("Overlaying Remarkable Non-Champion Teams (Silver Circles ●)...")
        all_teams_lower = {t.lower(): t for t in df_all_norm['team'].unique()}
        
        for rem_item in remarkable_list:
            try:
                r_team, r_year = parse_remarkable_arg(rem_item)
                r_team_clean = r_team.lower()
                
                if r_team_clean in all_teams_lower:
                    exact_name = all_teams_lower[r_team_clean]
                    # Target date around mid-year of that remarkable season
                    rem_dt = pd.to_datetime(f"{r_year}-06-01")
                    
                    sub_rem = df_all_norm[(df_all_norm['team'] == exact_name) & 
                                          (df_all_norm['date'] <= rem_dt)].sort_values('date')
                    if len(sub_rem) == 0:
                        sub_rem = df_all_norm[df_all_norm['team'] == exact_name].sort_values('date')
                        
                    if len(sub_rem) > 0:
                        rem_pt = sub_rem.iloc[-1]
                        r_def, r_off = rem_pt['norm_def'], rem_pt['norm_off']
                        
                        ax.scatter([r_def], [r_off], color='#c0c0c0', marker='o', s=110,
                                   edgecolor='#444444', linewidth=1.0, zorder=9)
                        
                        label_txt = f"{exact_name} '{str(r_year)[-2:]}"
                        txt = ax.text(r_def + 0.003, r_off - 0.005, label_txt, fontsize=8, fontweight='bold', color='#222222', zorder=12)
                        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
            except Exception as e:
                print(f"Warning: Failed to parse remarkable entry '{rem_item}': {e}")

    # Titles and Axes
    ax.set_title(f"World #1 Tactical Style Positions in Normalized Style Space ({dt_start.strftime('%Y')}–{dt_stop.strftime('%Y')})", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Defensive Score ($R^d / R^d_{10\\mathrm{th}}$)', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel('Offensive Score ($R^o / R^o_{10\\mathrm{th}}$)', fontsize=10, fontweight='bold', labelpad=8)
    
    ax.legend(loc='lower left', ncol=2, frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    
    # Save outputs
    out_name = f"figure_point_teams_{canonical_sys}{'_nowc' if nowc else ''}"
    pdf_path = os.path.join(figures_dir, f"{out_name}.pdf")
    png_path = os.path.join(figures_dir, f"{out_name}.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Successfully generated point teams scatter figure:")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}\n")
    return pdf_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate World #1 style scatter figure equivalent to number_one_relative_profile_scatter.pdf.")
    parser.add_argument('--rating', '--system', type=str, default='3eloC', help="Rating model (Default: 3eloC).")
    parser.add_argument('--startdate', type=str, default='1950', help="Start year/date (Default: 1950).")
    parser.add_argument('--enddate', type=str, default='2026', help="End year/date (Default: 2026).")
    parser.add_argument('--remarkable', action='append', default=[], help="Include remarkable non-champion team (e.g., --remarkable brazil-1982 --remarkable netherlands-1974).")
    parser.add_argument('--nowc', action='store_true', help="Do not include World Cup Champions gold stars.")
    
    args = parser.parse_args()
    
    figure_point_teams(
        rating=args.rating,
        startdate=args.startdate,
        enddate=args.enddate,
        remarkable_list=args.remarkable,
        nowc=args.nowc
    )
