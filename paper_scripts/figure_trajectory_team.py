import os
import sys
import argparse
import math
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

import multielo
from run_compute_team import run_compute_team, SYSTEM_CONFIGS, SYSTEM_ALIASES, load_params_file
from figure_elolution_team import TOURNAMENTS, get_confederation, parse_date_string


def calculate_grid_layout(num_teams):
    if num_teams == 1:
        return 1, 1
    elif num_teams == 2:
        return 1, 2
    elif num_teams <= 4:
        return 2, 2
    elif num_teams <= 6:
        return 2, 3
    elif num_teams <= 9:
        return 3, 3
    elif num_teams <= 12:
        return 3, 4
    else:
        rows = math.ceil(num_teams / 4)
        return rows, 4


def figure_trajectory_team(teams=None, rating='3eloC', startdate='1950', enddate='2026'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Default teams list if none provided
    if not teams:
        teams = ['brazil', 'italy', 'germany', 'argentina']
    elif isinstance(teams, str):
        teams = [teams]
        
    # Resolve system
    sys_str = str(rating).lower().strip()
    canonical_sys = SYSTEM_ALIASES.get(sys_str, '3eloC')
    config = SYSTEM_CONFIGS.get(canonical_sys, SYSTEM_CONFIGS['3eloC'])
    
    dt_start = parse_date_string(startdate, is_end=False)
    dt_stop = parse_date_string(enddate, is_end=True)
    
    print(f"=== Plotting Style Trajectories Grid ===")
    print(f"Teams ({len(teams)}): {teams}")
    print(f"Rating System : '{canonical_sys}' ({config['system']})")
    print(f"Date Range    : {dt_start.strftime('%Y-%m-%d')} to {dt_stop.strftime('%Y-%m-%d')}")
    
    # Ensure normalized daily rating dataset exists
    norm_csv_path = os.path.join(data_dir, f"ratings_{canonical_sys}_all_norm.csv")
    if not os.path.exists(norm_csv_path):
        print(f"Normalized ratings dataset not found at {norm_csv_path}. Triggering run_compute_team.py...")
        run_compute_team(team='all', system=canonical_sys, normalize=True)
        
    df_all_norm = pd.read_csv(norm_csv_path)
    df_all_norm['date'] = pd.to_datetime(df_all_norm['date'])
    
    # Also load match results to check tournament appearances
    raw_results_path = os.path.join(data_dir, 'results.csv')
    df_matches = multielo.load_dataset(path=raw_results_path)
    
    # Check that rating system contains style ratings (norm_off and norm_def)
    if 'norm_off' not in df_all_norm.columns or 'norm_def' not in df_all_norm.columns:
        raise ValueError(f"Rating system '{canonical_sys}' does not maintain separate Offensive and Defensive style ratings. "
                         f"Please select a multi-vector system such as 3eloC, 3eloH, or 3eloOD.")

    # Match team names case-insensitively
    all_teams_map = {t.lower(): t for t in df_all_norm['team'].unique()}
    target_teams_exact = []
    for t_req in teams:
        t_clean = str(t_req).strip().lower()
        if t_clean in all_teams_map:
            target_teams_exact.append(all_teams_map[t_clean])
        else:
            print(f"Warning: Team '{t_req}' not found in dataset. Skipping.")
            
    if not target_teams_exact:
        raise ValueError(f"None of the requested teams {teams} were found in dataset.")

    # Setup Subplot Grid
    n_teams = len(target_teams_exact)
    n_rows, n_cols = calculate_grid_layout(n_teams)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), dpi=300, squeeze=False)
    
    team_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for idx, team_name in enumerate(target_teams_exact):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        ax = axes[r_idx, c_idx]
        
        confed = get_confederation(team_name)
        confed_match_keys = {'UEFA': ['EURO'], 'CONMEBOL': ['COPA'], 'CAF': ['AFCON'], 'CONCACAF': ['GOLD'], 'AFC': ['ASIA']}[confed]
        allowed_tourn_keys = ['WC'] + confed_match_keys
        
        # Extract team trajectory within date range
        df_t = df_all_norm[(df_all_norm['team'] == team_name) & 
                           (df_all_norm['date'] >= dt_start) & 
                           (df_all_norm['date'] <= dt_stop)].sort_values('date').reset_index(drop=True)
        
        if len(df_t) == 0:
            ax.text(0.5, 0.5, f"No Data\n({team_name})", ha='center', va='center', fontsize=10)
            continue
            
        # Draw Top 10 World Baseline lines ($S = 1.0$)
        ax.axhline(1.0, color='#999999', linestyle='--', linewidth=1.0, zorder=1)
        ax.axvline(1.0, color='#999999', linestyle='--', linewidth=1.0, zorder=1)
        
        # Equal style balance line (y = x)
        ax.plot([0.5, 1.5], [0.5, 1.5], color='#cccccc', linestyle=':', linewidth=0.9, zorder=1)
        
        # Plot continuous trajectory path
        t_color = team_colors[idx % len(team_colors)]
        ax.plot(df_t['norm_def'], df_t['norm_off'], color=t_color, linewidth=1.8, alpha=0.85, zorder=3)
        
        # Extract matches for team to verify participation in tournament editions
        df_team_matches = df_matches[(df_matches['home_team'] == team_name) | (df_matches['away_team'] == team_name)]
        
        # Annotate Tournament Editions (World Cups + Continental Championships)
        for t_info in TOURNAMENTS:
            if t_info['name'] not in allowed_tourn_keys:
                continue
                
            t_start = pd.to_datetime(t_info['start'])
            t_end = pd.to_datetime(t_info['end'])
            
            if t_start >= dt_start and t_end <= dt_stop:
                # Check if team actually played in this tournament
                t_matches = df_team_matches[(df_team_matches['date'] >= t_start) & (df_team_matches['date'] <= t_end)]
                if len(t_matches) == 0:
                    continue
                    
                # Get team's trajectory point nearest to tournament start
                match_dt = t_matches['date'].min()
                idx_near = (df_t['date'] - match_dt).abs().idxmin()
                pt_row = df_t.iloc[idx_near]
                
                pt_def = pt_row['norm_def']
                pt_off = pt_row['norm_off']
                
                is_winner = (t_info['winner'].lower() == team_name.lower())
                
                # Yellow / Gold for Champion, Silver for non-champion participant
                m_color = '#ffd700' if is_winner else '#c0c0c0'
                m_edge = '#b8860b' if is_winner else '#666666'
                m_size = 110 if is_winner else 45
                m_shape = '*' if is_winner else 'o'
                
                ax.scatter([pt_def], [pt_off], color=m_color, marker=m_shape, s=m_size, zorder=6, edgecolor=m_edge, linewidth=0.8)
                
                # Format two-digit year label (e.g. '08', '10', '24')
                year_lbl = f"'{t_start.strftime('%y')}"
                if is_winner:
                    year_lbl += "★"
                    
                txt = ax.text(pt_def + 0.003, pt_off + 0.003, year_lbl, fontsize=7.5, fontweight='bold' if is_winner else 'normal',
                              color='#900000' if is_winner else '#333333', zorder=7)
                txt.set_path_effects([path_effects.withStroke(linewidth=1.8, foreground='white')])

        # Subplot Title & Annotations
        ax.set_title(f"{team_name} ({df_t['date'].iloc[0].year}–{df_t['date'].iloc[-1].year})", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel('Defensive Score ($R^d / R^d_{10\\mathrm{th}}$)', fontsize=9, fontweight='bold', labelpad=4)
        ax.set_ylabel('Offensive Score ($R^o / R^o_{10\\mathrm{th}}$)', fontsize=9, fontweight='bold', labelpad=4)
        
        # Dynamic axis limits centered around data
        min_d, max_d = df_t['norm_def'].min(), df_t['norm_def'].max()
        min_o, max_o = df_t['norm_off'].min(), df_t['norm_off'].max()
        
        pad_d = max(0.04, (max_d - min_d) * 0.15)
        pad_o = max(0.04, (max_o - min_o) * 0.15)
        
        ax.set_xlim(max(0.70, min_d - pad_d), max_d + pad_d)
        ax.set_ylim(max(0.70, min_o - pad_o), max_o + pad_o)
        
        ax.grid(True, linestyle=':', alpha=0.4)

    # Hide unused subplots in grid if any
    for idx in range(n_teams, n_rows * n_cols):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        fig.delaxes(axes[r_idx, c_idx])

    plt.suptitle(f"Tactical Style Trajectories ({canonical_sys})", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save outputs
    out_name = f"figure_trajectories_{canonical_sys}_{n_teams}teams"
    pdf_path = os.path.join(figures_dir, f"{out_name}.pdf")
    png_path = os.path.join(figures_dir, f"{out_name}.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Successfully generated trajectory figure:")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}\n")
    return pdf_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate tactical style trajectories grid figure equivalent to figure_all_trajectories_grid.pdf.")
    parser.add_argument('teams', nargs='*', default=['brazil', 'italy', 'germany', 'argentina'],
                        help="List of teams to plot (e.g. brazil italy germany argentina). Default: brazil italy germany argentina.")
    parser.add_argument('--rating', '--system', type=str, default='3eloC',
                        help="Rating system with O and D components (e.g. 3eloC, 3eloH, 3eloOD). Default: 3eloC.")
    parser.add_argument('--startdate', type=str, default='1950',
                        help="Start year/date (e.g. 1950). Default: 1950.")
    parser.add_argument('--enddate', type=str, default='2026',
                        help="End year/date (e.g. 2026). Default: 2026.")
                        
    args = parser.parse_args()
    
    figure_trajectory_team(
        teams=args.teams,
        rating=args.rating,
        startdate=args.startdate,
        enddate=args.enddate
    )
