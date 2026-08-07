import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# Import multielo package
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

import multielo
from run_compute_ratings import SYSTEM_CONFIGS, SYSTEM_ALIASES, load_params_file


def run_compute_team(team='all', system='3eloC', normalize=False, startdate=None, stopdate=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    params_dir = os.path.join(script_dir, 'params')
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, 'results.csv')
    if os.path.exists(dataset_path):
        df = multielo.load_dataset(path=dataset_path)
    else:
        df = multielo.load_dataset()
    
    # Resolve system key
    sys_str = str(system).lower().strip()
    canonical_sys = SYSTEM_ALIASES.get(sys_str, '3eloC')
    config = SYSTEM_CONFIGS.get(canonical_sys, SYSTEM_CONFIGS['3eloC'])
    
    params = load_params_file(params_dir, config['params_file'])
    
    print(f"=== Computing Day-by-Day Ratings ===")
    print(f"Target Team : '{team}'")
    print(f"Rating Model: '{canonical_sys}' ({config['system']})")
    print(f"Normalize   : {normalize} (Relative to 10th Place World Baseline on EVERY match date)")
    
    # Filter dates if provided
    if startdate:
        df = df[df['date'] >= pd.to_datetime(startdate)]
    if stopdate:
        df = df[df['date'] <= pd.to_datetime(stopdate)]
    df = df.reset_index(drop=True)
    
    # 1. Compute full chronological rating simulation over all matches
    df_rated = multielo.compute_ratings(df, system=config['system'], params=params)
    
    # Identify rating column names dynamically
    elo_col_h = 'elo_home' if 'elo_home' in df_rated.columns else ('fast_home' if 'fast_home' in df_rated.columns else 'fifa_home')
    elo_col_a = 'elo_away' if 'elo_away' in df_rated.columns else ('fast_away' if 'fast_away' in df_rated.columns else 'fifa_away')
    has_style = ('off_home' in df_rated.columns)
    
    # 2. Track team states on EVERY match date in international football history
    all_teams = sorted(set(df_rated['home_team']).union(set(df_rated['away_team'])))
    all_dates = sorted(df_rated['date'].unique())
    
    print(f"Tracking state across {len(all_dates):,} unique match dates for {len(all_teams)} active teams...")
    
    r_elo = {t: 1500.0 for t in all_teams}
    r_off = {t: 1500.0 for t in all_teams} if has_style else {}
    r_def = {t: 1500.0 for t in all_teams} if has_style else {}
    
    # Group match rows by date
    daily_groups = df_rated.groupby('date')
    
    daily_records = []
    
    for dt, group in daily_groups:
        # First update team ratings for teams that played on date dt
        for row in group.itertuples():
            r_elo[row.home_team] = getattr(row, elo_col_h)
            r_elo[row.away_team] = getattr(row, elo_col_a)
            if has_style:
                r_off[row.home_team] = row.off_home
                r_off[row.away_team] = row.off_away
                r_def[row.home_team] = row.def_home
                r_def[row.away_team] = row.def_away
                
        # Compute global 10th-place benchmarks on this match date
        s_elo_10th = sorted(r_elo.values(), reverse=True)[9] if len(r_elo) >= 10 else 1500.0
        s_off_10th = sorted(r_off.values(), reverse=True)[9] if has_style and len(r_off) >= 10 else 1500.0
        s_def_10th = sorted(r_def.values(), reverse=True)[9] if has_style and len(r_def) >= 10 else 1500.0
        
        # Store snapshot for all active teams on this match date
        for t_name in all_teams:
            rec = {
                'date': dt,
                'team': t_name,
                'elo': r_elo[t_name],
                'elo_10th': s_elo_10th,
                'norm_elo': r_elo[t_name] / max(1e-5, s_elo_10th)
            }
            if has_style:
                rec['elo_off'] = r_off[t_name]
                rec['elo_def'] = r_def[t_name]
                rec['off_10th'] = s_off_10th
                rec['def_10th'] = s_def_10th
                rec['norm_off'] = r_off[t_name] / max(1e-5, s_off_10th)
                rec['norm_def'] = r_def[t_name] / max(1e-5, s_def_10th)
                
            daily_records.append(rec)
            
    df_daily_all = pd.DataFrame(daily_records)
    
    # Filter target team if requested
    team_query = str(team).strip()
    all_teams_map = {t.lower(): t for t in all_teams}
    
    if team_query.lower() != 'all':
        if team_query.lower() not in all_teams_map:
            raise ValueError(f"Team '{team}' not found in dataset.")
        target_team_exact = all_teams_map[team_query.lower()]
        print(f"Exact matched team: '{target_team_exact}'")
        
        df_target = df_daily_all[df_daily_all['team'] == target_team_exact].copy().reset_index(drop=True)
        
        # Flag match days where team actually played
        match_dates = set(df_rated[(df_rated['home_team'] == target_team_exact) | (df_rated['away_team'] == target_team_exact)]['date'])
        df_target['played_match_today'] = df_target['date'].isin(match_dates)
        
        if not normalize:
            # Drop 10th-place benchmark columns if normalization was not requested
            cols_to_drop = [c for c in ['elo_10th', 'norm_elo', 'off_10th', 'def_10th', 'norm_off', 'norm_def'] if c in df_target.columns]
            df_target = df_target.drop(columns=cols_to_drop)
            
        out_filename = f"ratings_{canonical_sys}_{target_team_exact.lower()}{'_norm' if normalize else ''}.csv"
        out_path = os.path.join(data_dir, out_filename)
        df_target.to_csv(out_path, index=False)
        print(f"Successfully saved {len(df_target):,} daily snapshots for {target_team_exact} across all match dates to: {out_path}")
        return out_path
        
    else:
        if not normalize:
            cols_to_drop = [c for c in ['elo_10th', 'norm_elo', 'off_10th', 'def_10th', 'norm_off', 'norm_def'] if c in df_daily_all.columns]
            df_daily_all = df_daily_all.drop(columns=cols_to_drop)
            
        out_filename = f"ratings_{canonical_sys}_all{'_norm' if normalize else ''}.csv"
        out_path = os.path.join(data_dir, out_filename)
        df_daily_all.to_csv(out_path, index=False)
        print(f"Successfully saved all team daily trajectories ({len(df_daily_all):,} records) to: {out_path}")
        return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute day-by-day rating trajectories for a specific team or all teams.")
    parser.add_argument('team', nargs='?', default='all', help="Target team name (e.g. spain, brazil, Germany, all). Default is 'all'.")
    parser.add_argument('system', nargs='?', default='3eloC', help="Rating system (e.g. 3eloC, 3eloH, 3eloOD, 1eloC, eloratings, fifa). Default is '3eloC'.")
    parser.add_argument('-n', '--normalize', action='store_true', help="Normalize ratings relative to 10th place world baseline on every match date.")
    parser.add_argument('--startdate', type=str, default=None, help="Start date for inclusive filtering (YYYY-MM-DD).")
    parser.add_argument('--stopdate', type=str, default=None, help="Stop date for inclusive filtering (YYYY-MM-DD).")
    
    args = parser.parse_args()
    
    run_compute_team(
        team=args.team,
        system=args.system,
        normalize=args.normalize,
        startdate=args.startdate,
        stopdate=args.stopdate
    )
