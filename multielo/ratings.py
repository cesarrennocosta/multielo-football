import numpy as np
import pandas as pd
from bisect import bisect_right

def compute_ratings(df, system='3elo-complete', params=None):
    """
    Compute chronological Elo rating histories across international football match records.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of international match records (must contain date, home_team, away_team, home_score, away_score).
    system : str
        Rating system architecture. Options:
        - 'fifa-sum' : Official FIFA/Coca-Cola World Ranking (SUM formula).
        - 'eloratings' : World Football Elo Ratings (elorating.net benchmark).
        - '1elo-simple' : 4-parameter single-scale Elo.
        - '1elo-complete' : 10-parameter complete tier-weighted single-scale Elo.
        - '2eloOD' / '2elo-od' : Decoupled Offensive & Defensive (2eloOD) ratings.
        - '2elo-fast-slow' : Dual-timescale Fast+Slow outcome ratings.
        - '3elo-simple' / '3elo-hybrid' : Overall outcome + Decoupled style ratings.
        - '3elo-complete' : Complete multi-vector outcome + Decoupled style ratings.
    params : dict, optional
        Custom parameter dictionary. If None, default optimal parameters are used.
        
    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame augmented with computed pre-match and post-match ratings as well as 6-month lagged ratings.
    """
    system_key = str(system).lower().replace('_', '-').strip()
    df_out = df.copy()
    
    teams = set(df_out['home_team']).union(set(df_out['away_team']))
    
    if system_key == 'fifa-sum':
        res = _compute_fifa_sum(df_out, teams)
        return _compute_6mo_lag(res, rating_col_h='fifa_home', rating_col_a='fifa_away', out_diff_col='fifa_diff_6mo')
    elif system_key in ['eloratings', 'eloratings.net', 'elonet']:
        res = _compute_eloratings(df_out, teams)
        return _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo')
    elif system_key in ['1elo-simple', '1elo-s', '1elof', '1elos']:
        res = _compute_1elo_simple(df_out, teams, params)
        return _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo')
    elif system_key in ['1elo-complete', '1elo-c', '1eloc', '1elocc', '1elo-cc']:
        res = _compute_1elo_complete(df_out, teams, params)
        return _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo')
    elif system_key in ['1elog', '1elo-g', '1elogc', '1elo-gc']:
        res = _compute_1elo_g(df_out, teams, params)
        return _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo')
    elif system_key in ['2elog', '2elo-g']:
        res = _compute_2elo_g(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo1_home', rating_col_a='elo1_away', out_diff_col='elo1_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='elo2_home', rating_col_a='elo2_away', out_diff_col='elo2_diff_6mo')
        res['elo_diff_6mo'] = res['elo1_diff_6mo']
        return res
    elif system_key in ['1elox', '1elo-x']:
        res = _compute_1elo_x(df_out, teams, params)
        return _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo')
    elif system_key in ['2elo-od', '2elood', '2elo_od', '2elo-pure', '2elo-style', '2elo-sd', '2elood_pure', '2elo_pure']:
        res = _compute_2elo_od(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_off_6mo'] - res['diff_def_6mo']
        return res
    elif system_key in ['2eloodg', '2elo-odg', '2elo_odg', '2eloodgc', '2elo-odgc']:
        res = _compute_2elo_odg(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_off_6mo'] - res['diff_def_6mo']
        return res
    elif system_key in ['2eloodx', '2elo-odx', '2elo_odx']:
        res = _compute_2elo_odx(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_off_6mo'] - res['diff_def_6mo']
        return res
    elif system_key in ['2eloodc', '2elo-odc', '2elo_odc']:
        res = _compute_2elo_odc(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_off_6mo'] - res['diff_def_6mo']
        return res
    elif system_key in ['3eloodg', '3elo-odg', '3elo_odg']:
        res = _compute_3elo_odg(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='diff_overall_6mo')
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_overall_6mo'] = res['elo_home_6mo'] - res['elo_away_6mo'] if 'elo_home_6mo' in res.columns else res['diff_overall']
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_overall_6mo']
        return res
    elif system_key in ['3elood+1g', '3elo_od_1g', '3elood1g', '3elo-od-1g', '3elood+g']:
        res = _compute_3elo_od_1g(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='diff_overall_6mo')
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_overall_6mo'] = res['elo_home_6mo'] - res['elo_away_6mo'] if 'elo_home_6mo' in res.columns else res['diff_overall']
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_overall_6mo']
        return res
    elif system_key in ['4elood+2g', '4elo_od_2g', '4elood2g', '4elo-od-2g']:
        res = _compute_4elo_od_2g(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo1_home', rating_col_a='elo1_away', out_diff_col='elo1_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='elo2_home', rating_col_a='elo2_away', out_diff_col='elo2_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['elo1_diff_6mo'] = res['elo1_home_6mo'] - res['elo1_away_6mo'] if 'elo1_home_6mo' in res.columns else res['elo1_diff']
        res['elo2_diff_6mo'] = res['elo2_home_6mo'] - res['elo2_away_6mo'] if 'elo2_home_6mo' in res.columns else res['elo2_diff']
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['elo1_diff_6mo']
        return res
    elif system_key in ['2elo-fsc', '2elofsc', '2elo_fsc', '2elo-fast+slow-c', '2elo-fsc-complete']:
        res = _compute_2elo_fsc(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo_home_fast', rating_col_a='elo_away_fast', out_diff_col='elo_diff_fast_6mo')
        res = _compute_6mo_lag(res, rating_col_h='elo_home_slow', rating_col_a='elo_away_slow', out_diff_col='elo_diff_slow_6mo')
        res['elo_diff_6mo'] = res['elo_diff_fast_6mo']
        return res
    elif system_key in ['2elo-fsk', '2elofsk', '2elo_fsk']:
        res = _compute_2elo_fsk(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo_home_fast', rating_col_a='elo_away_fast', out_diff_col='elo_diff_fast_6mo')
        res = _compute_6mo_lag(res, rating_col_h='elo_home_slow', rating_col_a='elo_away_slow', out_diff_col='elo_diff_slow_6mo')
        res['elo_diff_6mo'] = res['elo_diff_fast_6mo']
        return res
    elif system_key in ['2elo-fsg', '2elofsg', '2elo_fsg']:
        res = _compute_2elo_fsg(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo_home_fast', rating_col_a='elo_away_fast', out_diff_col='elo_diff_fast_6mo')
        res = _compute_6mo_lag(res, rating_col_h='elo_home_slow', rating_col_a='elo_away_slow', out_diff_col='elo_diff_slow_6mo')
        res['elo_diff_6mo'] = res['elo_diff_fast_6mo']
        return res
    elif system_key in ['2elo-fast-slow', '2elo-fastslow', '2elo_fastslow', '2elo-fs', '2elofs', '2elofss', '2elofsx', '2elo-fast+slow']:
        res = _compute_2elo_fast_slow(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo_home_fast', rating_col_a='elo_away_fast', out_diff_col='elo_diff_fast_6mo')
        res = _compute_6mo_lag(res, rating_col_h='elo_home_slow', rating_col_a='elo_away_slow', out_diff_col='elo_diff_slow_6mo')
        res['elo_diff_6mo'] = res['elo_diff_fast_6mo']
        return res
    elif system_key in ['2eloodk', '2eloodx', '2elo-odk', '2elo-odx', '2elo_odk', '2elo_odx']:
        res = _compute_2elo_odx(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_off_6mo'] - res['diff_def_6mo']
        return res
    elif system_key in ['2eloodc', '2elo-odc', '2elo_odc']:
        res = _compute_2elo_odc(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['diff_off_6mo'] - res['diff_def_6mo']
        return res
    elif system_key in ['3elo-simple', '3elo-hybrid', '3elo-h', '3eloh']:
        res = _compute_3elo_hybrid(df_out, teams, params)
        return _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo')
    elif system_key in ['3elo-complete', '3elo-c', '3eloc', '3elo2odc+1c', '3elo_2odc_1c', '3elo2odc1c', '3eloodc']:
        res = _compute_3elo_complete(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        return res
    elif system_key in ['4elo', '4-elo', '4elo-multiscale', '4elo_multiscale', '4elo-ms', '4eloms']:
        res = _compute_4elo_multiscale(df_out, teams, params)
        return _compute_6mo_lag(res, rating_col_h='elo_fast_home', rating_col_a='elo_fast_away', out_diff_col='elo_diff_6mo')
    elif system_key in ['4elog', '4elo-g', '4elood+2g', '4eloodg', '4elo_g', '4elo2odc+2fsc', '4elo_2odc_2fsc', '4elo2odc2fsc']:
        res = _compute_4elo_g(df_out, teams, params)
        res = _compute_6mo_lag(res, rating_col_h='elo1_home', rating_col_a='elo1_away', out_diff_col='elo1_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='elo2_home', rating_col_a='elo2_away', out_diff_col='elo2_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='off_home', rating_col_a='off_away', out_diff_col='off_diff_6mo')
        res = _compute_6mo_lag(res, rating_col_h='def_home', rating_col_a='def_away', out_diff_col='def_diff_6mo')
        res['elo1_diff_6mo'] = res['elo1_home_6mo'] - res['elo1_away_6mo'] if 'elo1_home_6mo' in res.columns else res['elo1_diff']
        res['elo2_diff_6mo'] = res['elo2_home_6mo'] - res['elo2_away_6mo'] if 'elo2_home_6mo' in res.columns else res['elo2_diff']
        res['diff_off_6mo'] = res['off_home_6mo'] - res['def_away_6mo'] if 'off_home_6mo' in res.columns and 'def_away_6mo' in res.columns else res['diff_off']
        res['diff_def_6mo'] = res['off_away_6mo'] - res['def_home_6mo'] if 'off_away_6mo' in res.columns and 'def_home_6mo' in res.columns else res['diff_def']
        res['elo_diff_6mo'] = res['elo1_diff_6mo']
        return res
    else:
        raise ValueError(f"Unknown rating system: '{system}'. Supported: 'fifa-sum', 'eloratings', '1elo-simple', '1elo-complete', '2eloOD', '2eloFS', '2eloFSK', '2eloFSG', '2eloFSC', '2eloODK', '2eloODG', '2eloODC', '3elo-hybrid', '3elo-complete', '4elo-multiscale'.")


def _compute_6mo_lag(df, rating_col_h='elo_home', rating_col_a='elo_away', out_diff_col='elo_diff_6mo'):
    """
    Computes 6-month (182.5 days) prior rating history for home and away teams.
    """
    if rating_col_h not in df.columns or rating_col_a not in df.columns or 'date' not in df.columns:
        if rating_col_h in df.columns and rating_col_a in df.columns:
            df[out_diff_col] = df[rating_col_h] - df[rating_col_a]
        return df

    teams = set(df['home_team']).union(set(df['away_team']))
    init_val = float(df[rating_col_h].iloc[0]) if len(df) > 0 else 1500.0
    init_ts = pd.Timestamp('1870-01-01').timestamp()
    
    team_hist = {t: [(init_ts, init_val)] for t in teams}
    
    six_months_sec = 182.5 * 86400.0
    dt_series = pd.to_datetime(df['date'])
    dates = ((dt_series - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1)).values
    
    r_h_vals = df[rating_col_h].astype(float).values
    r_a_vals = df[rating_col_a].astype(float).values
    home_teams = df['home_team'].values
    away_teams = df['away_team'].values
    
    r_h_6mo = np.zeros(len(df), dtype=float)
    r_a_6mo = np.zeros(len(df), dtype=float)
    
    for idx in range(len(df)):
        h, a = home_teams[idx], away_teams[idx]
        rh_val = r_h_vals[idx]
        ra_val = r_a_vals[idx]
        m_time = dates[idx]
        target_6mo = m_time - six_months_sec
        
        times_h, vals_h = zip(*team_hist[h])
        idx_h = max(0, bisect_right(times_h, target_6mo) - 1)
        rh_6m = vals_h[idx_h]
        
        times_a, vals_a = zip(*team_hist[a])
        idx_a = max(0, bisect_right(times_a, target_6mo) - 1)
        ra_6m = vals_a[idx_a]
        
        r_h_6mo[idx] = rh_6m
        r_a_6mo[idx] = ra_6m
        
        team_hist[h].append((m_time, rh_val))
        team_hist[a].append((m_time, ra_val))
        
    df[f'{rating_col_h}_6mo'] = r_h_6mo
    df[f'{rating_col_a}_6mo'] = r_a_6mo
    df[out_diff_col] = r_h_6mo - r_a_6mo
    return df


def _get_fifa_importance(tourn, stage=None):
    t = str(tourn).lower()
    st = str(stage).lower() if stage and not pd.isna(stage) else ""
    
    if 'world cup' in t and not ('qualif' in t or 'q' in t):
        if any(q in st for q in ['quarter', 'semi', 'final', '3rd', 'third', 'qf', 'sf', 'f']):
            return 60.0
        return 50.0
        
    if any(c in t for c in ['euro', 'copa américa', 'copa america', 'african cup', 'afcon', 'asian cup', 'gold cup']) and not ('qualif' in t or 'q' in t):
        if any(q in st for q in ['quarter', 'semi', 'final', '3rd', 'third', 'qf', 'sf', 'f']):
            return 40.0
        return 35.0
        
    if 'qualif' in t or 'q' in t or 'play-off' in st or 'playoff' in st:
        return 25.0
        
    if 'nations' in t:
        if 'play-off' in st or 'final' in st:
            return 25.0
        return 15.0
        
    if 'friendly' in t:
        if 'outside' in st or 'non-imc' in st:
            return 5.0
        return 10.0
        
    return 20.0


def _get_elorating_k(tourn):
    t = str(tourn).lower()
    if 'world cup' in t:
        return 40.0 if ('qualif' in t or 'q' in t) else 60.0
    if any(c in t for c in ['euro', 'copa américa', 'copa america', 'african cup', 'afcon', 'asian cup', 'gold cup']):
        return 40.0 if ('qualif' in t or 'q' in t) else 50.0
    if 'qualif' in t or 'q' in t:
        return 40.0
    if 'friendly' in t:
        return 20.0
    return 30.0


def _compute_fifa_sum(df, teams):
    r_fifa = {t: 1000.0 for t in teams}
    fh, fa = [], []
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        fh.append(r_fifa[h]); fa.append(r_fifa[a])
        
        we_h = 1.0 / (1.0 + 10.0 ** ((r_fifa[a] - r_fifa[h]) / 600.0))
        we_a = 1.0 - we_h
        
        is_pso = getattr(row, 'penalty_shootout', False) or getattr(row, 'shootout', False)
        pso_winner = getattr(row, 'shootout_winner', None) or getattr(row, 'pso_winner', None)
        
        if is_pso and pso_winner is not None:
            if str(pso_winner).strip() == str(h).strip():
                sh, sa = 0.75, 0.5
            else:
                sh, sa = 0.5, 0.75
        else:
            if gh > ga:
                sh, sa = 1.0, 0.0
            elif gh < ga:
                sh, sa = 0.0, 1.0
            else:
                sh, sa = 0.5, 0.5
                
        tourn = row.tournament if hasattr(row, 'tournament') else 'Friendly'
        stage = getattr(row, 'stage', None)
        I = _get_fifa_importance(tourn, stage)
        
        delta_h = I * (sh - we_h)
        delta_a = I * (sa - we_a)
        
        st_lower = str(stage).lower() if stage and not pd.isna(stage) else ""
        t_lower = str(tourn).lower()
        is_knockout = any(k in st_lower for k in ['knockout', 'round of', 'quarter', 'semi', 'final', 'last 16', 'r16', 'qf', 'sf']) or ('knockout' in t_lower)
        is_final_comp = any(c in t_lower for c in ['world cup', 'euro', 'copa américa', 'copa america', 'afcon', 'asian cup', 'gold cup']) and not ('qualif' in t_lower)
        
        if is_knockout and is_final_comp:
            if sh < 0.5 and delta_h < 0:
                delta_h = 0.0
            if sa < 0.5 and delta_a < 0:
                delta_a = 0.0
                
        r_fifa[h] += delta_h
        r_fifa[a] += delta_a
        
    df['fifa_home'] = fh; df['fifa_away'] = fa
    df['fifa_diff'] = np.array(fh) - np.array(fa)
    return df


def _compute_eloratings(df, teams):
    r_elo = {t: 1500.0 for t in teams}
    eh, ea = [], []
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        eh.append(r_elo[h]); ea.append(r_elo[a])
        
        h_adv = 100.0 if neutral == 0 else 0.0
        we_h = 1.0 / (1.0 + 10.0 ** ((r_elo[a] - (r_elo[h] + h_adv)) / 400.0))
        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        
        gd = abs(gh - ga)
        if gd <= 1:
            G = 1.0
        elif gd == 2:
            G = 1.5
        elif gd == 3:
            G = 1.75
        else:
            G = 1.75 + (gd - 3.0) / 8.0
            
        K = _get_elorating_k(row.tournament if hasattr(row, 'tournament') else 'Friendly')
        
        r_elo[h] += K * G * (sh - we_h)
        r_elo[a] += K * G * ((1.0 - sh) - (1.0 - we_h))
        
    df['elo_home'] = eh; df['elo_away'] = ea
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df


def _compute_1elo_simple(df, teams, params):
    p = params or {'K_base': 35.0, 'M_overall': 2.0, 'H_overall': 100.0, 'divisor_overall': 400.0}
    Kb = float(p.get('K_base', 35.0))
    Mo = float(p.get('M_overall', 2.0))
    Ho = float(p.get('H_overall', 100.0))
    Div = float(p.get('divisor_overall', 400.0))
    
    r_elo = {t: 1500.0 for t in teams}
    eh, ea = [], []
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        eh.append(r_elo[h]); ea.append(r_elo[a])
        
        h_adv = Ho if neutral == 0 else 0.0
        exp_val = max(-100.0, min(100.0, (r_elo[a] - (r_elo[h] + h_adv)) / Div))
        we_h = 1.0 / (1.0 + 10.0 ** exp_val)
        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        
        k = (Kb * Mo if is_comp else Kb)
        r_elo[h] += k * (sh - we_h)
        r_elo[a] += k * ((1.0 - sh) - (1.0 - we_h))
        
    df['elo_home'] = eh; df['elo_away'] = ea
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df


def _get_1elo_complete_ktier(tourn, params):
    t = str(tourn).lower() if tourn and not pd.isna(tourn) else "friendly"
    kwc = float(params.get('K_WC', 60.0))
    kmaj = float(params.get('K_major', 50.0))
    kqual = float(params.get('K_qual', 40.0))
    kmin = float(params.get('K_minor', 30.0))
    kfri = float(params.get('K_friendly', 20.0))

    if 'world cup' in t:
        return kqual if ('qualif' in t or 'q' in t) else kwc
    if any(c in t for c in ['euro', 'copa américa', 'copa america', 'african cup', 'afcon', 'asian cup', 'gold cup']):
        return kqual if ('qualif' in t or 'q' in t) else kmaj
    if 'qualif' in t or 'q' in t:
        return kqual
    if 'friendly' in t:
        return kfri
    return kmin


def _compute_1elo_complete(df, teams, params):
    p = params or {}
    kwc = float(p.get('K_WC', 129.4524))
    kmaj = float(p.get('K_major', 119.2702))
    kqual = float(p.get('K_qual', 79.3888))
    kmin = float(p.get('K_minor', 54.5257))
    kfri = float(p.get('K_friendly', 40.1693))

    G2 = float(p.get('G2', 1.6502))
    am = float(p.get('a_margin', 2.8332))
    bm = float(p.get('b_margin', 0.6142))

    Ho = float(p.get('H_overall', 83.8974))
    Div = float(p.get('divisor_overall', 764.3723))

    params_dict = {
        'K_WC': kwc, 'K_major': kmaj, 'K_qual': kqual, 'K_minor': kmin, 'K_friendly': kfri,
        'G2': G2, 'a_margin': am, 'b_margin': bm, 'H_overall': Ho, 'divisor_overall': Div
    }

    r_elo = {t: 1500.0 for t in teams}
    eh, ea = [], []
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')

        eh.append(r_elo[h]); ea.append(r_elo[a])

        h_adv = Ho if neutral == 0 else 0.0
        exp_val = max(-100.0, min(100.0, (r_elo[a] - (r_elo[h] + h_adv)) / Div))
        we_h = 1.0 / (1.0 + 10.0 ** exp_val)
        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)

        max_m = params_dict.get('max_margin', None) if params_dict else None
        N = min(abs(gh - ga), int(max_m)) if max_m is not None else abs(gh - ga)
        if N <= 1:
            G = 1.0
        elif N == 2:
            G = G2
        else:
            G = am + bm * float(N)

        K_tier = _get_1elo_complete_ktier(tourn, params_dict)
        K_eff = K_tier * G

        r_elo[h] += K_eff * (sh - we_h)
        r_elo[a] += K_eff * ((1.0 - sh) - (1.0 - we_h))

    df['elo_home'] = eh; df['elo_away'] = ea
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df


def _compute_1elo_g(df, teams, params):
    """
    1eloG System: 1eloS single-scale K-factor (K_base * M_overall if comp else K_base)
    combined with 1eloC's non-linear goal difference margin function (G2, a_margin, b_margin).
    """
    p = params or {}
    Kb = float(p.get('K_base', 35.0))
    Mo = float(p.get('M_overall', 2.0))
    G2 = float(p.get('G2', 1.5))
    am = float(p.get('a_margin', 1.75))
    bm = float(p.get('b_margin', 0.125))
    Ho = float(p.get('H_overall', 100.0))
    Div = float(p.get('divisor_overall', 400.0))

    r_elo = {t: 1500.0 for t in teams}
    eh, ea = [], []
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')
        is_comp = (tourn != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True

        eh.append(r_elo[h]); ea.append(r_elo[a])

        h_adv = Ho if neutral == 0 else 0.0
        exp_val = max(-100.0, min(100.0, (r_elo[a] - (r_elo[h] + h_adv)) / Div))
        we_h = 1.0 / (1.0 + 10.0 ** exp_val)
        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)

        # 1eloC Goal Margin Component
        max_m = params.get('max_margin', None) if params else None
        N = min(abs(gh - ga), int(max_m)) if max_m is not None else abs(gh - ga)
        if N <= 1:
            G = 1.0
        elif N == 2:
            G = G2
        else:
            G = am + bm * float(N)

        # 1eloS K-Factor Component
        k_base = Kb * Mo if is_comp else Kb
        K_eff = k_base * G

        r_elo[h] += K_eff * (sh - we_h)
        r_elo[a] += K_eff * ((1.0 - sh) - (1.0 - we_h))

    df['elo_home'] = eh; df['elo_away'] = ea
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df


def _compute_2elo_g(df, teams, params):
    """
    2eloG System: Dual-scale 1eloG rating system (14 free parameters).
    Produces 2 rating outputs: Rating 1 (scale D1) and Rating 2 (scale D2 = D1 + D_offset_2).
    Forces D2 > D1 to guarantee multi-timescale separation.
    Both ratings use independent 1eloG non-linear goal margin updates G(g_h, g_a).
    """
    p = params or {
        'K_base_1': 35.0, 'M_overall_1': 2.0, 'G2_1': 1.5, 'a_margin_1': 1.0, 'b_margin_1': 0.5, 'H_overall_1': 100.0, 'divisor_1': 400.0,
        'K_base_2': 35.0, 'M_overall_2': 2.0, 'G2_2': 1.5, 'a_margin_2': 1.0, 'b_margin_2': 0.5, 'H_overall_2': 100.0, 'divisor_offset_2': 200.0
    }
    
    div_1 = float(p.get('divisor_1', p.get('D_base_1', p.get('divisor_base_1', 400.0))))
    div_offset_2 = float(p.get('divisor_offset_2', p.get('D_base_2', p.get('divisor_base_2', 200.0))))
    div_2 = max(50.0, div_1 + abs(div_offset_2))
    
    r1 = {t: 1500.0 for t in teams}
    r2 = {t: 1500.0 for t in teams}
    
    r1_h, r1_a, r2_h, r2_a = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        
        r1_h.append(r1[h]); r1_a.append(r1[a])
        r2_h.append(r2[h]); r2_a.append(r2[a])
        
        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        N = abs(gh - ga)
        
        # Goal margin G for Rating 1
        g2_1 = float(p.get('G2_1', p.get('G2', 1.5)))
        am_1 = float(p.get('a_margin_1', p.get('a_margin', 1.0)))
        bm_1 = float(p.get('b_margin_1', p.get('b_margin', 0.5)))
        if N <= 1:
            G_1 = 1.0
        elif N == 2:
            G_1 = g2_1
        else:
            G_1 = am_1 + bm_1 * float(N)
            
        # Goal margin G for Rating 2
        g2_2 = float(p.get('G2_2', p.get('G2', 1.5)))
        am_2 = float(p.get('a_margin_2', p.get('a_margin', 1.0)))
        bm_2 = float(p.get('b_margin_2', p.get('b_margin', 0.5)))
        if N <= 1:
            G_2 = 1.0
        elif N == 2:
            G_2 = g2_2
        else:
            G_2 = am_2 + bm_2 * float(N)
            
        # Update Rating 1
        h1_adv = float(p.get('H_overall_1', 100.0)) if neutral == 0 else 0.0
        exp1 = max(-100.0, min(100.0, (r1[a] - (r1[h] + h1_adv)) / div_1))
        we1 = 1.0 / (1.0 + 10.0 ** exp1)
        k1 = float(p.get('K_base_1', 35.0)) * (float(p.get('M_overall_1', 2.0)) if is_comp else 1.0)
        up1 = k1 * G_1 * (sh - we1)
        r1[h] += up1
        r1[a] -= up1
        
        # Update Rating 2
        h2_adv = float(p.get('H_overall_2', 100.0)) if neutral == 0 else 0.0
        exp2 = max(-100.0, min(100.0, (r2[a] - (r2[h] + h2_adv)) / div_2))
        we2 = 1.0 / (1.0 + 10.0 ** exp2)
        k2 = float(p.get('K_base_2', 35.0)) * (float(p.get('M_overall_2', 2.0)) if is_comp else 1.0)
        up2 = k2 * G_2 * (sh - we2)
        r2[h] += up2
        r2[a] -= up2
        
    df['elo1_home'] = r1_h; df['elo1_away'] = r1_a
    df['elo1_diff'] = np.array(r1_h) - np.array(r1_a)
    
    df['elo2_home'] = r2_h; df['elo2_away'] = r2_a
    df['elo2_diff'] = np.array(r2_h) - np.array(r2_a)
    
    # Primary output rating columns default to Rating 1
    df['elo_home'] = r1_h
    df['elo_away'] = r1_a
    df['elo_diff'] = np.array(r1_h) - np.array(r1_a)
    return df


def _compute_1elo_x(df, teams, params):
    """
    1eloX System: 1eloC's 5 tier-stratified K-factors (K_WC, K_major, K_qual, K_minor, K_friendly)
    combined with 1eloS's simple G=1.0 outcome update (no extra goal margin multiplier).
    """
    p = params or {}
    Ho = float(p.get('H_overall', 100.0))
    Div = float(p.get('divisor_overall', 400.0))

    r_elo = {t: 1500.0 for t in teams}
    eh, ea = [], []
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')

        eh.append(r_elo[h]); ea.append(r_elo[a])

        h_adv = Ho if neutral == 0 else 0.0
        exp_val = max(-100.0, min(100.0, (r_elo[a] - (r_elo[h] + h_adv)) / Div))
        we_h = 1.0 / (1.0 + 10.0 ** exp_val)
        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)

        # 1eloC Tier-Stratified K-Factor
        K_tier = _get_1elo_complete_ktier(tourn, p)

        r_elo[h] += K_tier * (sh - we_h)
        r_elo[a] += K_tier * ((1.0 - sh) - (1.0 - we_h))

    df['elo_home'] = eh; df['elo_away'] = ea
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df


def _compute_2elo_od(df, teams, params):
    """
    Computes 2-Elo Offense+Defense (2eloOD) ratings.
    Tracks decoupled Offensive (R_O) and Defensive (R_D) strength vectors per team.
    
    Expected scoring intensity:
      lambda_h = mu * 10 ** ((O_h - D_a + H_OD) / D_divisor)
      lambda_a = mu * 10 ** ((O_a - (D_h + H_OD)) / D_divisor)
      
    Updates:
      O_h += K * (Y_h - lambda_h)
      D_a += K * (lambda_h - Y_h)
      O_a += K * (Y_a - lambda_a)
      D_h += K * (lambda_a - Y_a)
    """
    p = params or {'K_od': 23.95, 'M_od': 1.0, 'H_od': 93.19, 'divisor_od': 1274.07, 'mu': 1.35}
    Ks = float(p.get('K_od', p.get('K_style', p.get('K_hybrid', 23.95))))
    Ms = float(p.get('M_od', p.get('M_style', 1.0)))
    Hs = float(p.get('H_od', p.get('H_style', 93.19)))
    Ds = float(p.get('divisor_od', p.get('divisor_style', p.get('divisor', 1274.07))))
    mu = float(p.get('mu', 1.35))
    
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}
    oh, oa, dh, da = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        h_adv = Hs if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv) / Ds))
        lh = mu * (10.0 ** exp_h)
        
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv)) / Ds))
        la = mu * (10.0 ** exp_a)
        
        k = Ks * Ms if is_comp else Ks
        r_off[h] += k * (gh - lh)
        r_def[a] += k * (lh - gh)
        r_off[a] += k * (ga - la)
        r_def[h] += k * (la - ga)
        
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = df['diff_off'] - df['diff_def']
    return df


def _compute_2elo_odg(df, teams, params):
    """
    2eloODG System: Offense+Defense dual Elo with 1eloG non-linear goal difference margin G(g_h, g_a).
    Combines decoupled offensive/defensive rating updates with goal margin scaling:
      N = |g_h - g_a|
      G = 1.0 (if N<=1), G2 (if N=2), a_margin + b_margin * N (if N>=3)
    """
    p = params or {
        'K_base': 25.0, 'M_overall': 1.5, 'G2': 1.5, 'a_margin': 1.75, 'b_margin': 0.125,
        'H_overall': 100.0, 'divisor_overall': 400.0, 'mu': 1.35
    }
    Ks = float(p.get('K_base', p.get('K_od', 25.0)))
    Ms = float(p.get('M_overall', p.get('M_od', 1.5)))
    Hs = float(p.get('H_overall', p.get('H_od', 100.0)))
    Ds = float(p.get('divisor_overall', p.get('divisor_od', p.get('divisor', 400.0))))
    mu = float(p.get('mu', 1.35))
    
    g2_v = float(p.get('G2', 1.5))
    am_v = float(p.get('a_margin', 1.75))
    bm_v = float(p.get('b_margin', 0.125))
    
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}
    oh, oa, dh, da = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        max_m = params.get('max_margin', None) if params else None
        N = min(abs(gh - ga), int(max_m)) if max_m is not None else abs(gh - ga)
        if N <= 1:
            G_mult = 1.0
        elif N == 2:
            G_mult = g2_v
        else:
            G_mult = am_v + bm_v * float(N)
            
        h_adv = Hs if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv) / Ds))
        lh = mu * (10.0 ** exp_h)
        
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv)) / Ds))
        la = mu * (10.0 ** exp_a)
        
        k_base = Ks * Ms if is_comp else Ks
        k_eff = k_base * G_mult
        
        r_off[h] += k_eff * (gh - lh)
        r_def[a] += k_eff * (lh - gh)
        r_off[a] += k_eff * (ga - la)
        r_def[h] += k_eff * (la - ga)
        
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = df['diff_off'] - df['diff_def']
    return df


def _compute_2elo_odx(df, teams, params):
    """
    2eloODX System: Decoupled Offense+Defense (2eloOD) ratings with multi-tier K-factors (8 free parameters).
    Tracks decoupled Offensive (R_O) and Defensive (R_D) strength vectors with tournament-tier weighting.
    """
    p = params or {}
    kwc = float(p.get('K_WC_od', p.get('K_WC', 60.0)))
    kmaj = float(p.get('K_major_od', p.get('K_major', 50.0)))
    kqual = float(p.get('K_qual_od', p.get('K_qual', 40.0)))
    kmin = float(p.get('K_minor_od', p.get('K_minor', 30.0)))
    kfri = float(p.get('K_friendly_od', p.get('K_friendly', 20.0)))
    
    Hs = float(p.get('H_od', p.get('H_style', 93.19)))
    Ds = float(p.get('divisor_od', p.get('divisor_style', p.get('divisor', 1274.07))))
    mu = float(p.get('mu', 1.35))
    
    ktier_params = {'K_WC': kwc, 'K_major': kmaj, 'K_qual': kqual, 'K_minor': kmin, 'K_friendly': kfri}
    
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}
    oh, oa, dh, da = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')
        
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        h_adv = Hs if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv) / Ds))
        lh = mu * (10.0 ** exp_h)
        
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv)) / Ds))
        la = mu * (10.0 ** exp_a)
        
        K_tier = _get_1elo_complete_ktier(tourn, ktier_params)
        
        r_off[h] += K_tier * (gh - lh)
        r_def[a] += K_tier * (lh - gh)
        r_off[a] += K_tier * (ga - la)
        r_def[h] += K_tier * (la - ga)
        
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = df['diff_off'] - df['diff_def']
    return df


def _compute_2elo_odc(df, teams, params):
    """
    2eloODC System: Complete Decoupled Offense+Defense (2eloOD) ratings with multi-tier K-factors AND non-linear goal margin G(g_h, g_a) (11 free parameters).
    """
    p = params or {}
    kwc = float(p.get('K_WC_od', p.get('K_WC', 65.0936)))
    kmaj = float(p.get('K_major_od', p.get('K_major', 41.3558)))
    kqual = float(p.get('K_qual_od', p.get('K_qual', 29.7150)))
    kmin = float(p.get('K_minor_od', p.get('K_minor', 28.8660)))
    kfri = float(p.get('K_friendly_od', p.get('K_friendly', 26.8581)))
    
    g2_v = float(p.get('G2_od', p.get('G2', 1.9400)))
    am_v = float(p.get('a_margin_od', p.get('a_margin', 1.3200)))
    bm_v = float(p.get('b_margin_od', p.get('b_margin', 0.0850)))
    
    Hs = float(p.get('H_od', p.get('H_style', 74.1727)))
    Ds = float(p.get('divisor_od', p.get('divisor_style', p.get('divisor', 1935.0339))))
    mu = float(p.get('mu', 1.35))
    
    ktier_params = {'K_WC': kwc, 'K_major': kmaj, 'K_qual': kqual, 'K_minor': kmin, 'K_friendly': kfri}
    
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}
    oh, oa, dh, da = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')
        
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        max_m = ktier_params.get('max_margin', None) if ktier_params else None
        N = min(abs(gh - ga), int(max_m)) if max_m is not None else abs(gh - ga)
        if N <= 1:
            G_mult = 1.0
        elif N == 2:
            G_mult = g2_v
        else:
            G_mult = am_v + bm_v * float(N)
            
        h_adv = Hs if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv) / Ds))
        lh = mu * (10.0 ** exp_h)
        
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv)) / Ds))
        la = mu * (10.0 ** exp_a)
        
        K_tier = _get_1elo_complete_ktier(tourn, ktier_params)
        K_eff = K_tier * G_mult
        
        r_off[h] += K_eff * (gh - lh)
        r_def[a] += K_eff * (lh - gh)
        r_off[a] += K_eff * (ga - la)
        r_def[h] += K_eff * (la - ga)
        
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = df['diff_off'] - df['diff_def']
    return df


def _compute_2elo_fast_slow(df, teams, params):
    p = params or {
        'K_fast': 111.97, 'M_fast': 1.72, 'H_fast': 48.35, 'divisor_fast': 201.04,
        'K_slow': 51.68, 'M_slow': 1.0, 'H_slow': 48.35, 'divisor_slow': 201.04
    }
    div_f = float(p.get('divisor_fast', p.get('D_fast', 201.04)))
    div_s = float(p.get('divisor_slow', p.get('D_slow', 201.04)))
    
    r_fast = {t: 1500.0 for t in teams}
    r_slow = {t: 1500.0 for t in teams}
    fh, fa, sh, sa = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        
        fh.append(r_fast[h]); fa.append(r_fast[a])
        sh.append(r_slow[h]); sa.append(r_slow[a])
        
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        
        # Fast Timescale
        hf_adv = float(p.get('H_fast', 48.35)) if neutral == 0 else 0.0
        exp_f = max(-100.0, min(100.0, (r_fast[a] - (r_fast[h] + hf_adv)) / div_f))
        we_f = 1.0 / (1.0 + 10.0 ** exp_f)
        kf = float(p.get('K_fast', 111.97)) * (float(p.get('M_fast', 1.72)) if is_comp else 1.0)
        r_fast[h] += kf * (s_h - we_f)
        r_fast[a] += kf * ((1.0 - s_h) - (1.0 - we_f))
        
        # Slow Timescale
        hs_adv = float(p.get('H_slow', 48.35)) if neutral == 0 else 0.0
        exp_s = max(-100.0, min(100.0, (r_slow[a] - (r_slow[h] + hs_adv)) / div_s))
        we_s = 1.0 / (1.0 + 10.0 ** exp_s)
        ks = float(p.get('K_slow', 51.68)) * (float(p.get('M_slow', 1.0)) if is_comp else 1.0)
        r_slow[h] += ks * (s_h - we_s)
        r_slow[a] += ks * ((1.0 - s_h) - (1.0 - we_s))
        
    df['fast_home'] = fh; df['fast_away'] = fa
    df['slow_home'] = sh; df['slow_away'] = sa
    
    df['elo_home_fast'] = fh; df['elo_away_fast'] = fa
    df['elo_diff_fast'] = np.array(fh) - np.array(fa)
    
    df['elo_home_slow'] = sh; df['elo_away_slow'] = sa
    df['elo_diff_slow'] = np.array(sh) - np.array(sa)
    
    # Primary output rating columns default to fast rating
    df['elo_home'] = fh
    df['elo_away'] = fa
    df['elo_diff'] = np.array(fh) - np.array(fa)
    return df


def _compute_2elo_fsk(df, teams, params):
    """
    2eloFSK System: Dual-Timescale Fast+Slow Elo WITH 5-tier K-factors, WITHOUT Goal Margin G (14 free parameters).
    """
    p = params or {}
    
    # Fast K-tiers
    kwc_f = float(p.get('K_WC_fast', p.get('K_WC_f', p.get('K_WC', 60.0))))
    kmaj_f = float(p.get('K_major_fast', p.get('K_major_f', p.get('K_major', 50.0))))
    kqual_f = float(p.get('K_qual_fast', p.get('K_qual_f', p.get('K_qual', 40.0))))
    kmin_f = float(p.get('K_minor_fast', p.get('K_minor_f', p.get('K_minor', 30.0))))
    kfri_f = float(p.get('K_friendly_fast', p.get('K_friendly_f', p.get('K_friendly', 20.0))))
    Hf = float(p.get('H_fast', p.get('H_f', p.get('H_overall', 100.0))))
    Div_f = float(p.get('divisor_fast', p.get('divisor_f', p.get('divisor_overall', 400.0))))
    ktier_f_dict = {'K_WC': kwc_f, 'K_major': kmaj_f, 'K_qual': kqual_f, 'K_minor': kmin_f, 'K_friendly': kfri_f}
    
    # Slow K-tiers
    kwc_s = float(p.get('K_WC_slow', p.get('K_WC_s', p.get('K_WC', 30.0))))
    kmaj_s = float(p.get('K_major_slow', p.get('K_major_s', p.get('K_major', 25.0))))
    kqual_s = float(p.get('K_qual_slow', p.get('K_qual_s', p.get('K_qual', 20.0))))
    kmin_s = float(p.get('K_minor_slow', p.get('K_minor_s', p.get('K_minor', 15.0))))
    kfri_s = float(p.get('K_friendly_slow', p.get('K_friendly_s', p.get('K_friendly', 10.0))))
    Hs = float(p.get('H_slow', p.get('H_s', p.get('H_overall', 100.0))))
    Div_s = float(p.get('divisor_slow', p.get('divisor_s', p.get('divisor_overall', 400.0))))
    ktier_s_dict = {'K_WC': kwc_s, 'K_major': kmaj_s, 'K_qual': kqual_s, 'K_minor': kmin_s, 'K_friendly': kfri_s}
    
    r_fast = {t: 1500.0 for t in teams}
    r_slow = {t: 1500.0 for t in teams}
    fh, fa, sh, sa = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')
        
        fh.append(r_fast[h]); fa.append(r_fast[a])
        sh.append(r_slow[h]); sa.append(r_slow[a])
        
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        
        # Fast Update (WITH 5-tier K, NO G)
        h_adv_f = Hf if neutral == 0 else 0.0
        exp_f = max(-100.0, min(100.0, (r_fast[a] - (r_fast[h] + h_adv_f)) / Div_f))
        we_f = 1.0 / (1.0 + 10.0 ** exp_f)
        K_tier_f = _get_1elo_complete_ktier(tourn, ktier_f_dict)
        r_fast[h] += K_tier_f * (s_h - we_f)
        r_fast[a] += K_tier_f * ((1.0 - s_h) - (1.0 - we_f))
        
        # Slow Update (WITH 5-tier K, NO G)
        h_adv_s = Hs if neutral == 0 else 0.0
        exp_s = max(-100.0, min(100.0, (r_slow[a] - (r_slow[h] + h_adv_s)) / Div_s))
        we_s = 1.0 / (1.0 + 10.0 ** exp_s)
        K_tier_s = _get_1elo_complete_ktier(tourn, ktier_s_dict)
        r_slow[h] += K_tier_s * (s_h - we_s)
        r_slow[a] += K_tier_s * ((1.0 - s_h) - (1.0 - we_s))
        
    df['fast_home'] = fh; df['fast_away'] = fa
    df['slow_home'] = sh; df['slow_away'] = sa
    df['elo_home_fast'] = fh; df['elo_away_fast'] = fa
    df['elo_diff_fast'] = np.array(fh) - np.array(fa)
    df['elo_home_slow'] = sh; df['elo_away_slow'] = sa
    df['elo_diff_slow'] = np.array(sh) - np.array(sa)
    df['elo_home'] = fh; df['elo_away'] = fa
    df['elo_diff'] = np.array(fh) - np.array(fa)
    return df


def _compute_2elo_fsg(df, teams, params):
    """
    2eloFSG System: Dual-Timescale Fast+Slow Elo WITHOUT 5-tier K-factors, WITH Goal Margin G (14 free parameters).
    """
    p = params or {}
    
    # Fast parameters
    kf_base = float(p.get('K_fast', p.get('K_base_fast', 111.97)))
    mf_comp = float(p.get('M_fast', p.get('M_overall_fast', 1.72)))
    g2_f = float(p.get('G2_fast', p.get('G2_f', p.get('G2', 1.5))))
    am_f = float(p.get('a_margin_fast', p.get('a_margin_f', p.get('a_margin', 1.75))))
    bm_f = float(p.get('b_margin_fast', p.get('b_margin_f', p.get('b_margin', 0.125))))
    Hf = float(p.get('H_fast', p.get('H_f', p.get('H_overall', 100.0))))
    Div_f = float(p.get('divisor_fast', p.get('divisor_f', p.get('divisor_overall', 400.0))))
    
    # Slow parameters
    ks_base = float(p.get('K_slow', p.get('K_base_slow', 51.68)))
    ms_comp = float(p.get('M_slow', p.get('M_overall_slow', 1.0)))
    g2_s = float(p.get('G2_slow', p.get('G2_s', p.get('G2', 1.5))))
    am_s = float(p.get('a_margin_slow', p.get('a_margin_s', p.get('a_margin', 1.75))))
    bm_s = float(p.get('b_margin_slow', p.get('b_margin_s', p.get('b_margin', 0.125))))
    Hs = float(p.get('H_slow', p.get('H_s', p.get('H_overall', 100.0))))
    Div_s = float(p.get('divisor_slow', p.get('divisor_s', p.get('divisor_overall', 400.0))))
    
    r_fast = {t: 1500.0 for t in teams}
    r_slow = {t: 1500.0 for t in teams}
    fh, fa, sh, sa = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        
        fh.append(r_fast[h]); fa.append(r_fast[a])
        sh.append(r_slow[h]); sa.append(r_slow[a])
        
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        max_m = params.get('max_margin', None) if params else None
        N = min(abs(gh - ga), int(max_m)) if max_m is not None else abs(gh - ga)
        
        # Fast Update (NO 5-tier K, WITH G)
        G_f = 1.0 if N <= 1 else (g2_f if N == 2 else am_f + bm_f * float(N))
        h_adv_f = Hf if neutral == 0 else 0.0
        exp_f = max(-100.0, min(100.0, (r_fast[a] - (r_fast[h] + h_adv_f)) / Div_f))
        we_f = 1.0 / (1.0 + 10.0 ** exp_f)
        kf = kf_base * (mf_comp if is_comp else 1.0) * G_f
        r_fast[h] += kf * (s_h - we_f)
        r_fast[a] += kf * ((1.0 - s_h) - (1.0 - we_f))
        
        # Slow Update (NO 5-tier K, WITH G)
        G_s = 1.0 if N <= 1 else (g2_s if N == 2 else am_s + bm_s * float(N))
        h_adv_s = Hs if neutral == 0 else 0.0
        exp_s = max(-100.0, min(100.0, (r_slow[a] - (r_slow[h] + h_adv_s)) / Div_s))
        we_s = 1.0 / (1.0 + 10.0 ** exp_s)
        ks = ks_base * (ms_comp if is_comp else 1.0) * G_s
        r_slow[h] += ks * (s_h - we_s)
        r_slow[a] += ks * ((1.0 - s_h) - (1.0 - we_s))
        
    df['fast_home'] = fh; df['fast_away'] = fa
    df['slow_home'] = sh; df['slow_away'] = sa
    df['elo_home_fast'] = fh; df['elo_away_fast'] = fa
    df['elo_diff_fast'] = np.array(fh) - np.array(fa)
    df['elo_home_slow'] = sh; df['elo_away_slow'] = sa
    df['elo_diff_slow'] = np.array(sh) - np.array(sa)
    df['elo_home'] = fh; df['elo_away'] = fa
    df['elo_diff'] = np.array(fh) - np.array(fa)
    return df


def _compute_2elo_fsc(df, teams, params):
    """
    2eloFSC System: Complete Dual-Timescale Fast+Slow Elo combining 5-tier K-factors and non-linear goal difference margin G on both timescales (20 free parameters).
    """
    p = params or {}
    
    # Fast parameters
    kwc_f = float(p.get('K_WC_fast', p.get('K_WC_f', p.get('K_WC', 60.0))))
    kmaj_f = float(p.get('K_major_fast', p.get('K_major_f', p.get('K_major', 50.0))))
    kqual_f = float(p.get('K_qual_fast', p.get('K_qual_f', p.get('K_qual', 40.0))))
    kmin_f = float(p.get('K_minor_fast', p.get('K_minor_f', p.get('K_minor', 30.0))))
    kfri_f = float(p.get('K_friendly_fast', p.get('K_friendly_f', p.get('K_friendly', 20.0))))
    
    g2_f = float(p.get('G2_fast', p.get('G2_f', p.get('G2', 1.5))))
    am_f = float(p.get('a_margin_fast', p.get('a_margin_f', p.get('a_margin', 1.75))))
    bm_f = float(p.get('b_margin_fast', p.get('b_margin_f', p.get('b_margin', 0.125))))
    
    Hf = float(p.get('H_fast', p.get('H_f', p.get('H_overall', 100.0))))
    Div_f = float(p.get('divisor_fast', p.get('divisor_f', p.get('divisor_overall', 400.0))))
    
    ktier_f_dict = {'K_WC': kwc_f, 'K_major': kmaj_f, 'K_qual': kqual_f, 'K_minor': kmin_f, 'K_friendly': kfri_f}
    
    # Slow parameters
    kwc_s = float(p.get('K_WC_slow', p.get('K_WC_s', p.get('K_WC', 30.0))))
    kmaj_s = float(p.get('K_major_slow', p.get('K_major_s', p.get('K_major', 25.0))))
    kqual_s = float(p.get('K_qual_slow', p.get('K_qual_s', p.get('K_qual', 20.0))))
    kmin_s = float(p.get('K_minor_slow', p.get('K_minor_s', p.get('K_minor', 15.0))))
    kfri_s = float(p.get('K_friendly_slow', p.get('K_friendly_s', p.get('K_friendly', 10.0))))
    
    g2_s = float(p.get('G2_slow', p.get('G2_s', p.get('G2', 1.5))))
    am_s = float(p.get('a_margin_slow', p.get('a_margin_s', p.get('a_margin', 1.75))))
    bm_s = float(p.get('b_margin_slow', p.get('b_margin_s', p.get('b_margin', 0.125))))
    
    Hs = float(p.get('H_slow', p.get('H_s', p.get('H_overall', 100.0))))
    Div_s = float(p.get('divisor_slow', p.get('divisor_s', p.get('divisor_overall', 400.0))))
    
    ktier_s_dict = {'K_WC': kwc_s, 'K_major': kmaj_s, 'K_qual': kqual_s, 'K_minor': kmin_s, 'K_friendly': kfri_s}
    
    r_fast = {t: 1500.0 for t in teams}
    r_slow = {t: 1500.0 for t in teams}
    fh, fa, sh, sa = [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')
        
        fh.append(r_fast[h]); fa.append(r_fast[a])
        sh.append(r_slow[h]); sa.append(r_slow[a])
        
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        N = abs(gh - ga)
        
        # 1. Fast Update (WITH Multi-Tier K & Goal Margin G)
        G_f = 1.0 if N <= 1 else (g2_f if N == 2 else am_f + bm_f * float(N))
        h_adv_f = Hf if neutral == 0 else 0.0
        exp_f = max(-100.0, min(100.0, (r_fast[a] - (r_fast[h] + h_adv_f)) / Div_f))
        we_f = 1.0 / (1.0 + 10.0 ** exp_f)
        K_tier_f = _get_1elo_complete_ktier(tourn, ktier_f_dict)
        K_eff_f = K_tier_f * G_f
        
        r_fast[h] += K_eff_f * (s_h - we_f)
        r_fast[a] += K_eff_f * ((1.0 - s_h) - (1.0 - we_f))
        
        # 2. Slow Update (WITH Multi-Tier K & Goal Margin G)
        G_s = 1.0 if N <= 1 else (g2_s if N == 2 else am_s + bm_s * float(N))
        h_adv_s = Hs if neutral == 0 else 0.0
        exp_s = max(-100.0, min(100.0, (r_slow[a] - (r_slow[h] + h_adv_s)) / Div_s))
        we_s = 1.0 / (1.0 + 10.0 ** exp_s)
        K_tier_s = _get_1elo_complete_ktier(tourn, ktier_s_dict)
        K_eff_s = K_tier_s * G_s
        
        r_slow[h] += K_eff_s * (s_h - we_s)
        r_slow[a] += K_eff_s * ((1.0 - s_h) - (1.0 - we_s))
        
    df['fast_home'] = fh; df['fast_away'] = fa
    df['slow_home'] = sh; df['slow_away'] = sa
    
    df['elo_home_fast'] = fh; df['elo_away_fast'] = fa
    df['elo_diff_fast'] = np.array(fh) - np.array(fa)
    
    df['elo_home_slow'] = sh; df['elo_away_slow'] = sa
    df['elo_diff_slow'] = np.array(sh) - np.array(sa)
    
    df['elo_home'] = fh
    df['elo_away'] = fa
    df['elo_diff'] = np.array(fh) - np.array(fa)
    return df


def _compute_3elo_hybrid(df, teams, params):
    p = params or {
        'K_base': 29.6457, 'M_overall': 2.4302, 'H_overall': 294.0188, 'divisor_overall': 327.4699,
        'M_style': 0.9687, 'H_style': 24.6170, 'divisor_style': 885.7728, 'K_scale': 0.5582, 'mu': 1.35
    }
    Kb = float(p.get('K_base', 29.6457))
    Mo = float(p.get('M_overall', 2.4302))
    Ho = float(p.get('H_overall', 294.0188))
    Div_o = float(p.get('divisor_overall', 327.4699))
    Ms = float(p.get('M_style', 0.9687))
    Hs = float(p.get('H_style', 24.6170))
    Ds = float(p.get('divisor_style', 885.7728))
    Ks = float(p.get('K_scale', 0.5582))
    mu = float(p.get('mu', 1.35))
    
    r_overall = {t: 1500.0 for t in teams}
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}
    
    eh, ea, oh, oa, dh, da = [], [], [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        
        eh.append(r_overall[h]); ea.append(r_overall[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        h_adv_o = Ho if neutral == 0 else 0.0
        exp_o = max(-100.0, min(100.0, (r_overall[a] - (r_overall[h] + h_adv_o)) / Div_o))
        we_h = 1.0 / (1.0 + 10.0 ** exp_o)
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        ko = Kb * Mo if is_comp else Kb
        r_overall[h] += ko * (s_h - we_h)
        r_overall[a] += ko * ((1.0 - s_h) - (1.0 - we_h))
        
        h_adv_s = Hs if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv_s) / Ds))
        lh = mu * (10.0 ** exp_h)
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv_s)) / Ds))
        la = mu * (10.0 ** exp_a)
        
        k_style = (Kb * Ms if is_comp else Kb) * Ks
        r_off[h] += k_style * (gh - lh)
        r_def[a] += k_style * (lh - gh)
        r_off[a] += k_style * (ga - la)
        r_def[h] += k_style * (la - ga)
        
    df['elo_home'] = eh; df['elo_away'] = ea
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = np.array(eh) - np.array(ea)
    df['elo_diff_style'] = df['diff_off'] - df['diff_def']
    return df


def _compute_3elo_complete(df, teams, params):
    p = params or {}
    has_ktiers = any(k in p for k in ['K_WC', 'K_major', 'K_qual', 'K_minor', 'K_friendly'])
    
    Kb = float(p.get('K_base', 32.05371))
    Mo = float(p.get('M_overall', 2.501047))
    Ho = float(p.get('H_overall', 218.209652))
    Div_o = float(p.get('divisor_overall', 1267.582933))
    
    G2 = float(p.get('G2', 1.097433))
    am = float(p.get('a_margin', 4.32698))
    bm = float(p.get('b_margin', 3.603202))
    
    Ms = float(p.get('M_style', 1.072913))
    Hs = float(p.get('H_style', 60.945526))
    Ds = float(p.get('divisor_style', 974.553479))
    Ks = float(p.get('K_scale', 0.552142))
    mu = float(p.get('mu', 1.35))
    
    r_overall = {t: 1500.0 for t in teams}
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}
    
    eh, ea, oh, oa, dh, da = [], [], [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')
        is_comp = (str(tourn).lower() != 'friendly')
        
        eh.append(r_overall[h]); ea.append(r_overall[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        # Overall Outcome Update
        h_adv_o = Ho if neutral == 0 else 0.0
        exp_o = max(-100.0, min(100.0, (r_overall[a] - (r_overall[h] + h_adv_o)) / Div_o))
        we_h = 1.0 / (1.0 + 10.0 ** exp_o)
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        
        diff = abs(gh - ga)
        if diff <= 1:
            G = 1.0
        elif diff == 2:
            G = G2
        else:
            G = (am + diff) / max(1e-5, bm) if bm > 0.5 else am + (diff - 3.0) / max(1e-5, bm)
            
        if has_ktiers:
            ktier = _get_1elo_complete_ktier(tourn, p)
        else:
            ktier = Kb * Mo if is_comp else Kb
            
        ko = ktier * G
        r_overall[h] += ko * (s_h - we_h)
        r_overall[a] += ko * ((1.0 - s_h) - (1.0 - we_h))
        
        # Offense/Defense Style Update
        h_adv_s = Hs if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv_s) / Ds))
        lh = mu * (10.0 ** exp_h)
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv_s)) / Ds))
        la = mu * (10.0 ** exp_a)
        
        k_base_style = ktier if has_ktiers else Kb
        k_style = (k_base_style * Ms if is_comp else k_base_style) * Ks
        
        r_off[h] += k_style * (gh - lh)
        r_def[a] += k_style * (lh - gh)
        r_off[a] += k_style * (ga - la)
        r_def[h] += k_style * (la - ga)
        
    df['elo_home'] = eh; df['elo_away'] = ea
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = np.array(eh) - np.array(ea)
    df['elo_diff_style'] = df['diff_off'] - df['diff_def']
    return df


def _compute_4elo_multiscale(df, teams, params):
    """
    Dual-Timescale Style Matrix Rating (4-Elo Multi-Scale).
    Combines 2-Elo Fast+Slow (fast form outcome + slow structural outcome) with 2-Elo Style (offense + defense goal capacity).
    12 Parameters:
      Fast: K_fast, M_fast, H_fast, divisor_fast
      Slow: K_slow, M_slow, H_slow, divisor_slow
      Style: M_style, H_style, divisor_style, K_scale
    """
    p = params or {}
    
    Kf = float(p.get('K_fast', 45.0))
    Mf = float(p.get('M_fast', 1.5))
    Hf = float(p.get('H_fast', 120.0))
    Df = float(p.get('divisor_fast', 350.0))
    
    Ks_slow = float(p.get('K_slow', 15.0))
    Ms_slow = float(p.get('M_slow', 2.0))
    Hs_slow = float(p.get('H_slow', 250.0))
    Ds_slow = float(p.get('divisor_slow', 500.0))
    
    Ms_style = float(p.get('M_style', 1.0))
    Hs_style = float(p.get('H_style', 60.0))
    Ds_style = float(p.get('divisor_style', 950.0))
    Kscale = float(p.get('K_scale', 0.55))
    mu = float(p.get('mu', 1.35))
    
    r_fast = {t: 1500.0 for t in teams}
    r_slow = {t: 1500.0 for t in teams}
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}
    
    fh, fa, sh, sa, oh, oa, dh, da = [], [], [], [], [], [], [], []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        tourn = getattr(row, 'tournament', 'Friendly')
        is_comp = (str(tourn).lower() != 'friendly')
        
        fh.append(r_fast[h]); fa.append(r_fast[a])
        sh.append(r_slow[h]); sa.append(r_slow[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        
        # 1. Fast Outcome Update
        h_adv_f = Hf if neutral == 0 else 0.0
        exp_f = max(-100.0, min(100.0, (r_fast[a] - (r_fast[h] + h_adv_f)) / Df))
        we_f = 1.0 / (1.0 + 10.0 ** exp_f)
        k_f = Kf * Mf if is_comp else Kf
        r_fast[h] += k_f * (s_h - we_f)
        r_fast[a] += k_f * ((1.0 - s_h) - (1.0 - we_f))
        
        # 2. Slow Outcome Update
        h_adv_s_slow = Hs_slow if neutral == 0 else 0.0
        exp_s = max(-100.0, min(100.0, (r_slow[a] - (r_slow[h] + h_adv_s_slow)) / Ds_slow))
        we_s = 1.0 / (1.0 + 10.0 ** exp_s)
        k_s = Ks_slow * Ms_slow if is_comp else Ks_slow
        r_slow[h] += k_s * (s_h - we_s)
        r_slow[a] += k_s * ((1.0 - s_h) - (1.0 - we_s))
        
        # 3. Offense/Defense Style Update
        h_adv_st = Hs_style if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv_st) / Ds_style))
        lh = mu * (10.0 ** exp_h)
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv_st)) / Ds_style))
        la = mu * (10.0 ** exp_a)
        
        k_style = (Kf * Ms_style if is_comp else Kf) * Kscale
        r_off[h] += k_style * (gh - lh)
        r_def[a] += k_style * (lh - gh)
        r_off[a] += k_style * (ga - la)
        r_def[h] += k_style * (la - ga)
        
    df['elo_home_fast'] = fh; df['elo_away_fast'] = fa
    df['elo_fast_home'] = fh; df['elo_fast_away'] = fa
    
    df['elo_home_slow'] = sh; df['elo_away_slow'] = sa
    df['elo_slow_home'] = sh; df['elo_slow_away'] = sa
    
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff_fast'] = np.array(fh) - np.array(fa)
    df['elo_diff_slow'] = np.array(sh) - np.array(sa)
    df['elo_diff'] = df['elo_diff_fast']
    df['elo_diff_style'] = df['diff_off'] - df['diff_def']
    return df


def _compute_3elo_odg(df, teams, params):
    """
    3eloODG System: Mixture of 1eloG (Overall Outcome Goal Margin Elo) and 2eloODG (Decoupled Offense/Defense Goal Margin Elo).
    Tracks 3 rating vectors: R_overall, R_off, R_def.
    Both overall outcome updates and decoupled O/D updates use non-linear goal margin multipliers G(g_h, g_a).
    """
    p = params or {
        'K_base': 35.0, 'M_overall': 2.0, 'G2': 1.5, 'a_margin': 1.75, 'b_margin': 0.125,
        'H_overall': 100.0, 'divisor_overall': 400.0,
        'K_od': 25.0, 'M_od': 1.5, 'G2_od': 1.5, 'a_margin_od': 1.75, 'b_margin_od': 0.125,
        'H_od': 100.0, 'divisor_od': 400.0, 'mu': 1.35
    }
    # 1eloG Overall parameters
    Kb_o = float(p.get('K_base', 35.0))
    Mo_o = float(p.get('M_overall', 2.0))
    Ho_o = float(p.get('H_overall', 100.0))
    Div_o = float(p.get('divisor_overall', 400.0))
    g2_o = float(p.get('G2', 1.5))
    am_o = float(p.get('a_margin', 1.75))
    bm_o = float(p.get('b_margin', 0.125))

    # 2eloODG Offense/Defense parameters
    Ks_od = float(p.get('K_od', p.get('K_base_od', 25.0)))
    Ms_od = float(p.get('M_od', p.get('M_overall_od', 1.5)))
    Hs_od = float(p.get('H_od', p.get('H_overall_od', 100.0)))
    Ds_od = float(p.get('divisor_od', p.get('divisor_overall_od', 400.0)))
    g2_od = float(p.get('G2_od', g2_o))
    am_od = float(p.get('a_margin_od', am_o))
    bm_od = float(p.get('b_margin_od', bm_o))
    mu = float(p.get('mu', 1.35))

    r_overall = {t: 1500.0 for t in teams}
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}

    eh, ea, oh, oa, dh, da = [], [], [], [], [], []

    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True

        eh.append(r_overall[h]); ea.append(r_overall[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])

        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        N = abs(gh - ga)

        # Goal Margin G for Overall 1eloG
        if N <= 1:
            G_o = 1.0
        elif N == 2:
            G_o = g2_o
        else:
            G_o = am_o + bm_o * float(N)

        # Update Overall 1eloG
        h_adv_o = Ho_o if neutral == 0 else 0.0
        exp_o = max(-100.0, min(100.0, (r_overall[a] - (r_overall[h] + h_adv_o)) / Div_o))
        we_h = 1.0 / (1.0 + 10.0 ** exp_o)
        ko = Kb_o * Mo_o if is_comp else Kb_o
        r_overall[h] += ko * G_o * (sh - we_h)
        r_overall[a] += ko * G_o * ((1.0 - sh) - (1.0 - we_h))

        # Goal Margin G for Offense/Defense 2eloODG
        if N <= 1:
            G_od = 1.0
        elif N == 2:
            G_od = g2_od
        else:
            G_od = am_od + bm_od * float(N)

        # Update Decoupled 2eloODG
        h_adv_od = Hs_od if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv_od) / Ds_od))
        lh = mu * (10.0 ** exp_h)
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv_od)) / Ds_od))
        la = mu * (10.0 ** exp_a)

        k_od = (Ks_od * Ms_od if is_comp else Ks_od) * G_od
        r_off[h] += k_od * (gh - lh)
        r_def[a] += k_od * (lh - gh)
        r_off[a] += k_od * (ga - la)
        r_def[h] += k_od * (la - ga)

    df['elo_home'] = eh; df['elo_away'] = ea
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_overall'] = np.array(eh) - np.array(ea)
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = df['diff_overall']
    return df


def _compute_4elo_g(df, teams, params):
    """
    4eloG System (4eloOD+2G): Combination of 2eloG (Dual-Timescale Goal Margin Elo) and 2eloODG (Decoupled Offense/Defense Goal Margin Elo).
    Tracks 4 rating vectors: R_1, R_2, R_off, R_def.
    Outputs: elo1_diff, elo2_diff, diff_off, diff_def.
    """
    p = params or {
        'K_base_1': 35.0, 'M_overall_1': 2.0, 'G2_1': 1.5, 'a_margin_1': 1.75, 'b_margin_1': 0.125, 'H_overall_1': 100.0, 'divisor_1': 400.0,
        'K_base_2': 35.0, 'M_overall_2': 2.0, 'G2_2': 1.5, 'a_margin_2': 1.75, 'b_margin_2': 0.125, 'H_overall_2': 100.0, 'divisor_offset_2': 200.0,
        'K_od': 25.0, 'M_od': 1.5, 'G2_od': 1.5, 'a_margin_od': 1.75, 'b_margin_od': 0.125, 'H_od': 100.0, 'divisor_od': 400.0, 'mu': 1.35
    }
    # 2eloG Rating 1
    Kb_1 = float(p.get('K_base_1', 35.0)); Mo_1 = float(p.get('M_overall_1', 2.0))
    g2_1 = float(p.get('G2_1', 1.5)); am_1 = float(p.get('a_margin_1', 1.75)); bm_1 = float(p.get('b_margin_1', 0.125))
    Ho_1 = float(p.get('H_overall_1', 100.0)); Div_1 = float(p.get('divisor_1', 400.0))

    # 2eloG Rating 2
    Kb_2 = float(p.get('K_base_2', 35.0)); Mo_2 = float(p.get('M_overall_2', 2.0))
    g2_2 = float(p.get('G2_2', 1.5)); am_2 = float(p.get('a_margin_2', 1.75)); bm_2 = float(p.get('b_margin_2', 0.125))
    Ho_2 = float(p.get('H_overall_2', 100.0)); Div_2 = Div_1 + float(p.get('divisor_offset_2', 200.0))

    # 2eloODG Offense/Defense
    Ks_od = float(p.get('K_od', 25.0)); Ms_od = float(p.get('M_od', 1.5))
    g2_od = float(p.get('G2_od', 1.5)); am_od = float(p.get('a_margin_od', 1.75)); bm_od = float(p.get('b_margin_od', 0.125))
    Hs_od = float(p.get('H_od', 100.0)); Ds_od = float(p.get('divisor_od', 400.0)); mu = float(p.get('mu', 1.35))

    r1 = {t: 1500.0 for t in teams}; r2 = {t: 1500.0 for t in teams}
    r_off = {t: 1500.0 for t in teams}; r_def = {t: 1500.0 for t in teams}

    e1h, e1a, e2h, e2a, oh, oa, dh, da = [], [], [], [], [], [], [], []

    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True

        e1h.append(r1[h]); e1a.append(r1[a])
        e2h.append(r2[h]); e2a.append(r2[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])

        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        N = abs(gh - ga)

        # 1. 2eloG Update - Rating 1
        G_1 = 1.0 if N <= 1 else (g2_1 if N == 2 else am_1 + bm_1 * float(N))
        h_adv1 = Ho_1 if neutral == 0 else 0.0
        exp1 = max(-100.0, min(100.0, (r1[a] - (r1[h] + h_adv1)) / Div_1))
        we1_h = 1.0 / (1.0 + 10.0 ** exp1)
        k1 = Kb_1 * Mo_1 if is_comp else Kb_1
        r1[h] += k1 * G_1 * (sh - we1_h)
        r1[a] += k1 * G_1 * ((1.0 - sh) - (1.0 - we1_h))

        # 2. 2eloG Update - Rating 2
        G_2 = 1.0 if N <= 1 else (g2_2 if N == 2 else am_2 + bm_2 * float(N))
        h_adv2 = Ho_2 if neutral == 0 else 0.0
        exp2 = max(-100.0, min(100.0, (r2[a] - (r2[h] + h_adv2)) / Div_2))
        we2_h = 1.0 / (1.0 + 10.0 ** exp2)
        k2 = Kb_2 * Mo_2 if is_comp else Kb_2
        r2[h] += k2 * G_2 * (sh - we2_h)
        r2[a] += k2 * G_2 * ((1.0 - sh) - (1.0 - we2_h))

        # 3. 2eloODG Update - Offense / Defense
        G_od = 1.0 if N <= 1 else (g2_od if N == 2 else am_od + bm_od * float(N))
        h_adv_od = Hs_od if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv_od) / Ds_od))
        lh = mu * (10.0 ** exp_h)
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv_od)) / Ds_od))
        la = mu * (10.0 ** exp_a)

        k_od = (Ks_od * Ms_od if is_comp else Ks_od) * G_od
        r_off[h] += k_od * (gh - lh)
        r_def[a] += k_od * (lh - gh)
        r_off[a] += k_od * (ga - la)
        r_def[h] += k_od * (la - ga)

    df['elo1_home'] = e1h; df['elo1_away'] = e1a
    df['elo2_home'] = e2h; df['elo2_away'] = e2a
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da

    df['elo1_diff'] = np.array(e1h) - np.array(e1a)
    df['elo2_diff'] = np.array(e2h) - np.array(e2a)
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)

    df['elo_home'] = e1h; df['elo_away'] = e1a
    df['elo_diff'] = df['elo1_diff']
    return df


def _compute_3elo_od_1g(df, teams, params):
    """
    3eloOD+1G System: Hybrid of Pure 2eloOD (Offense/Defense, no G multiplier) + 1eloG (Overall Goal Margin Elo).
    Tracks 3 rating vectors: R_overall, R_off, R_def.
    Overall outcome update uses non-linear goal margin multiplier G(g_h, g_a).
    Offense/Defense update uses standard expected goals (pure 2eloOD).
    """
    p = params or {
        'K_base': 35.0, 'M_overall': 2.0, 'G2': 1.5, 'a_margin': 1.75, 'b_margin': 0.125,
        'H_overall': 100.0, 'divisor_overall': 400.0,
        'K_od': 25.0, 'M_od': 1.5, 'H_od': 100.0, 'divisor_od': 400.0, 'mu': 1.35
    }
    Kb_o = float(p.get('K_base', 35.0)); Mo_o = float(p.get('M_overall', 2.0))
    Ho_o = float(p.get('H_overall', 100.0)); Div_o = float(p.get('divisor_overall', 400.0))
    g2_o = float(p.get('G2', 1.5)); am_o = float(p.get('a_margin', 1.75)); bm_o = float(p.get('b_margin', 0.125))

    Ks_od = float(p.get('K_od', 25.0)); Ms_od = float(p.get('M_od', 1.5))
    Hs_od = float(p.get('H_od', 100.0)); Ds_od = float(p.get('divisor_od', 400.0)); mu = float(p.get('mu', 1.35))

    r_overall = {t: 1500.0 for t in teams}
    r_off = {t: 1500.0 for t in teams}
    r_def = {t: 1500.0 for t in teams}

    eh, ea, oh, oa, dh, da = [], [], [], [], [], []

    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True

        eh.append(r_overall[h]); ea.append(r_overall[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])

        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        N = abs(gh - ga)

        # 1. Update Overall 1eloG (WITH Goal Margin G)
        G_o = 1.0 if N <= 1 else (g2_o if N == 2 else am_o + bm_o * float(N))
        h_adv_o = Ho_o if neutral == 0 else 0.0
        exp_o = max(-100.0, min(100.0, (r_overall[a] - (r_overall[h] + h_adv_o)) / Div_o))
        we_h = 1.0 / (1.0 + 10.0 ** exp_o)
        ko = Kb_o * Mo_o if is_comp else Kb_o
        r_overall[h] += ko * G_o * (sh - we_h)
        r_overall[a] += ko * G_o * ((1.0 - sh) - (1.0 - we_h))

        # 2. Update Decoupled 2eloOD (PURE, NO G)
        h_adv_od = Hs_od if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv_od) / Ds_od))
        lh = mu * (10.0 ** exp_h)
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv_od)) / Ds_od))
        la = mu * (10.0 ** exp_a)

        k_od = Ks_od * Ms_od if is_comp else Ks_od
        r_off[h] += k_od * (gh - lh)
        r_def[a] += k_od * (lh - gh)
        r_off[a] += k_od * (ga - la)
        r_def[h] += k_od * (la - ga)

    df['elo_home'] = eh; df['elo_away'] = ea
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da
    df['diff_overall'] = np.array(eh) - np.array(ea)
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)
    df['elo_diff'] = df['diff_overall']
    return df


def _compute_4elo_od_2g(df, teams, params):
    """
    4eloOD+2G System: Combination of Pure 2eloOD (Offense/Defense, no G multiplier) + 2eloG (Dual-Timescale Goal Margin Elo).
    Tracks 4 rating vectors: R_1, R_2, R_off, R_def.
    Outputs: elo1_diff, elo2_diff, diff_off, diff_def.
    """
    p = params or {
        'K_base_1': 35.0, 'M_overall_1': 2.0, 'G2_1': 1.5, 'a_margin_1': 1.75, 'b_margin_1': 0.125, 'H_overall_1': 100.0, 'divisor_1': 400.0,
        'K_base_2': 35.0, 'M_overall_2': 2.0, 'G2_2': 1.5, 'a_margin_2': 1.75, 'b_margin_2': 0.125, 'H_overall_2': 100.0, 'divisor_offset_2': 200.0,
        'K_od': 25.0, 'M_od': 1.5, 'H_od': 100.0, 'divisor_od': 400.0, 'mu': 1.35
    }
    # 2eloG Rating 1
    Kb_1 = float(p.get('K_base_1', 35.0)); Mo_1 = float(p.get('M_overall_1', 2.0))
    g2_1 = float(p.get('G2_1', 1.5)); am_1 = float(p.get('a_margin_1', 1.75)); bm_1 = float(p.get('b_margin_1', 0.125))
    Ho_1 = float(p.get('H_overall_1', 100.0)); Div_1 = float(p.get('divisor_1', 400.0))

    # 2eloG Rating 2
    Kb_2 = float(p.get('K_base_2', 35.0)); Mo_2 = float(p.get('M_overall_2', 2.0))
    g2_2 = float(p.get('G2_2', 1.5)); am_2 = float(p.get('a_margin_2', 1.75)); bm_2 = float(p.get('b_margin_2', 0.125))
    Ho_2 = float(p.get('H_overall_2', 100.0)); Div_2 = Div_1 + float(p.get('divisor_offset_2', 200.0))

    # Pure 2eloOD Offense/Defense (NO G)
    Ks_od = float(p.get('K_od', 25.0)); Ms_od = float(p.get('M_od', 1.5))
    Hs_od = float(p.get('H_od', 100.0)); Ds_od = float(p.get('divisor_od', 400.0)); mu = float(p.get('mu', 1.35))

    r1 = {t: 1500.0 for t in teams}; r2 = {t: 1500.0 for t in teams}
    r_off = {t: 1500.0 for t in teams}; r_def = {t: 1500.0 for t in teams}

    e1h, e1a, e2h, e2a, oh, oa, dh, da = [], [], [], [], [], [], [], []

    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = float(row.home_score), float(row.away_score)
        neutral = int(row.neutral) if hasattr(row, 'neutral') and not pd.isna(row.neutral) else 0
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True

        e1h.append(r1[h]); e1a.append(r1[a])
        e2h.append(r2[h]); e2a.append(r2[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])

        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        N = abs(gh - ga)

        # 1. 2eloG Update - Rating 1 (WITH G)
        G_1 = 1.0 if N <= 1 else (g2_1 if N == 2 else am_1 + bm_1 * float(N))
        h_adv1 = Ho_1 if neutral == 0 else 0.0
        exp1 = max(-100.0, min(100.0, (r1[a] - (r1[h] + h_adv1)) / Div_1))
        we1_h = 1.0 / (1.0 + 10.0 ** exp1)
        k1 = Kb_1 * Mo_1 if is_comp else Kb_1
        r1[h] += k1 * G_1 * (sh - we1_h)
        r1[a] += k1 * G_1 * ((1.0 - sh) - (1.0 - we1_h))

        # 2. 2eloG Update - Rating 2 (WITH G)
        G_2 = 1.0 if N <= 1 else (g2_2 if N == 2 else am_2 + bm_2 * float(N))
        h_adv2 = Ho_2 if neutral == 0 else 0.0
        exp2 = max(-100.0, min(100.0, (r2[a] - (r2[h] + h_adv2)) / Div_2))
        we2_h = 1.0 / (1.0 + 10.0 ** exp2)
        k2 = Kb_2 * Mo_2 if is_comp else Kb_2
        r2[h] += k2 * G_2 * (sh - we2_h)
        r2[a] += k2 * G_2 * ((1.0 - sh) - (1.0 - we2_h))

        # 3. Pure 2eloOD Update - Offense / Defense (PURE, NO G)
        h_adv_od = Hs_od if neutral == 0 else 0.0
        exp_h = max(-100.0, min(100.0, (r_off[h] - r_def[a] + h_adv_od) / Ds_od))
        lh = mu * (10.0 ** exp_h)
        exp_a = max(-100.0, min(100.0, (r_off[a] - (r_def[h] + h_adv_od)) / Ds_od))
        la = mu * (10.0 ** exp_a)

        k_od = Ks_od * Ms_od if is_comp else Ks_od
        r_off[h] += k_od * (gh - lh)
        r_def[a] += k_od * (lh - gh)
        r_off[a] += k_od * (ga - la)
        r_def[h] += k_od * (la - ga)

    df['elo1_home'] = e1h; df['elo1_away'] = e1a
    df['elo2_home'] = e2h; df['elo2_away'] = e2a
    df['off_home'] = oh; df['off_away'] = oa
    df['def_home'] = dh; df['def_away'] = da

    df['elo1_diff'] = np.array(e1h) - np.array(e1a)
    df['elo2_diff'] = np.array(e2h) - np.array(e2a)
    df['diff_off'] = np.array(oh) - np.array(da)
    df['diff_def'] = np.array(oa) - np.array(dh)

    df['elo_home'] = e1h; df['elo_away'] = e1a
    df['elo_diff'] = df['elo1_diff']
    return df
