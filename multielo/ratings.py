import numpy as np
import pandas as pd

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
        - '2elo-pure' / '2elo-style' : Decoupled Offensive & Defensive style ratings.
        - '2elo-fast-slow' : Dual-timescale Fast+Slow outcome ratings.
        - '3elo-simple' / '3elo-hybrid' : Overall outcome + Decoupled style ratings.
        - '3elo-complete' : Complete multi-vector outcome + Decoupled style ratings.
        - '4elo' : 4-vector multi-scale (Fast/Slow x Offense/Defense) ratings.
    params : dict, optional
        Custom parameter dictionary. If None, default optimal parameters are used.
        
    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame augmented with computed pre-match and post-match ratings.
    """
    system_key = str(system).lower().replace('_', '-').strip()
    df_out = df.copy()
    
    teams = set(df_out['home_team']).union(set(df_out['away_team']))
    
    if system_key == 'fifa-sum':
        return _compute_fifa_sum(df_out, teams)
    elif system_key in ['eloratings', 'eloratings.net', 'elonet']:
        return _compute_eloratings(df_out, teams)
    elif system_key in ['1elo-simple', '1elo-s']:
        return _compute_1elo_simple(df_out, teams, params)
    elif system_key in ['1elo-complete', '1elo-c', '1eloc']:
        return _compute_1elo_complete(df_out, teams, params)
    elif system_key in ['2elo-pure', '2elo-style', '2elo-sd']:
        return _compute_2elo_style(df_out, teams, params)
    elif system_key in ['2elo-fast-slow', '2elo-fs']:
        return _compute_2elo_fast_slow(df_out, teams, params)
    elif system_key in ['3elo-simple', '3elo-hybrid', '3elo-h', '3eloh']:
        return _compute_3elo_hybrid(df_out, teams, params)
    elif system_key in ['3elo-complete', '3elo-c', '3eloc']:
        return _compute_3elo_complete(df_out, teams, params)
    else:
        raise ValueError(f"Unknown rating system: '{system}'. Supported: 'fifa-sum', 'eloratings', '1elo-simple', '1elo-complete', '2elo-pure', '2elo-fast-slow', '3elo-hybrid', '3elo-complete'.")


def _get_fifa_importance(tourn):
    t = str(tourn).lower()
    if 'world cup' in t: return 25.0 if 'qualification' in t or 'q' in t else 50.0
    if 'euro' in t or 'copa américa' in t: return 25.0 if 'qualification' in t else 35.0
    if 'nations' in t: return 15.0
    if 'friendly' in t: return 10.0
    return 20.0

def _get_elorating_k(tourn):
    t = str(tourn).lower()
    if 'world cup' in t: return 50.0 if 'qualification' in t or 'q' in t else 60.0
    if 'euro' in t or 'copa américa' in t: return 30.0 if 'qualification' in t else 40.0
    if 'friendly' in t: return 20.0
    return 30.0


def _compute_fifa_sum(df, teams):
    r_fifa = {t: 1000.0 for t in teams}
    fh, fa = [], []
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        gh, ga = int(row.home_score), int(row.away_score)
        fh.append(r_fifa[h]); fa.append(r_fifa[a])
        
        we_h = 1.0 / (1.0 + 10.0 ** ((r_fifa[a] - r_fifa[h]) / 600.0))
        sh = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        I = _get_fifa_importance(row.tournament if hasattr(row, 'tournament') else 'Friendly')
        
        r_fifa[h] += I * (sh - we_h)
        r_fifa[a] += I * ((1.0 - sh) - (1.0 - we_h))
        
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
        
        diff = abs(gh - ga)
        G = 1.0 if diff <= 1 else (1.5 if diff == 2 else (11.0 + diff) / 8.0)
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


def _compute_1elo_complete(df, teams, params):
    p = params or {
        'K_base': 32.0, 'M_overall': 2.2, 'H_overall': 100.0, 'divisor_overall': 371.0,
        'G2': 1.5, 'a_margin': 7.39, 'b_margin': 4.41
    }
    Kb = float(p.get('K_base', 32.0))
    Mo = float(p.get('M_overall', 2.2))
    Ho = float(p.get('H_overall', 100.0))
    Div = float(p.get('divisor_overall', 371.0))
    G2 = float(p.get('G2', 1.5))
    am = float(p.get('a_margin', 7.39))
    bm = float(p.get('b_margin', 4.41))
    
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
        
        diff = abs(gh - ga)
        G = 1.0 if diff <= 1 else (G2 if diff == 2 else (am + diff) / max(1e-5, bm))
        k = (Kb * Mo if is_comp else Kb) * G
        
        r_elo[h] += k * (sh - we_h)
        r_elo[a] += k * ((1.0 - sh) - (1.0 - we_h))
        
    df['elo_home'] = eh; df['elo_away'] = ea
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df


def _compute_2elo_style(df, teams, params):
    p = params or {'K_style': 23.95, 'M_style': 1.0, 'H_style': 93.19, 'divisor_style': 1274.07, 'mu': 1.35}
    Ks = float(p.get('K_style', 23.95))
    Ms = float(p.get('M_style', 1.0))
    Hs = float(p.get('H_style', 93.19))
    Ds = float(p.get('divisor_style', 1274.07))
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
    return df


def _compute_2elo_fast_slow(df, teams, params):
    p = params or {
        'K_fast': 111.97, 'M_fast': 1.72, 'H_fast': 48.35, 'divisor_fast': 201.04,
        'K_slow': 51.68, 'M_slow': 1.0, 'H_slow': 48.35, 'divisor_slow': 201.04
    }
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
        
        # Fast
        hf_adv = float(p.get('H_fast', 48.35)) if neutral == 0 else 0.0
        exp_f = max(-100.0, min(100.0, (r_fast[a] - (r_fast[h] + hf_adv)) / float(p.get('divisor_fast', 201.04))))
        we_f = 1.0 / (1.0 + 10.0 ** exp_f)
        kf = float(p.get('K_fast', 111.97)) * (float(p.get('M_fast', 1.72)) if is_comp else 1.0)
        r_fast[h] += kf * (s_h - we_f)
        r_fast[a] += kf * ((1.0 - s_h) - (1.0 - we_f))
        
        # Slow
        hs_adv = float(p.get('H_slow', 48.35)) if neutral == 0 else 0.0
        exp_s = max(-100.0, min(100.0, (r_slow[a] - (r_slow[h] + hs_adv)) / float(p.get('divisor_slow', 201.04))))
        we_s = 1.0 / (1.0 + 10.0 ** exp_s)
        ks = float(p.get('K_slow', 51.68)) * (float(p.get('M_slow', 1.0)) if is_comp else 1.0)
        r_slow[h] += ks * (s_h - we_s)
        r_slow[a] += ks * ((1.0 - s_h) - (1.0 - we_s))
        
    df['fast_home'] = fh; df['fast_away'] = fa
    df['slow_home'] = sh; df['slow_away'] = sa
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
        
        # 1. Overall
        h_adv_o = Ho if neutral == 0 else 0.0
        exp_o = max(-100.0, min(100.0, (r_overall[a] - (r_overall[h] + h_adv_o)) / Div_o))
        we_h = 1.0 / (1.0 + 10.0 ** exp_o)
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        ko = Kb * Mo if is_comp else Kb
        r_overall[h] += ko * (s_h - we_h)
        r_overall[a] += ko * ((1.0 - s_h) - (1.0 - we_h))
        
        # 2. Style
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
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df


def _compute_3elo_complete(df, teams, params):
    p = params or {
        'K_base': 32.05371, 'M_overall': 2.501047, 'H_overall': 218.209652, 'divisor_overall': 1267.582933,
        'G2': 1.097433, 'a_margin': 4.32698, 'b_margin': 3.603202, 'M_style': 1.072913,
        'H_style': 60.945526, 'divisor_style': 974.553479, 'K_scale': 0.552142, 'mu': 1.35
    }
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
        is_comp = (row.tournament != 'Friendly') if hasattr(row, 'tournament') and not pd.isna(row.tournament) else True
        
        eh.append(r_overall[h]); ea.append(r_overall[a])
        oh.append(r_off[h]); oa.append(r_off[a])
        dh.append(r_def[h]); da.append(r_def[a])
        
        # 1. Overall with Margin Multiplier G(N)
        h_adv_o = Ho if neutral == 0 else 0.0
        exp_o = max(-100.0, min(100.0, (r_overall[a] - (r_overall[h] + h_adv_o)) / Div_o))
        we_h = 1.0 / (1.0 + 10.0 ** exp_o)
        s_h = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        
        diff = abs(gh - ga)
        G = 1.0 if diff <= 1 else (G2 if diff == 2 else (am + diff) / max(1e-5, bm))
        ko = (Kb * Mo if is_comp else Kb) * G
        r_overall[h] += ko * (s_h - we_h)
        r_overall[a] += ko * ((1.0 - s_h) - (1.0 - we_h))
        
        # 2. Style (Offense & Defense)
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
    df['elo_diff'] = np.array(eh) - np.array(ea)
    return df
