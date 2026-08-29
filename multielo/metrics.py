import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
from bisect import bisect_right
from .models import get_model_specs, build_design_matrix, dixon_coles_tau_vec, train_model, TrainedModel
from .data_builder import get_balanced_learning_dataset, get_fifa_base_weight, categorize_tournament_class

MAX_G = 10
g_arr = np.arange(MAX_G + 1)

def score_distance(h1, a1, h2, a2):
    w_cat = 1.5; w_gd = 0.5; w_vol = 0.3
    cat1 = 'H' if h1 > a1 else ('A' if h1 < a1 else 'D')
    cat2 = 'H' if h2 > a2 else ('A' if h2 < a2 else 'D')
    d_cat = 0.0 if cat1 == cat2 else (1.0 if (cat1 == 'D' or cat2 == 'D') else 2.0)
    d_gd = abs((h1 - a1) - (h2 - a2))
    d_vol = abs((h1 + a1) - (h2 + a2))
    return w_cat * d_cat + w_gd * d_gd + w_vol * d_vol

_DIST_MATRIX = np.zeros((MAX_G + 1, MAX_G + 1, MAX_G + 1, MAX_G + 1))
for h1 in range(MAX_G + 1):
    for a1 in range(MAX_G + 1):
        for h2 in range(MAX_G + 1):
            for a2 in range(MAX_G + 1):
                _DIST_MATRIX[h1, a1, h2, a2] = score_distance(h1, a1, h2, a2)

def compute_rps(p_win_a, p_draw, p_win_b, outcome):
    """
    Compute Ranked Probability Score (RPS) for trichotomous outcome forecasts.
    """
    if outcome in ['H', 1, 'win', 'home']:
        yh, yd, ya = 1.0, 0.0, 0.0
    elif outcome in ['D', 0, 'draw']:
        yh, yd, ya = 0.0, 1.0, 0.0
    else:
        yh, yd, ya = 0.0, 0.0, 1.0
        
    term1 = (p_win_a - yh) ** 2
    term2 = ((p_win_a + p_draw) - (yh + yd)) ** 2
    return 0.5 * (term1 + term2)

def compute_esd(score_matrix, actual_g_a, actual_g_b, w_cat=1.5, w_gd=0.5, w_vol=0.3):
    """
    Compute Expected Square Difference (ESD) scoreline dissimilarity metric.
    """
    esd = 0.0
    max_ga, max_gb = score_matrix.shape
    
    actual_cat = 1 if actual_g_a > actual_g_b else (0 if actual_g_a == actual_g_b else -1)
    actual_gd = actual_g_a - actual_g_b
    actual_vol = actual_g_a + actual_g_b
    
    for x in range(max_ga):
        for y in range(max_gb):
            p_xy = score_matrix[x, y]
            if p_xy <= 0:
                continue
                
            pred_cat = 1 if x > y else (0 if x == y else -1)
            if pred_cat == actual_cat:
                d_cat = 0.0
            elif (pred_cat == 0 and actual_cat != 0) or (pred_cat != 0 and actual_cat == 0):
                d_cat = 1.0
            else:
                d_cat = 2.0
                
            d_gd = abs((x - y) - actual_gd)
            d_vol = abs((x + y) - actual_vol)
            
            d_total = w_cat * d_cat + w_gd * d_gd + w_vol * d_vol
            esd += p_xy * d_total
            
    return esd

def compute_aic(log_likelihood, k_params):
    """
    Compute Akaike Information Criterion (AIC).
    """
    return 2.0 * k_params - 2.0 * log_likelihood

def get_eval_match_weights(df, eval_weight_mode='fifa_topology'):
    """
    Returns base match evaluation weights based on tournament tier mode:
      - 'fifa_topology' / 'weighted': FIFA Men's World Ranking weighting factors (World Cup Finals=60, Continental Finals=50, Qualifiers/Nations League=25, Other/Friendlies=10).
      - 'finals': Weight 1.0 for World Cup & Continental Championship final tournaments, 0.0 for others.
      - 'standard': Equal weight (1.0) for all matches.
    """
    mode = str(eval_weight_mode).lower().strip()
    if 'tournament' in df.columns:
        tourn_series = df['tournament']
    elif 'tourn_name' in df.columns:
        tourn_series = df['tourn_name']
    else:
        return np.ones(len(df), dtype=float)
        
    if mode in ['finals', 'finals_only', 'finals-only']:
        def is_final(t):
            t_l = str(t).lower()
            if 'qualification' in t_l or 'qualifying' in t_l:
                return 0.0
            if t_l in ['fifa world cup', 'world cup']:
                return 1.0
            if any(k in t_l for k in ['uefa euro', 'copa américa', 'copa america', 'african cup of nations', 'afc asian cup', 'gold cup', 'confederations cup', 'nations league finals']):
                return 1.0
            return 0.0
        return tourn_series.apply(is_final).values
        
    elif mode in ['fifa_topology', 'topology', 'fifa', 'weighted', 'default']:
        return tourn_series.apply(get_fifa_base_weight).values
    else:
        return np.ones(len(df), dtype=float)

def get_home_away_cols_for_diff(c):
    if c == 'elo_diff':
        return 'elo_home', 'elo_away'
    elif c == 'elo_diff_fast':
        return 'elo_home_fast', 'elo_away_fast'
    elif c == 'elo_diff_slow':
        return 'elo_home_slow', 'elo_away_slow'
    elif c == 'diff_off':
        return 'off_home', 'off_away'
    elif c == 'diff_def':
        return 'def_home', 'def_away'
    elif c.endswith('_diff'):
        base = c[:-5]
        return f"{base}_home", f"{base}_away"
    elif c.startswith('diff_'):
        base = c[5:]
        return f"{base}_home", f"{base}_away"
    return None, None

def ensure_6mo_lag_ratings(df, rating_col='elo_diff', rating_col_6mo=None):
    """
    Ensures that 6-month lagged rating differences are present in the DataFrame for all rating columns.
    """
    if isinstance(rating_col, (list, tuple)):
        r_cols = list(rating_col)
    else:
        r_cols = [rating_col]
        
    df_out = df.copy()
    col_6mo_list = []
    
    missing = [c for c in r_cols if f"{c}_6mo" not in df_out.columns]
    if not missing:
        col_6mo_list = [f"{c}_6mo" for c in r_cols]
        res_6mo = col_6mo_list if isinstance(rating_col, (list, tuple)) else col_6mo_list[0]
        return df_out, res_6mo

    six_months_sec = 182.5 * 86400.0
    dates = pd.to_datetime(df_out['date']).astype('int64') // 10**9
    teams = set(df_out['home_team']).union(set(df_out['away_team']))
    init_time = pd.Timestamp('1870-01-01').timestamp()
    
    has_off = ('off_home' in df_out.columns) and ('def_away' in df_out.columns)
    has_fs = ('elo_home_fast' in df_out.columns) and ('elo_home_slow' in df_out.columns)
    has_elo = ('elo_home' in df_out.columns)
    
    hist = {}
    if has_off:
        hist['off'] = {t: [(init_time, 1500.0)] for t in teams}
        hist['def'] = {t: [(init_time, 1500.0)] for t in teams}
    if has_fs:
        hist['fast'] = {t: [(init_time, 1500.0)] for t in teams}
        hist['slow'] = {t: [(init_time, 1500.0)] for t in teams}
    if has_elo:
        hist['elo'] = {t: [(init_time, 1500.0)] for t in teams}

    for idx, row in df_out.iterrows():
        t_sec = dates.iloc[idx]
        ht, at = row['home_team'], row['away_team']
        
        if has_off:
            hist['off'][ht].append((t_sec, row['off_home']))
            hist['off'][at].append((t_sec, row['off_away']))
            hist['def'][ht].append((t_sec, row['def_home']))
            hist['def'][at].append((t_sec, row['def_away']))
        if has_fs:
            hist['fast'][ht].append((t_sec, row['elo_home_fast']))
            hist['fast'][at].append((t_sec, row['elo_away_fast']))
            hist['slow'][ht].append((t_sec, row['elo_home_slow']))
            hist['slow'][at].append((t_sec, row['elo_away_slow']))
        if has_elo:
            hist['elo'][ht].append((t_sec, row['elo_home']))
            hist['elo'][at].append((t_sec, row['elo_away']))

    def get_lag_val(hist_list, current_sec):
        target_sec = current_sec - six_months_sec
        times = [x[0] for x in hist_list]
        pos = bisect_right(times, target_sec) - 1
        if pos < 0:
            return hist_list[0][1]
        return hist_list[pos][1]

    for c in r_cols:
        target_6mo = f"{c}_6mo"
        col_6mo_list.append(target_6mo)
        if target_6mo in df_out.columns:
            continue
            
        home_c, away_c = get_home_away_cols_for_diff(c)
        diff_6mo_vals = []
        
        for idx, row in df_out.iterrows():
            t_sec = dates.iloc[idx]
            ht, at = row['home_team'], row['away_team']
            
            if 'off' in c and has_off:
                h_val = get_lag_val(hist['off'][ht], t_sec)
                a_val = get_lag_val(hist['off'][at], t_sec)
            elif 'def' in c and has_off:
                h_val = get_lag_val(hist['def'][ht], t_sec)
                a_val = get_lag_val(hist['def'][at], t_sec)
            elif 'fast' in c and has_fs:
                h_val = get_lag_val(hist['fast'][ht], t_sec)
                a_val = get_lag_val(hist['fast'][at], t_sec)
            elif 'slow' in c and has_fs:
                h_val = get_lag_val(hist['slow'][ht], t_sec)
                a_val = get_lag_val(hist['slow'][at], t_sec)
            elif has_elo:
                h_val = get_lag_val(hist['elo'][ht], t_sec)
                a_val = get_lag_val(hist['elo'][at], t_sec)
            else:
                h_val = float(row[home_c]) if home_c and home_c in row else 1500.0
                a_val = float(row[away_c]) if away_c and away_c in row else 1500.0
                
            diff_6mo_vals.append(h_val - a_val)
            
        df_out[target_6mo] = diff_6mo_vals

    res_6mo = col_6mo_list if isinstance(rating_col, (list, tuple)) else col_6mo_list[0]
    return df_out, res_6mo

def evaluate_5cv(df, model_code='M32', rating_col='elo_diff', rating_col_6mo=None, tourn_col='tourn_weight', k_rating=14, eval_weight_mode='fifa_topology', use_balanced_dataset=True, weighted_training=True, year_min=1950, year_max=None):
    """
    Perform 5-fold cross-validation loss evaluation post-1950 (inclusive).
    
    Default Configuration:
      - use_balanced_dataset=True: Evaluates on the Stratified 4x-Expanded Balanced Learning Dataset with pre-assigned 5-CV folds.
      - Fast metrics (RPS & ESD): Weighted by FIFA topology base weights * class_factor (0.25 for 4x sampled qualifiers & friendlies).
      - Slow metrics (RPS & ESD 6-month lag): Keeps FIFA weights but EXCLUDES friendlies (w_slow = 0 for friendlies).
    """
    df_eval = df.copy().reset_index(drop=True)
    if 'date' in df_eval.columns:
        df_eval['date'] = pd.to_datetime(df_eval['date'])
    if 'year' not in df_eval.columns:
        df_eval['year'] = df_eval['date'].dt.year
    if 'result' not in df_eval.columns:
        df_eval['result'] = ['H' if gh > ga else ('D' if gh == ga else 'A') for gh, ga in zip(df_eval['home_score'], df_eval['away_score'])]
        
    df_eval = df_eval[df_eval['year'] >= year_min]
    if year_max is not None:
        df_eval = df_eval[df_eval['year'] <= year_max]
    df_eval = df_eval.sort_values('date').reset_index(drop=True)
    
    # Load balanced dataset if requested
    if use_balanced_dataset:
        df_bal = get_balanced_learning_dataset()
        df_eval['date_str'] = pd.to_datetime(df_eval['date']).dt.strftime('%Y-%m-%d')
        df_bal['date_str'] = pd.to_datetime(df_bal['date']).dt.strftime('%Y-%m-%d')
        
        # Merge pre-assigned folds and calculated w_fast, w_slow using string dates
        df_eval = df_eval.merge(
            df_bal[['date_str', 'home_team', 'away_team', 'fold', 'tourn_class_balanced', 'w_fast', 'w_slow']],
            on=['date_str', 'home_team', 'away_team'],
            how='inner'
        ).sort_values('date').reset_index(drop=True)
    else:
        n_samples = len(df_eval)
        df_eval['fold'] = np.arange(n_samples) % 5
        tourn_c = 'tournament' if 'tournament' in df_eval.columns else 'tourn_name'
        df_eval['tourn_class_balanced'] = df_eval[tourn_c].apply(categorize_tournament_class)
        base_w = get_eval_match_weights(df_eval, eval_weight_mode=eval_weight_mode)
        df_eval['w_fast'] = base_w
        df_eval['w_slow'] = np.where(df_eval['tourn_class_balanced'] == 'friendlies_lower', 0.0, base_w)
        
    if str(eval_weight_mode).lower().strip() in ['finals', 'finals_only', 'finals-only']:
        df_eval = df_eval[df_eval['tourn_class_balanced'].isin(['world_cup', 'continental_confed'])].reset_index(drop=True)
        if len(df_eval) == 0:
            raise ValueError("No final tournament matches found in evaluation dataset.")
            
    df_eval, col_6mo = ensure_6mo_lag_ratings(df_eval, rating_col=rating_col, rating_col_6mo=rating_col_6mo)
    
    fold_metrics = []
    for fold in range(5):
        df_tr = df_eval[df_eval['fold'] != fold].reset_index(drop=True)
        df_te = df_eval[df_eval['fold'] == fold].reset_index(drop=True)
        
        mod = train_model(df_tr, model_code=model_code, rating_col=rating_col, tourn_col=tourn_col, k_rating=k_rating, weighted_training=weighted_training)
        
        rps_fast_l, rps_slow_l = [], []
        esd_fast_l, esd_slow_l = [], []
        
        w_fast_l = df_te['w_fast'].values if 'w_fast' in df_te.columns else np.ones(len(df_te))
        w_slow_l = df_te['w_slow'].values if 'w_slow' in df_te.columns else np.ones(len(df_te))
        
        for row in df_te.itertuples():
            if isinstance(rating_col, (list, tuple)):
                diff_fast = [float(getattr(row, col) if hasattr(row, col) else 0.0) for col in rating_col]
            else:
                diff_fast = float(getattr(row, rating_col) if hasattr(row, rating_col) else getattr(row, 'diff_val', 0.0))
                
            if isinstance(col_6mo, (list, tuple)):
                diff_slow = [float(getattr(row, col) if hasattr(row, col) else 0.0) for col in col_6mo]
            elif col_6mo and hasattr(row, col_6mo):
                diff_slow = float(getattr(row, col_6mo))
            else:
                diff_slow = diff_fast
            
            is_n = getattr(row, 'neutral', False)
            yr = getattr(row, 'year', 2025)
            tw = getattr(row, 'tourn_weight', 30.0)
            
            p_fast = mod.predict(diff_fast, is_neutral=is_n, year=yr, tourn_weight=tw, max_goals=MAX_G)
            rps_fast_l.append(compute_rps(p_fast['p_home_win'], p_fast['p_draw'], p_fast['p_away_win'], row.result))
            
            gh = min(int(row.home_score), MAX_G)
            ga = min(int(row.away_score), MAX_G)
            esd_fast_l.append(np.sum(p_fast['scoreline_matrix'] * _DIST_MATRIX[:, :, gh, ga]))
            
            p_slow = mod.predict(diff_slow, is_neutral=is_n, year=yr, tourn_weight=tw, max_goals=MAX_G)
            rps_slow_l.append(compute_rps(p_slow['p_home_win'], p_slow['p_draw'], p_slow['p_away_win'], row.result))
            esd_slow_l.append(np.sum(p_slow['scoreline_matrix'] * _DIST_MATRIX[:, :, gh, ga]))
            
        w_fast_sum = np.sum(w_fast_l)
        if w_fast_sum <= 0:
            w_fast_sum = 1.0
            w_fast_l = np.ones_like(w_fast_l)

        w_slow_sum = np.sum(w_slow_l)
        if w_slow_sum <= 0:
            w_slow_sum = 1.0
            w_slow_l = np.ones_like(w_slow_l)
            
        rf = np.sum(np.array(rps_fast_l) * w_fast_l) / w_fast_sum
        ef = np.sum(np.array(esd_fast_l) * w_fast_l) / w_fast_sum
        
        rs = np.sum(np.array(rps_slow_l) * w_slow_l) / w_slow_sum
        es = np.sum(np.array(esd_slow_l) * w_slow_l) / w_slow_sum
        
        fesd = rf + 0.06 * ef
        jall = rf + 0.06 * ef + rs
        
        fold_metrics.append([rf, rs, ef, es, fesd, jall])
        
    means = np.mean(fold_metrics, axis=0)
    return {
        'CV_RPS_fast': float(means[0]),
        'CV_RPS_slow': float(means[1]),
        'CV_ESD_fast': float(means[2]),
        'CV_ESD_slow': float(means[3]),
        'CV_Fast+ESD': float(means[4]),
        'CV_Joint_ALL': float(means[5])
    }

def evaluate_aics(df, model_code='M32', rating_col='fifa_diff', rating_col_6mo=None, tourn_col='tourn_weight', k_rating=14, trained_model=None, eval_weight_mode='fifa_topology', use_balanced_dataset=True, year_min=1950, year_max=2018):
    """
    Compute full-dataset log-likelihoods and calculate Metric-Specific Dixon-Coles AICs.
    """
    df_eval = df.copy().reset_index(drop=True)
    if 'date' in df_eval.columns:
        df_eval['date'] = pd.to_datetime(df_eval['date'])
    if 'year' not in df_eval.columns:
        df_eval['year'] = df_eval['date'].dt.year
        
    df_eval = df_eval[(df_eval['year'] >= year_min) & (df_eval['year'] <= year_max)].sort_values('date').reset_index(drop=True)
    
    if use_balanced_dataset:
        df_bal = get_balanced_learning_dataset()
        df_bal['date'] = pd.to_datetime(df_bal['date'])
        df_eval = df_eval.merge(
            df_bal[['date', 'home_team', 'away_team', 'tourn_class_balanced', 'w_fast', 'w_slow']],
            on=['date', 'home_team', 'away_team'],
            how='inner'
        ).sort_values('date').reset_index(drop=True)
    else:
        tourn_c = 'tournament' if 'tournament' in df_eval.columns else 'tourn_name'
        df_eval['tourn_class_balanced'] = df_eval[tourn_c].apply(categorize_tournament_class)
        base_w = get_eval_match_weights(df_eval, eval_weight_mode=eval_weight_mode)
        df_eval['w_fast'] = base_w
        df_eval['w_slow'] = np.where(df_eval['tourn_class_balanced'] == 'friendlies_lower', 0.0, base_w)
        
    df_eval, col_6mo = ensure_6mo_lag_ratings(df_eval, rating_col=rating_col, rating_col_6mo=rating_col_6mo)
        
    mod = trained_model or train_model(df_eval, model_code=model_code, rating_col=rating_col, tourn_col=tourn_col, k_rating=k_rating)
    k_tot = mod.k_tot
    
    X_fast = build_design_matrix(df_eval, model_code=model_code, rating_col=rating_col, tourn_col=tourn_col)
    y_h = df_eval['home_score'].values
    y_a = df_eval['away_score'].values
    
    w_fast_v = df_eval['w_fast'].values
    w_slow_v = df_eval['w_slow'].values
    
    mu_h_f = np.exp(np.dot(X_fast.values, mod.params_home))
    mu_a_f = np.exp(np.dot(X_fast.values, mod.params_away))
    
    p0_f = poisson.pmf(y_h, mu_h_f) * poisson.pmf(y_a, mu_a_f)
    tau_f = dixon_coles_tau_vec(y_h, y_a, mu_h_f, mu_a_f, mod.rho) if mod.is_bivariate else np.ones(len(y_h))
    log_p0_f = np.log(np.maximum(1e-12, p0_f * tau_f))
    ll_fast = np.sum(log_p0_f * w_fast_v) / (np.sum(w_fast_v) / len(w_fast_v))
    
    df_eval_slow = df_eval.copy()
    if isinstance(rating_col, (list, tuple)):
        for r_c, c_6 in zip(rating_col, col_6mo):
            df_eval_slow[r_c] = df_eval_slow[c_6]
    else:
        df_eval_slow[rating_col] = df_eval_slow[col_6mo]
    X_slow = build_design_matrix(df_eval_slow, model_code=model_code, rating_col=rating_col, tourn_col=tourn_col)
    
    mu_h_s = np.exp(np.dot(X_slow.values, mod.params_home))
    mu_a_s = np.exp(np.dot(X_slow.values, mod.params_away))
    
    p0_s = poisson.pmf(y_h, mu_h_s) * poisson.pmf(y_a, mu_a_s)
    tau_s = dixon_coles_tau_vec(y_h, y_a, mu_h_s, mu_a_s, mod.rho) if mod.is_bivariate else np.ones(len(y_h))
    log_p0_s = np.log(np.maximum(1e-12, p0_s * tau_s))
    
    w_slow_denom = np.sum(w_slow_v)
    if w_slow_denom > 0:
        ll_slow = np.sum(log_p0_s * w_slow_v) / (w_slow_denom / len(w_slow_v))
    else:
        ll_slow = ll_fast
    
    ll_all = 0.5 * ll_fast + 0.5 * ll_slow
    
    aic_fast = compute_aic(ll_fast, k_tot)
    aic_slow = compute_aic(ll_slow, k_tot)
    aic_all = compute_aic(ll_all, k_tot)
    
    return {
        'AIC_RPS_fast': float(aic_fast),
        'AIC_RPS_slow': float(aic_slow),
        'AIC_ESD_fast': float(aic_fast),
        'AIC_Fast+ESD': float(aic_fast),
        'AIC_Joint_ALL': float(aic_all),
        'AIC_fast': float(aic_fast),
        'AIC_slow': float(aic_slow),
        'AIC_esd': float(aic_fast),
        'AIC_fastesd': float(aic_fast),
        'AIC_all': float(aic_all),
        'k_tot': k_tot
    }

SURROGATE_CHANCE_BASELINES_10K = {
    'RPS_fast':  {'mean': 0.294277, 'std': 0.003113},
    'RPS_slow':  {'mean': 0.294277, 'std': 0.003328},
    'ESD_fast':  {'mean': 5.728059, 'std': 0.005018},
    'Fast+ESD':  {'mean': 0.637961, 'std': 0.003132},
    'Joint_ALL': {'mean': 0.932238, 'std': 0.006449}
}

def compute_metric_zscores(metrics_dict):
    """
    Compute Z-score performance standardizations relative to the 10,000-run Monte Carlo chance baseline.
    
    Since all metrics are loss values (where lower score = better performance),
    Z-score is defined as:
        Z = (mean_chance_baseline - score) / std_deviation
    
    A positive Z-score indicates outperformance vs chance in standard deviations (e.g., +36.0 sigma).
    """
    z_scores = {}
    mapping = {
        'RPS_fast': ['CV_RPS_fast', 'RPS_fast', 'Val_CV_RPS_fast', 'Holdout_Test_RPS_fast'],
        'RPS_slow': ['CV_RPS_slow', 'RPS_slow', 'Val_CV_RPS_slow', 'Holdout_Test_RPS_slow'],
        'ESD_fast': ['CV_ESD_fast', 'ESD_fast'],
        'Fast+ESD': ['CV_Fast+ESD', 'Fast+ESD'],
        'Joint_ALL': ['CV_Joint_ALL', 'Joint_ALL', 'Val_CV_Joint_ALL', 'Holdout_Test_Joint_ALL']
    }

    for metric_name, keys in mapping.items():
        base = SURROGATE_CHANCE_BASELINES_10K[metric_name]
        mu0, std0 = base['mean'], base['std']
        
        for k in keys:
            if k in metrics_dict:
                val = float(metrics_dict[k])
                z_val = (mu0 - val) / std0
                z_key = f"Z_{k}" if not k.startswith("Z_") else k
                z_scores[z_key] = float(z_val)

    return z_scores

def evaluate_model(df, model_code='M32', rating_col='elo_diff', rating_col_6mo=None, tourn_col='tourn_weight', k_rating=14, eval_weight_mode='fifa_topology', use_balanced_dataset=True, weighted_training=True, year_min=1950, year_max=None):
    """
    Consolidated evaluation function computing 5-CV metrics, AICs, and Chance Z-Scores.
    Supports Scheme A (weighted_training=True, M01-M32) and Scheme B (weighted_training=False, M33-M64).
    """
    raw_code = str(model_code).upper().strip()
    m_num = int(raw_code.replace("M", "")) if raw_code.replace("M", "").isdigit() else 32
    
    if m_num > 32:
        base_code = f"M{(m_num - 32):02d}"
        weighted_training = False
    else:
        base_code = f"M{m_num:02d}"

    mod = train_model(df, model_code=base_code, rating_col=rating_col, tourn_col=tourn_col, k_rating=k_rating, weighted_training=weighted_training)
    cv_dict = evaluate_5cv(df, model_code=base_code, rating_col=rating_col, rating_col_6mo=rating_col_6mo, tourn_col=tourn_col, k_rating=k_rating, eval_weight_mode=eval_weight_mode, use_balanced_dataset=use_balanced_dataset, weighted_training=weighted_training, year_min=year_min, year_max=year_max)
    aic_dict = evaluate_aics(df, model_code=base_code, rating_col=rating_col, rating_col_6mo=rating_col_6mo, tourn_col=tourn_col, k_rating=k_rating, trained_model=mod, eval_weight_mode=eval_weight_mode, use_balanced_dataset=use_balanced_dataset)
    z_dict = compute_metric_zscores(cv_dict)
    
    res = {
        'model_code': raw_code,
        'base_model_code': base_code,
        'specs': str(mod.specs),
        'k_params': mod.k_tot,
        'scheme': 'Scheme B (Category/Unweighted)' if m_num > 32 else 'Scheme A (Weighted FIFA)',
        **cv_dict,
        **z_dict,
        **aic_dict
    }
    return res
