"""
Taxonomy, Training, Parameter Management, and Prediction Specifications for the 32 Poisson GLM Models (M01-M32).
"""

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import poisson

GLM_TAXONOMY = {
    # Independent Poisson Models (P)
    'M01': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Linear', 'decay': False, 'competition': False},
    'M02': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Linear', 'decay': False, 'competition': True},
    'M03': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Linear', 'decay': True, 'competition': False},
    'M04': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Linear', 'decay': True, 'competition': True},
    'M05': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': False, 'competition': False},
    'M06': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': False, 'competition': True},
    'M07': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': True, 'competition': False},
    'M08': {'dist': 'Poisson', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': True, 'competition': True},
    'M09': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Linear', 'decay': False, 'competition': False},
    'M10': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Linear', 'decay': False, 'competition': True},
    'M11': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Linear', 'decay': True, 'competition': False},
    'M12': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Linear', 'decay': True, 'competition': True},
    'M13': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': False, 'competition': False},
    'M14': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': False, 'competition': True},
    'M15': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': True, 'competition': False},
    'M16': {'dist': 'Poisson', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': True, 'competition': True},
    
    # Bivariate Dixon-Coles Models (B)
    'M17': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Linear', 'decay': False, 'competition': False},
    'M18': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Linear', 'decay': False, 'competition': True},
    'M19': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Linear', 'decay': True, 'competition': False},
    'M20': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Linear', 'decay': True, 'competition': True},
    'M21': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': False, 'competition': False},
    'M22': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': False, 'competition': True},
    'M23': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': True, 'competition': False},
    'M24': {'dist': 'DixonColes', 'coupling': 'Shared', 'response': 'Quadratic', 'decay': True, 'competition': True},
    'M25': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Linear', 'decay': False, 'competition': False},
    'M26': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Linear', 'decay': False, 'competition': True},
    'M27': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Linear', 'decay': True, 'competition': False},
    'M28': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Linear', 'decay': True, 'competition': True},
    'M29': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': False, 'competition': False},
    'M30': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': False, 'competition': True},
    'M31': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': True, 'competition': False},
    'M32': {'dist': 'DixonColes', 'coupling': 'Independent', 'response': 'Quadratic', 'decay': True, 'competition': True},
}

DEFAULT_PARAMS_DIR = os.path.expanduser("~/.multielo/params")

def get_standard_params_path(model_code='M32', rating_system='fifa-sum'):
    """Return standard default file path for saving/loading model parameters."""
    code = str(model_code).upper().strip()
    sys_name = str(rating_system).lower().strip()
    
    if os.path.exists("params"):
        return os.path.join("params", f"params_{sys_name}_{code}.json")
        
    os.makedirs(DEFAULT_PARAMS_DIR, exist_ok=True)
    return os.path.join(DEFAULT_PARAMS_DIR, f"params_{sys_name}_{code}.json")

def get_model_specs(model_code='M32'):
    """
    Retrieve architectural specifications for a given GLM model code (M01-M32).
    """
    code = str(model_code).upper().strip()
    if code not in GLM_TAXONOMY:
        raise ValueError(f"Unknown model code '{model_code}'. Must be between M01 and M32.")
    return GLM_TAXONOMY[code]

def dixon_coles_tau_vec(y_h, y_a, lh, la, rho):
    """Vectorized Dixon-Coles tau adjustment factor."""
    tau = np.ones(len(y_h))
    m00 = (y_h == 0) & (y_a == 0)
    m01 = (y_h == 0) & (y_a == 1)
    m10 = (y_h == 1) & (y_a == 0)
    m11 = (y_h == 1) & (y_a == 1)
    
    tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau[m01] = 1.0 + lh[m01] * rho
    tau[m10] = 1.0 + la[m10] * rho
    tau[m11] = 1.0 - rho
    return tau

def build_design_matrix(df, model_code='M32', rating_col='fifa_diff', tourn_col='tourn_weight'):
    """
    Build design matrix X for any of the 32 GLM model specifications.
    Supports single scalar rating_col or multidimensional lists/tuples of rating_col names.
    """
    specs = get_model_specs(model_code)
    
    if isinstance(rating_col, (list, tuple)):
        r_cols = list(rating_col)
    else:
        r_cols = [rating_col]
        
    neutral = df['neutral'].values if 'neutral' in df.columns else np.zeros(len(df))
    
    if 'year' in df.columns:
        year_val = (df['year'].values - 1950.0) / 75.0
    elif 'date' in df.columns:
        years = pd.to_datetime(df['date']).dt.year.values
        year_val = (years - 1950.0) / 75.0
    else:
        year_val = np.zeros(len(df))
        
    X_dict = {'const': np.ones(len(df))}
    
    for idx, col in enumerate(r_cols):
        if col in df.columns:
            diff_vals = np.clip(df[col].values / 400.0, -10.0, 10.0)
        elif 'diff_val' in df.columns and idx == 0:
            diff_vals = np.clip(df['diff_val'].values / 400.0, -10.0, 10.0)
        else:
            diff_vals = np.zeros(len(df))
            
        c_name = f"diff_{idx+1}" if len(r_cols) > 1 else "diff"
        X_dict[c_name] = diff_vals
        
        if specs['response'] == 'Quadratic':
            c_sq = f"diff2_{idx+1}" if len(r_cols) > 1 else "diff2"
            X_dict[c_sq] = np.clip(diff_vals ** 2, 0.0, 100.0)
            
    if specs['coupling'] == 'Independent':
        X_dict['neutral'] = neutral
        
    if specs['decay']:
        X_dict['year'] = year_val
        
    if specs['competition']:
        if tourn_col in df.columns:
            X_dict['tourn_w'] = df[tourn_col].values / 50.0
        elif 'tournament' in df.columns:
            t_map = {'Friendly': 20, 'FIFA World Cup': 60, 'Copa América': 50, 'UEFA Euro': 50}
            X_dict['tourn_w'] = np.array([t_map.get(t, 30) for t in df['tournament']]) / 50.0
        else:
            X_dict['tourn_w'] = np.ones(len(df)) * (30.0 / 50.0)
            
    return pd.DataFrame(X_dict)

SYSTEM_K_RATING = {
    'fifa-sum': 0,
    'eloratings': 0,
    '1eloF': 4,
    '1elo-simple': 4,
    '1eloG': 7,
    '1elo-g': 7,
    '1eloX': 7,
    '1elo-x': 7,
    '2eloG': 14,
    '2elo-g': 14,
    '2eloOD': 5,
    '2elo-od': 5,
    '2eloODG': 7,
    '2elo-odg': 7,
    '2eloFS': 8,
    '2elo-fast-slow': 8,
    '3eloH': 8,
    '3elo-hybrid': 8,
    '3eloC': 14,
    '3elo-complete': 14,
    '3eloODG': 11,
    '3elo-odg': 11,
    '4elo': 12,
    '4eloM': 12,
    '4elo-multiscale': 12,
    '4eloG': 15,
    '4elo-g': 15,
    '4elood+2g': 14,
    '4elo_od_2g': 14,
    '4elood2g': 14,
    '3elood+1g': 10,
    '3elo_od_1g': 10,
    '3elood1g': 10
}

class TrainedModel:
    """
    Encapsulates a trained GLM Poisson / Dixon-Coles model from M01 to M32.
    """
    def __init__(self, model_code, params_home, params_away, rho=0.0, rating_col='fifa_diff', rating_system='fifa-sum', k_rating=None):
        self.model_code = str(model_code).upper().strip()
        self.specs = get_model_specs(self.model_code)
        self.params_home = np.array(params_home, dtype=float)
        self.params_away = np.array(params_away, dtype=float)
        self.rho = float(rho)
        self.rating_col = rating_col
        self.rating_system = rating_system
        self.is_bivariate = (self.specs['dist'] == 'DixonColes')
        self.k_glm = len(self.params_home) + len(self.params_away) + (1 if self.is_bivariate else 0)
        
        if k_rating is not None:
            self.k_rating = int(k_rating)
        else:
            # Infer k_rating from system prefix
            matched_k = 0
            for sys_key, k_val in SYSTEM_K_RATING.items():
                if sys_key.lower() in self.rating_system.lower():
                    matched_k = k_val
                    break
            self.k_rating = matched_k
            
        self.k_tot = self.k_glm + self.k_rating

    def predict(self, rating_diff, is_neutral=False, year=2025, tourn_weight=30.0, max_goals=14):
        """
        Predict expected goals, scoreline probability matrix, and match outcome probabilities.
        Supports single scalar rating_diff or multidimensional lists/tuples/arrays of rating_diff.
        """
        if isinstance(rating_diff, (list, tuple, np.ndarray)):
            diff_vals = [float(v) / 400.0 for v in rating_diff]
        else:
            diff_vals = [float(rating_diff) / 400.0]
            
        neutral = 1.0 if is_neutral else 0.0
        year_val = (float(year) - 1950.0) / 75.0
        tw_val = float(tourn_weight) / 50.0
        
        row_feat = [1.0]
        for d in diff_vals:
            row_feat.append(d)
            if self.specs['response'] == 'Quadratic':
                row_feat.append(d ** 2)
                
        if self.specs['coupling'] == 'Independent':
            row_feat.append(neutral)
        if self.specs['decay']:
            row_feat.append(year_val)
        if self.specs['competition']:
            row_feat.append(tw_val)
            
        x_vec = np.array(row_feat, dtype=float)
        
        dot_h = np.dot(x_vec, self.params_home)
        dot_a = np.dot(x_vec, self.params_away)
        
        dot_h = np.nan_to_num(dot_h, nan=0.0, posinf=10.0, neginf=-10.0)
        dot_a = np.nan_to_num(dot_a, nan=0.0, posinf=10.0, neginf=-10.0)
        
        lh = max(1e-4, min(20.0, np.exp(min(10.0, max(-10.0, dot_h)))))
        la = max(1e-4, min(20.0, np.exp(min(10.0, max(-10.0, dot_a)))))
        
        g_arr = np.arange(max_goals + 1)
        pmf_h = poisson.pmf(g_arr, lh)
        pmf_a = poisson.pmf(g_arr, la)
        
        sum_h = np.sum(pmf_h)
        sum_a = np.sum(pmf_a)
        
        if np.isnan(sum_h) or sum_h <= 1e-12:
            pmf_h = np.ones(max_goals + 1) / float(max_goals + 1)
        else:
            pmf_h /= sum_h
            
        if np.isnan(sum_a) or sum_a <= 1e-12:
            pmf_a = np.ones(max_goals + 1) / float(max_goals + 1)
        else:
            pmf_a /= sum_a
            
        joint_pmf = np.outer(pmf_h, pmf_a)
        if self.is_bivariate:
            tau_grid = np.ones((max_goals + 1, max_goals + 1))
            tau_grid[0, 0] = 1.0 - lh * la * self.rho
            tau_grid[0, 1] = 1.0 + lh * self.rho
            tau_grid[1, 0] = 1.0 + la * self.rho
            tau_grid[1, 1] = 1.0 - self.rho
            joint_pmf *= tau_grid
            joint_pmf = np.maximum(0.0, joint_pmf)
            joint_pmf /= np.sum(joint_pmf)
            
        p_win_h = float(np.sum(np.tril(joint_pmf, -1)))
        p_draw = float(np.sum(np.diag(joint_pmf)))
        p_win_a = float(np.sum(np.triu(joint_pmf, 1)))
        
        most_likely_idx = np.unravel_index(np.argmax(joint_pmf), joint_pmf.shape)
        
        return {
            'model_code': self.model_code,
            'expected_goals_home': float(lh),
            'expected_goals_away': float(la),
            'p_home_win': p_win_h,
            'p_draw': p_draw,
            'p_away_win': p_win_a,
            'scoreline_matrix': joint_pmf,
            'most_likely_score': (int(most_likely_idx[0]), int(most_likely_idx[1])),
            'parameters_k': self.k_tot
        }

    def save(self, file_path=None):
        """Save trained model parameters to JSON file. If file_path is None, saves to standard location."""
        if file_path is None:
            file_path = get_standard_params_path(self.model_code, self.rating_system)
            
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            'model_code': self.model_code,
            'rating_system': self.rating_system,
            'specs': self.specs,
            'params_home': self.params_home.tolist(),
            'params_away': self.params_away.tolist(),
            'rho': self.rho,
            'rating_col': self.rating_col,
            'k_rating': self.k_rating,
            'k_tot': self.k_tot
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return file_path

    @classmethod
    def load(cls, file_path=None, model_code='M32', rating_system='fifa-sum'):
        """Load trained model parameters from JSON file. If file_path is None, loads from standard location."""
        if file_path is None:
            file_path = get_standard_params_path(model_code, rating_system)
            
        if not os.path.exists(file_path):
            local_fallback = f"params/params_{rating_system}_{model_code}.json"
            if os.path.exists(local_fallback):
                file_path = local_fallback
            else:
                raise FileNotFoundError(f"No saved parameters found for model {model_code} and rating system {rating_system} at {file_path}")
                
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return cls(
            model_code=data.get('model_code', model_code),
            params_home=data.get('params_home', data.get('glm_fitted_parameters', {}).get('beta_home', [])),
            params_away=data.get('params_away', data.get('glm_fitted_parameters', {}).get('beta_away', [])),
            rho=data.get('rho', data.get('glm_fitted_parameters', {}).get('rho', 0.0)),
            rating_col=data.get('rating_col', 'fifa_diff'),
            rating_system=data.get('rating_system', rating_system),
            k_rating=data.get('k_rating', 14)
        )

def train_model(df, model_code='M32', rating_col='fifa_diff', tourn_col='tourn_weight', rating_system='fifa-sum', k_rating=14, weighted_training=True):
    """
    Train any of the 64 GLM models (M01-M32 with Scheme A FIFA weights, M33-M64 with Scheme B category/unweighted).
    """
    raw_code = str(model_code).upper().strip()
    m_num = int(raw_code.replace("M", "")) if raw_code.replace("M", "").isdigit() else 32
    
    if m_num > 32:
        base_code = f"M{(m_num - 32):02d}"
        weighted_training = False
    else:
        base_code = f"M{m_num:02d}"
        
    code = base_code
    specs = get_model_specs(code)
    
    X_full = build_design_matrix(df, model_code=code, rating_col=rating_col, tourn_col=tourn_col)
    y_h = df['home_score'].values
    y_a = df['away_score'].values
    
    w_vec = None
    if weighted_training:
        if tourn_col in df.columns:
            w_vec = df[tourn_col].values
        elif 'fifa_weight' in df.columns:
            w_vec = df['fifa_weight'].values
        elif 'eval_weight' in df.columns:
            w_vec = df['eval_weight'].values
            
    try:
        if w_vec is not None:
            glm_h = sm.GLM(y_h, X_full, family=sm.families.Poisson(), freq_weights=w_vec).fit()
        else:
            glm_h = sm.GLM(y_h, X_full, family=sm.families.Poisson()).fit()
    except Exception:
        try:
            glm_h = sm.GLM(y_h, X_full, family=sm.families.Poisson()).fit_regularized(alpha=1e-4, L1_wt=0.0)
        except Exception:
            X_clip = np.clip(X_full.values, -10.0, 10.0)
            X_df = pd.DataFrame(X_clip, columns=X_full.columns)
            glm_h = sm.GLM(y_h, X_df, family=sm.families.Poisson()).fit_regularized(alpha=1e-3, L1_wt=0.0)
            
    try:
        if w_vec is not None:
            glm_a = sm.GLM(y_a, X_full, family=sm.families.Poisson(), freq_weights=w_vec).fit()
        else:
            glm_a = sm.GLM(y_a, X_full, family=sm.families.Poisson()).fit()
    except Exception:
        try:
            glm_a = sm.GLM(y_a, X_full, family=sm.families.Poisson()).fit_regularized(alpha=1e-4, L1_wt=0.0)
        except Exception:
            X_clip = np.clip(X_full.values, -10.0, 10.0)
            X_df = pd.DataFrame(X_clip, columns=X_full.columns)
            glm_a = sm.GLM(y_a, X_df, family=sm.families.Poisson()).fit_regularized(alpha=1e-3, L1_wt=0.0)
    
    mu_h = glm_h.predict(X_full)
    mu_a = glm_a.predict(X_full)
    
    opt_rho = 0.0
    if specs['dist'] == 'DixonColes':
        p0_vec = poisson.pmf(y_h, mu_h) * poisson.pmf(y_a, mu_a)
        def neg_ll_vec(r):
            rho_v = r[0]
            tau = dixon_coles_tau_vec(y_h, y_a, mu_h, mu_a, rho_v)
            if np.any(tau <= 1e-6): return 1e9
            log_p0 = np.log(np.maximum(1e-12, p0_vec * tau))
            if w_vec is not None:
                return -np.sum(log_p0 * w_vec)
            return -np.sum(log_p0)
            
        opt = minimize(neg_ll_vec, [0.0], bounds=[(-0.5, 0.5)], method='L-BFGS-B')
        if opt.success:
            opt_rho = float(opt.x[0])
        
    return TrainedModel(
        model_code=code,
        params_home=glm_h.params.values if hasattr(glm_h.params, 'values') else np.array(glm_h.params),
        params_away=glm_a.params.values if hasattr(glm_a.params, 'values') else np.array(glm_a.params),
        rho=opt_rho,
        rating_col=rating_col,
        rating_system=rating_system,
        k_rating=k_rating
    )
