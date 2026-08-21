import os
import json
import numpy as np
from scipy.stats import poisson
from .models import get_model_specs, TrainedModel

def predict(team_a_ratings, team_b_ratings, model_specs='M32', is_neutral=False, year=2025, tourn_weight=30.0, params=None, trained_model=None, max_goals=14):
    """
    Predict match outcome probabilities and goal scoreline matrix between Team A and Team B
    using ANY of the 32 GLM models (M01 to M32).
    
    Parameters
    ----------
    team_a_ratings : float or dict
        Rating(s) for Team A (Home / Team A).
        - If float: scalar overall rating or rating difference $\Delta R$.
        - If dict: e.g. {'elo': 2100} or rating difference.
    team_b_ratings : float or dict
        Rating(s) for Team B (Away / Team B).
    model_specs : str or dict
        GLM model architecture code (e.g. 'M32', 'M16', 'M01'). Default is 'M32'.
    is_neutral : bool
        Whether match is played at a neutral venue. Default is False.
    year : int or float
        Match year (used for time-decay models M03, M04, ..., M32). Default is 2025.
    tourn_weight : float
        Competition weight (used for competition-weighted models M02, M04, ..., M32). Default is 30.0.
    params : dict, optional
        Custom GLM parameters dictionary containing 'params_home', 'params_away', and optional 'rho'.
    trained_model : TrainedModel, optional
        Pre-trained model instance returned by `multielo.train_model(...)`.
    max_goals : int
        Maximum number of goals per team in scoreline grid. Default is 14.
        
    Returns
    -------
    dict
        Prediction results containing:
        - 'p_win_a': Win probability for Team A.
        - 'p_draw': Draw probability.
        - 'p_win_b': Win probability for Team B.
        - 'expected_goals_a': Expected goals for Team A (lambda).
        - 'expected_goals_b': Expected goals for Team B (mu).
        - 'score_matrix': (max_goals+1)x(max_goals+1) scoreline probability grid.
        - 'most_likely_score': Tuple (goals_a, goals_b) with highest probability.
        - 'model_code': Architecture code used.
    """
    if trained_model is not None and isinstance(trained_model, TrainedModel):
        if isinstance(team_a_ratings, (int, float)) and isinstance(team_b_ratings, (int, float)):
            diff = float(team_a_ratings) - float(team_b_ratings)
        elif isinstance(team_a_ratings, (int, float)) and team_b_ratings is None:
            diff = float(team_a_ratings)
        else:
            ra = team_a_ratings if isinstance(team_a_ratings, dict) else {'elo': float(team_a_ratings)}
            rb = team_b_ratings if isinstance(team_b_ratings, dict) else {'elo': float(team_b_ratings)}
            diff = ra.get('elo', 1500.0) - rb.get('elo', 1500.0)
            
        res = trained_model.predict(diff, is_neutral=is_neutral, year=year, tourn_weight=tourn_weight, max_goals=max_goals)
        return {
            'p_win_a': res['p_home_win'],
            'p_draw': res['p_draw'],
            'p_win_b': res['p_away_win'],
            'expected_goals_a': res['expected_goals_home'],
            'expected_goals_b': res['expected_goals_away'],
            'score_matrix': res['scoreline_matrix'],
            'most_likely_score': res['most_likely_score'],
            'model_code': res['model_code'],
            'parameters_k': res['parameters_k']
        }
        
    model_code = str(model_specs).upper().strip() if isinstance(model_specs, str) else 'M32'
    specs = get_model_specs(model_code)
    
    # Calculate rating difference
    if isinstance(team_a_ratings, (int, float)) and isinstance(team_b_ratings, (int, float)):
        diff = float(team_a_ratings) - float(team_b_ratings)
    elif isinstance(team_a_ratings, (int, float)) and team_b_ratings is None:
        diff = float(team_a_ratings)
    else:
        ra = team_a_ratings if isinstance(team_a_ratings, dict) else {'elo': float(team_a_ratings)}
        rb = team_b_ratings if isinstance(team_b_ratings, dict) else {'elo': float(team_b_ratings)}
        diff = ra.get('elo', 1500.0) - rb.get('elo', 1500.0)
        
    diff_norm = diff / 400.0
    neutral = 1.0 if is_neutral else 0.0
    year_val = (float(year) - 1950.0) / 75.0
    tw_val = float(tourn_weight) / 50.0
    
    # Construct feature vector
    row_feat = [1.0, diff_norm]
    if specs['response'] == 'Quadratic':
        row_feat.append(diff_norm ** 2)
    if specs['coupling'] == 'Independent':
        row_feat.append(neutral)
    if specs['decay']:
        row_feat.append(year_val)
    if specs['competition']:
        row_feat.append(tw_val)
        
    x_vec = np.array(row_feat, dtype=float)
    
    # Extract parameter weights
    if params and 'params_home' in params and 'params_away' in params:
        beta_h = np.array(params['params_home'], dtype=float)
        beta_a = np.array(params['params_away'], dtype=float)
        rho = float(params.get('rho', 0.0))
    elif params and 'glm_fitted_parameters' in params:
        gfp = params['glm_fitted_parameters']
        beta_h = np.array(gfp.get('beta_home', []), dtype=float)
        beta_a = np.array(gfp.get('beta_away', []), dtype=float)
        rho = float(gfp.get('rho', 0.0))
    else:
        # Default heuristics for un-trained prediction
        bh_const = 0.125
        ba_const = -0.150
        b1_h = 0.35
        b1_a = -0.35
        beta_h = np.array([bh_const, b1_h] + [0.0] * (len(x_vec) - 2))
        beta_a = np.array([ba_const, b1_a] + [0.0] * (len(x_vec) - 2))
        rho = -0.045 if specs['dist'] == 'DixonColes' else 0.0

    # Ensure length match
    if len(beta_h) != len(x_vec):
        beta_h = np.resize(beta_h, len(x_vec))
    if len(beta_a) != len(x_vec):
        beta_a = np.resize(beta_a, len(x_vec))
        
    lh = max(1e-4, np.exp(np.dot(x_vec, beta_h)))
    la = max(1e-4, np.exp(np.dot(x_vec, beta_a)))
    
    g_arr = np.arange(max_goals + 1)
    pmf_h = poisson.pmf(g_arr, lh)
    pmf_a = poisson.pmf(g_arr, la)
    pmf_h /= np.sum(pmf_h)
    pmf_a /= np.sum(pmf_a)
    
    joint_pmf = np.outer(pmf_h, pmf_a)
    if specs['dist'] == 'DixonColes':
        tau_grid = np.ones((max_goals + 1, max_goals + 1))
        tau_grid[0, 0] = 1.0 - lh * la * rho
        tau_grid[0, 1] = 1.0 + lh * rho
        tau_grid[1, 0] = 1.0 + la * rho
        tau_grid[1, 1] = 1.0 - rho
        joint_pmf *= tau_grid
        joint_pmf = np.maximum(0.0, joint_pmf)
        joint_pmf /= np.sum(joint_pmf)
        
    p_win_a = float(np.sum(np.tril(joint_pmf, -1)))
    p_draw = float(np.sum(np.diag(joint_pmf)))
    p_win_b = float(np.sum(np.triu(joint_pmf, 1)))
    
    most_likely_idx = np.unravel_index(np.argmax(joint_pmf), joint_pmf.shape)
    
    return {
        'p_win_a': p_win_a,
        'p_draw': p_draw,
        'p_win_b': p_win_b,
        'expected_goals_a': float(lh),
        'expected_goals_b': float(la),
        'score_matrix': joint_pmf,
        'most_likely_score': (int(most_likely_idx[0]), int(most_likely_idx[1])),
        'model_code': model_code
    }
