import numpy as np
from scipy.stats import poisson

def predict(team_a_ratings, team_b_ratings, model_specs='M32', is_neutral=False, params=None):
    """
    Predict match outcome probabilities and goal scoreline matrix between Team A and Team B.
    
    Parameters
    ----------
    team_a_ratings : float or dict
        Rating(s) for Team A (Home / Team A).
        - If float: scalar overall Elo rating.
        - If dict: e.g. {'elo': 2100, 'off': 2050, 'def': 2000} or {'overall': 2100, 'off': 2050, 'def': 2000}.
    team_b_ratings : float or dict
        Rating(s) for Team B (Away / Team B).
    model_specs : str
        GLM model architecture code (e.g. 'M32', 'M31', 'M14', 'M01'). Default is 'M32'.
    is_neutral : bool
        Whether match is played at a neutral venue. Default is False.
    params : dict, optional
        Custom GLM and rating system parameters.
        
    Returns
    -------
    dict
        Prediction results containing:
        - 'p_win_a': Win probability for Team A.
        - 'p_draw': Draw probability.
        - 'p_win_b': Win probability for Team B.
        - 'expected_goals_a': Expected goals for Team A (lambda).
        - 'expected_goals_b': Expected goals for Team B (mu).
        - 'score_matrix': 15x15 scoreline probability grid (0..14 goals).
        - 'most_likely_score': Tuple (goals_a, goals_b) with highest probability.
    """
    # Parse ratings
    ra = team_a_ratings if isinstance(team_a_ratings, dict) else {'elo': float(team_a_ratings)}
    rb = team_b_ratings if isinstance(team_b_ratings, dict) else {'elo': float(team_b_ratings)}
    
    elo_a = ra.get('elo', ra.get('overall', 1500.0))
    elo_b = rb.get('elo', rb.get('overall', 1500.0))
    off_a = ra.get('off', ra.get('offensive', elo_a))
    def_a = ra.get('def', ra.get('defensive', elo_a))
    off_b = rb.get('off', rb.get('offensive', elo_b))
    def_b = rb.get('def', rb.get('defensive', elo_b))
    
    # Rating differentials
    elo_diff = elo_a - elo_b
    od_diff_a = off_a - def_b
    od_diff_b = off_b - def_a
    
    # Default parameters for M32 (3-Elo Complete ALL)
    p = params or {
        'b0_h': 0.125, 'b0_a': -0.150, 'bh_h': 0.180, 'bh_a': -0.180,
        'b_elo1': 0.0012, 'b_elo2': 0.000001, 'b_od1': 0.0015, 'b_od2': 0.000001,
        'rho': -0.045, 'mu_base': 1.35, 'divisor_style': 974.5535, 'H_style': 60.9455
    }
    
    # Calculate Poisson intensity rates lambda (Team A) and mu (Team B)
    Ds = float(p.get('divisor_style', 974.5535))
    Hs = float(p.get('H_style', 60.9455)) if not is_neutral else 0.0
    mu_base = float(p.get('mu_base', 1.35))
    
    exp_a = max(-100.0, min(100.0, (od_diff_a + Hs) / Ds))
    lambda_a = mu_base * (10.0 ** exp_a)
    
    exp_b = max(-100.0, min(100.0, (od_diff_b - Hs) / Ds))
    lambda_b = mu_base * (10.0 ** exp_b)
    
    # Build 15x15 goal scoreline probability matrix (0 to 14 goals)
    max_g = 14
    goals = np.arange(max_g + 1)
    pa = poisson.pmf(goals, max(1e-5, lambda_a))
    pb = poisson.pmf(goals, max(1e-5, lambda_b))
    pa /= pa.sum()
    pb /= pb.sum()
    
    matrix = np.outer(pa, pb)
    
    # Apply Dixon-Coles low-score adjustment if model is bivariate (M17-M32)
    model_code = str(model_specs).upper().strip()
    is_bivariate = model_code.startswith('B') or (model_code.startswith('M') and int(model_code[1:]) >= 17)
    
    if is_bivariate:
        rho = float(p.get('rho', -0.045))
        tau = np.ones((max_g + 1, max_g + 1))
        tau[0, 0] = 1.0 - lambda_a * lambda_b * rho
        tau[1, 0] = 1.0 + lambda_a * rho
        tau[0, 1] = 1.0 + lambda_b * rho
        tau[1, 1] = 1.0 - rho
        matrix = np.maximum(0.0, matrix * tau)
        matrix /= matrix.sum()
        
    p_win_a = float(np.sum(np.tril(matrix, k=-1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_win_b = float(np.sum(np.triu(matrix, k=1)))
    
    most_likely_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    
    return {
        'p_win_a': p_win_a,
        'p_draw': p_draw,
        'p_win_b': p_win_b,
        'expected_goals_a': float(lambda_a),
        'expected_goals_b': float(lambda_b),
        'score_matrix': matrix,
        'most_likely_score': (int(most_likely_idx[0]), int(most_likely_idx[1]))
    }
