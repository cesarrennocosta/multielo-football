import numpy as np

def compute_rps(p_win_a, p_draw, p_win_b, outcome):
    """
    Compute Ranked Probability Score (RPS) for trichotomous outcome forecasts.
    
    Parameters
    ----------
    p_win_a : float
        Predicted probability of Team A win.
    p_draw : float
        Predicted probability of draw.
    p_win_b : float
        Predicted probability of Team B win (Away win).
    outcome : str or int
        Observed match outcome: 'H' / 1 (Home win), 'D' / 0 (Draw), 'A' / -1 (Away win).
        
    Returns
    -------
    float
        RPS loss score (lower is better).
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
    
    Parameters
    ----------
    score_matrix : np.ndarray
        15x15 scoreline joint probability matrix (0..14 goals).
    actual_g_a : int
        Actual goals scored by Team A.
    actual_g_b : int
        Actual goals scored by Team B.
    w_cat : float
        Weight for outcome category error.
    w_gd : float
        Weight for goal difference discrepancy.
    w_vol : float
        Weight for total goal volume error.
        
    Returns
    -------
    float
        ESD scoreline error metric (lower is better).
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
    
    Parameters
    ----------
    log_likelihood : float
        Maximized joint log-likelihood.
    k_params : int
        Total parameter count (rating parameters + GLM parameters).
        
    Returns
    -------
    float
        AIC score (lower is better).
    """
    return 2.0 * k_params - 2.0 * log_likelihood
