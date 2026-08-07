"""
Taxonomy and Specifications for the 32 Poisson GLM Models (M01-M32).
"""

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

def get_model_specs(model_code='M32'):
    """
    Retrieve architectural specifications for a given GLM model code (M01-M32).
    """
    code = str(model_code).upper().strip()
    if code not in GLM_TAXONOMY:
        raise ValueError(f"Unknown model code '{model_code}'. Must be between M01 and M32.")
    return GLM_TAXONOMY[code]
