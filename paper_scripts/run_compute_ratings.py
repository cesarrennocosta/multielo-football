import os
import sys
import json
import argparse
import pandas as pd

# Import multielo package
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

import multielo

# Mapping systems to parameter JSON file candidates and output CSV filenames
SYSTEM_CONFIGS = {
    '3eloC': {
        'system': '3elo-complete',
        'params_file': 'params_rating_3eloC_M32_ALL.json',
        'output': 'ratings_3eloC.csv'
    },
    '3eloH': {
        'system': '3elo-hybrid',
        'params_file': 'params_rating_3eloH_M32_ALL.json',
        'output': 'ratings_3eloH.csv'
    },
    '3eloOD': {
        'system': '2elo-pure',
        'params_file': 'params_rating_2eloOD_pure_M32_ALL.json',
        'output': 'ratings_3eloOD.csv'
    },
    '1eloC': {
        'system': '1elo-complete',
        'params_file': 'params_rating_1eloC_M32_ALL.json',
        'output': 'ratings_1eloC.csv'
    },
    '1eloF': {
        'system': '1elo-simple',
        'params_file': 'params_rating_1eloF_M32_ALL.json',
        'output': 'ratings_1eloF.csv'
    },
    'eloratings': {
        'system': 'eloratings',
        'params_file': None,
        'output': 'ratings_eloratings.csv'
    },
    'fifa': {
        'system': 'fifa-sum',
        'params_file': None,
        'output': 'ratings_fifa.csv'
    }
}

# Alias lookup for user-friendly flags
SYSTEM_ALIASES = {
    '3eloc': '3eloC', '3elo-complete': '3eloC',
    '3eloh': '3eloH', '3elo-hybrid': '3eloH',
    '3elood': '3eloOD', '2elood_pure': '3eloOD', '2elo-pure': '3eloOD', '2elo-style': '3eloOD',
    '1eloc': '1eloC', '1elo-complete': '1eloC',
    '1elof': '1eloF', '1elo-simple': '1eloF', '1elos': '1eloF',
    'eloratings': 'eloratings', 'eloratings.net': 'eloratings',
    'fifa': 'fifa', 'fifa-sum': 'fifa'
}


def load_params_file(params_dir, filename):
    if not filename:
        return None
    file_path = os.path.join(params_dir, filename)
    if os.path.exists(file_path):
        print(f"Loading system parameters from: {file_path}")
        with open(file_path, 'r') as f:
            return json.load(f)
    print(f"Notice: Parameter file {filename} not found in params/. Using default system parameters.")
    return None


def run_compute_single_system(sys_key, df, script_dir, startdate=None, stopdate=None):
    config = SYSTEM_CONFIGS[sys_key]
    params_dir = os.path.join(script_dir, 'params')
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Filter dataset by startdate and stopdate (inclusive)
    df_filtered = df.copy()
    if startdate:
        start_dt = pd.to_datetime(startdate)
        df_filtered = df_filtered[df_filtered['date'] >= start_dt]
        print(f"Filter startdate (inclusive): {start_dt.strftime('%Y-%m-%d')}")
    if stopdate:
        stop_dt = pd.to_datetime(stopdate)
        df_filtered = df_filtered[df_filtered['date'] <= stop_dt]
        print(f"Filter stopdate (inclusive): {stop_dt.strftime('%Y-%m-%d')}")
        
    df_filtered = df_filtered.reset_index(drop=True)
    print(f"Matches to process after date filtering: {len(df_filtered):,}")
    
    params = load_params_file(params_dir, config['params_file'])
    
    print(f"Computing ratings for system '{sys_key}' ({config['system']})...")
    df_rated = multielo.compute_ratings(df_filtered, system=config['system'], params=params)
    
    output_path = os.path.join(data_dir, config['output'])
    df_rated.to_csv(output_path, index=False)
    print(f"Successfully saved computed ratings to: {output_path}\n")
    return output_path


def run_compute_ratings(system_name='3eloC', startdate=None, stopdate=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    dataset_path = os.path.join(data_dir, 'results.csv')
    
    print(f"=== Running Rating Computation (Requested: '{system_name}') ===")
    df = multielo.load_dataset(path=dataset_path)
    
    key = str(system_name).lower().strip()
    
    if key == 'all':
        print("Computing ratings for ALL supported systems...")
        for s_key in SYSTEM_CONFIGS.keys():
            run_compute_single_system(s_key, df, script_dir, startdate=startdate, stopdate=stopdate)
    else:
        canonical_key = SYSTEM_ALIASES.get(key, None)
        if not canonical_key or canonical_key not in SYSTEM_CONFIGS:
            valid_keys = list(SYSTEM_CONFIGS.keys()) + ['all']
            raise ValueError(f"Invalid system '{system_name}'. Choose from: {valid_keys}")
            
        run_compute_single_system(canonical_key, df, script_dir, startdate=startdate, stopdate=stopdate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute chronological Elo ratings using multielo package.")
    parser.add_argument('--system', type=str, default='3eloC',
                        help="Rating system to compute (e.g. 3eloC, 3eloH, 3eloOD, 1eloC, 1eloF, eloratings, fifa, all). Default is 3eloC.")
    parser.add_argument('--startdate', type=str, default=None,
                        help="Start date for inclusive match filtering (YYYY-MM-DD). Matches before this date are ignored.")
    parser.add_argument('--stopdate', type=str, default=None,
                        help="Stop date for inclusive match filtering (YYYY-MM-DD). Matches after this date are ignored.")
    args = parser.parse_args()
    
    run_compute_ratings(system_name=args.system, startdate=args.startdate, stopdate=args.stopdate)
