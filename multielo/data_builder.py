import os
import sys
import pandas as pd
import numpy as np

def categorize_tournament_class(tournament_name):
    """
    Categorizes a match tournament string into 4 balanced classes:
      1. 'world_cup': FIFA World Cup final tournament matches.
      2. 'continental_confed': UEFA Euro, Copa América, AFCON, Asian Cup, Gold Cup, Confederations Cup, Nations League Finals.
      3. 'qualifiers': World Cup and Continental qualifiers.
      4. 'friendlies_lower': Friendlies and lower-level regional championships/cups.
    """
    t = str(tournament_name).lower().strip()
    if t in ['fifa world cup', 'world cup']:
        return 'world_cup'
    elif 'qualification' in t or 'qualifying' in t:
        return 'qualifiers'
    elif any(k in t for k in ['uefa euro', 'copa américa', 'copa america', 'african cup of nations', 'afc asian cup', 'gold cup', 'confederations cup', 'nations league finals']):
        return 'continental_confed'
    else:
        return 'friendlies_lower'

def get_fifa_base_weight(tournament_name):
    """
    Returns base FIFA topology weight for a tournament:
      - World Cup Finals: 60.0
      - Continental Finals / Confederations Cup: 50.0
      - Qualifiers (WC & Continental) / Nations League: 25.0
      - Friendlies / Lower Tiers: 10.0
    """
    t_l = str(tournament_name).lower().strip()
    if t_l in ['fifa world cup', 'world cup']:
        return 60.0
    elif any(k in t_l for k in ['uefa euro', 'copa américa', 'copa america', 'african cup of nations', 'afc asian cup', 'gold cup', 'confederations cup', 'nations league finals']):
        return 50.0
    elif 'qualification' in t_l or 'qualifying' in t_l or 'nations league' in t_l:
        return 25.0
    else:
        return 10.0

def build_balanced_learning_dataset(df_raw, random_state=42, multiplier_qual_friendly=4, save_path=None):
    """
    Constructs the Stratified 4x-Expanded Balanced Learning Dataset from 1950 to present.
    """
    df = df_raw.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
    else:
        raise ValueError("DataFrame must contain a 'date' column.")

    df_50 = df[df['year'] >= 1950].copy().sort_values('date').reset_index(drop=True)
    
    tourn_col = 'tournament' if 'tournament' in df_50.columns else 'tourn_name'
    df_50['tourn_class_balanced'] = df_50[tourn_col].apply(categorize_tournament_class)
    
    df_wc = df_50[df_50['tourn_class_balanced'] == 'world_cup'].sort_values('date').reset_index(drop=True)
    n_wc = len(df_wc)
    
    if n_wc == 0:
        raise ValueError("No post-1950 World Cup final matches found in dataset.")
        
    df_cc = df_50[df_50['tourn_class_balanced'] == 'continental_confed'].sample(n=min(len(df_50[df_50['tourn_class_balanced'] == 'continental_confed']), n_wc), random_state=random_state).sort_values('date').reset_index(drop=True)
    df_qu = df_50[df_50['tourn_class_balanced'] == 'qualifiers'].sample(n=min(len(df_50[df_50['tourn_class_balanced'] == 'qualifiers']), multiplier_qual_friendly * n_wc), random_state=random_state).sort_values('date').reset_index(drop=True)
    df_fl = df_50[df_50['tourn_class_balanced'] == 'friendlies_lower'].sample(n=min(len(df_50[df_50['tourn_class_balanced'] == 'friendlies_lower']), multiplier_qual_friendly * n_wc), random_state=random_state).sort_values('date').reset_index(drop=True)

    # Assign Class Factor (0.25 for 4x sampled classes, 1.0 for 1x sampled classes)
    df_wc['class_factor'] = 1.0
    df_cc['class_factor'] = 1.0
    df_qu['class_factor'] = 1.0 / float(multiplier_qual_friendly)
    df_fl['class_factor'] = 1.0 / float(multiplier_qual_friendly)

    # Stratified 5-CV Fold Allocation per class
    for sub_df in [df_wc, df_cc, df_qu, df_fl]:
        sub_df['fold'] = np.arange(len(sub_df)) % 5

    df_balanced = pd.concat([df_wc, df_cc, df_qu, df_fl], ignore_index=True).sort_values('date').reset_index(drop=True)
    
    # Compute FIFA Base Weight & Effective Fast/Slow Weights
    df_balanced['fifa_base_weight'] = df_balanced[tourn_col].apply(get_fifa_base_weight)
    df_balanced['w_fast'] = df_balanced['fifa_base_weight'] * df_balanced['class_factor']
    df_balanced['w_slow'] = np.where(df_balanced['tourn_class_balanced'] == 'friendlies_lower', 0.0, df_balanced['w_fast'])
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(curr_dir))
    
    default_save_paths = [
        os.path.join(curr_dir, "..", "paper", "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(curr_dir, "..", "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(project_root, "paper", "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(project_root, "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(project_root, "world_cup_balanced_5cv_dataset.csv"),
        "/home/crcosta/multielo_package/paper/data/balanced_learning_dataset_5cv.csv",
        "/home/crcosta/paper/data/balanced_learning_dataset_5cv.csv"
    ]
    if save_path:
        default_save_paths.insert(0, save_path)
        
    for p in set(default_save_paths):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            df_balanced.to_csv(p, index=False)
        except Exception:
            pass
            
    return df_balanced

def get_balanced_learning_dataset(df_raw=None, random_state=42, force_rebuild=False):
    """
    Retrieves the persisted balanced 5-CV learning dataset, or builds it from df_raw if missing/forced.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(curr_dir))
    
    possible_paths = [
        "/home/crcosta/multielo_package/paper/data/balanced_learning_dataset_5cv.csv",
        "/home/crcosta/multielo_package/multielo_package/paper/data/balanced_learning_dataset_5cv.csv",
        "/home/crcosta/paper/data/balanced_learning_dataset_5cv.csv",
        os.path.join(curr_dir, "..", "paper", "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(curr_dir, "..", "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(project_root, "paper", "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(project_root, "data", "balanced_learning_dataset_5cv.csv"),
        os.path.join(project_root, "world_cup_balanced_5cv_dataset.csv"),
        "balanced_learning_dataset_5cv.csv",
        "paper/data/balanced_learning_dataset_5cv.csv"
    ]
    
    if not force_rebuild:
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                except Exception:
                    pass
                    
    if df_raw is None:
        raw_paths = [
            "/home/crcosta/multielo_package/paper/data/world_cup_features_dataset.csv",
            "/home/crcosta/multielo_package/world_cup_features_dataset.csv",
            "/home/crcosta/paper/data/world_cup_features_dataset.csv",
            os.path.join(curr_dir, "..", "paper", "data", "world_cup_features_dataset.csv"),
            os.path.join(project_root, "paper", "data", "world_cup_features_dataset.csv"),
            os.path.join(project_root, "world_cup_features_dataset.csv"),
            "paper/data/world_cup_features_dataset.csv",
            "world_cup_features_dataset.csv"
        ]
        for rp in raw_paths:
            if os.path.exists(rp):
                df_raw = pd.read_csv(rp)
                break
                
    if df_raw is None:
        raise FileNotFoundError("Raw features dataset not found to build balanced learning dataset.")
        
    return build_balanced_learning_dataset(df_raw, random_state=random_state)
