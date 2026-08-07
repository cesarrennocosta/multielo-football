import os
import shutil
import pandas as pd

def run_download_dataset():
    """
    Download the international football match results dataset and save to paper_scripts/data/results.csv.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    target_csv = os.path.join(data_dir, 'results.csv')
    
    print("=== Step 1: Downloading / Syncing Match Dataset ===")
    
    # Method A: Try kagglehub
    downloaded = False
    try:
        import kagglehub
        print("Attempting download via kagglehub (martj42/international-football-results-from-1872-to-2017)...")
        path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
        # Find results.csv in downloaded directory
        for root, _, files in os.walk(path):
            if 'results.csv' in files:
                src_path = os.path.join(root, 'results.csv')
                shutil.copy(src_path, target_csv)
                downloaded = True
                print(f"Successfully downloaded and copied dataset to: {target_csv}")
                break
    except Exception as e:
        print(f"kagglehub notice: {e}")
        
    # Method B: Fallback to local source dataset
    if not downloaded:
        parent_dir = os.path.dirname(os.path.dirname(script_dir))
        candidates = [
            os.path.join(parent_dir, 'datset', 'results.csv'),
            os.path.join(parent_dir, 'repository', 'multiElo', 'data', 'results.csv'),
            os.path.join(parent_dir, 'world_cup_features_dataset.csv')
        ]
        for c in candidates:
            if os.path.exists(c):
                shutil.copy(c, target_csv)
                downloaded = True
                print(f"Copied local reference dataset from {c} to {target_csv}")
                break

    if not downloaded or not os.path.exists(target_csv):
        raise FileNotFoundError("Failed to download or locate dataset. Please place 'results.csv' inside paper_scripts/data/.")

    # Validate dataset content
    df = pd.read_csv(target_csv, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score']).sort_values('date').reset_index(drop=True)
    
    print(f"Dataset successfully prepared at: {target_csv}")
    print(f"Total Match Records: {len(df):,}")
    print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

if __name__ == '__main__':
    run_download_dataset()
