import os
import pandas as pd

def load_dataset(path=None):
    """
    Load international football match results dataset.
    
    Parameters
    ----------
    path : str, optional
        Path to local CSV dataset. If None, resolves from default workspace locations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing cleaned international match records sorted chronologically.
    """
    if path and os.path.exists(path):
        target_path = path
    else:
        # Search candidate locations
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(base_dir, 'datset', 'results.csv'),
            os.path.join(base_dir, 'repository', 'multiElo', 'data', 'results.csv'),
            os.path.join(base_dir, 'world_cup_features_dataset.csv'),
            'results.csv'
        ]
        target_path = None
        for c in candidates:
            if os.path.exists(c):
                target_path = c
                break
                
    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError("Could not locate match dataset. Please provide path or run multielo.download_dataset().")
        
    print(f"Loading match dataset from: {target_path}")
    df = pd.read_csv(target_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score']).sort_values(by='date').reset_index(drop=True)
    return df

def download_dataset(dataset_name="martj42/international-football-results-from-1872-to-2017", output_dir="."):
    """
    Download the latest international football results dataset from Kaggle.
    
    Parameters
    ----------
    dataset_name : str
        Kaggle dataset identifier. Default is 'martj42/international-football-results-from-1872-to-2017'.
    output_dir : str
        Target directory to save the downloaded results.csv.
        
    Returns
    -------
    str
        Path to downloaded CSV dataset.
    """
    try:
        import kagglehub
        print(f"Downloading Kaggle dataset: {dataset_name} via kagglehub...")
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {path}")
        return path
    except ImportError:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print(f"Downloading Kaggle dataset: {dataset_name} via kaggle API...")
            api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
            target_csv = os.path.join(output_dir, 'results.csv')
            return target_csv
        except Exception as e:
            raise RuntimeError(
                f"Kaggle API / kagglehub not configured. Install kagglehub (`pip install kagglehub`) "
                f"or place 'results.csv' manually. Error: {e}"
            )
