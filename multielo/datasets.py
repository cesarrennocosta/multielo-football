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
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = None
    
    if path and os.path.exists(path):
        target_path = path
    else:
        # Search candidate locations
        candidates = [
            os.path.join(pkg_dir, 'paper_scripts', 'data', 'results.csv'),
            os.path.join(os.getcwd(), 'paper_scripts', 'data', 'results.csv'),
            os.path.join(pkg_dir, 'data', 'results.csv'),
            'results.csv'
        ]
        for c in candidates:
            if os.path.exists(c):
                target_path = c
                break
                
    if not target_path or not os.path.exists(target_path):
        # Auto-download via HTTPS fallback if dataset is missing
        print("Dataset missing. Attempting automatic download from raw.githubusercontent.com...")
        try:
            import urllib.request
            target_path = os.path.join(pkg_dir, 'paper_scripts', 'data', 'results.csv')
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
            urllib.request.urlretrieve(url, target_path)
            print(f"Successfully downloaded match dataset to: {target_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not locate or download match dataset. Error: {e}")
        
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
