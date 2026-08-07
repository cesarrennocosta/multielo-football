import multielo

def main():
    print(f"=== MultiElo Package Quickstart (v{multielo.__version__}) ===")
    
    # 1. Load match dataset
    try:
        df = multielo.load_dataset()
        print(f"Dataset loaded: {len(df)} total match records.")
    except Exception as e:
        print(f"Notice: {e}")
        print("You can download the dataset using `multielo.download_dataset()`.")
        return
        
    # 2. Compute Ratings for 3-Elo Complete system
    print("\nComputing 3-Elo Complete ratings across match history...")
    df_rated = multielo.compute_ratings(df, system='3elo-complete')
    print("Ratings calculated successfully!")
    
    # Extract recent rating snapshot for Spain vs England
    spain_row = df_rated[(df_rated['home_team'] == 'Spain') | (df_rated['away_team'] == 'Spain')].iloc[-1]
    spain_ratings = {
        'elo': spain_row['elo_home'] if spain_row['home_team'] == 'Spain' else spain_row['elo_away'],
        'off': spain_row['off_home'] if spain_row['home_team'] == 'Spain' else spain_row['off_away'],
        'def': spain_row['def_home'] if spain_row['home_team'] == 'Spain' else spain_row['def_away']
    }
    
    england_row = df_rated[(df_rated['home_team'] == 'England') | (df_rated['away_team'] == 'England')].iloc[-1]
    england_ratings = {
        'elo': england_row['elo_home'] if england_row['home_team'] == 'England' else england_row['elo_away'],
        'off': england_row['off_home'] if england_row['home_team'] == 'England' else england_row['off_away'],
        'def': england_row['def_home'] if england_row['home_team'] == 'England' else england_row['def_away']
    }
    
    print(f"\nSpain Ratings: {spain_ratings}")
    print(f"England Ratings: {england_ratings}")
    
    # 3. Predict Match Outcome using M32 GLM Model
    print("\nPredicting match outcome (Spain vs England, Neutral Venue) using Model M32...")
    pred = multielo.predict(spain_ratings, england_ratings, model_specs='M32', is_neutral=True)
    
    print(f"P(Spain Win) : {pred['p_win_a']*100:.2f}%")
    print(f"P(Draw)      : {pred['p_draw']*100:.2f}%")
    print(f"P(England Win): {pred['p_win_b']*100:.2f}%")
    print(f"Expected Score : Spain {pred['expected_goals_a']:.2f} - {pred['expected_goals_b']:.2f} England")
    print(f"Most Likely Scoreline: {pred['most_likely_score'][0]} - {pred['most_likely_score'][1]}")
    
    # 4. Compute Loss Metrics
    rps = multielo.compute_rps(pred['p_win_a'], pred['p_draw'], pred['p_win_b'], outcome='H')
    esd = multielo.compute_esd(pred['score_matrix'], actual_g_a=2, actual_g_b=1)
    print(f"\nEvaluation Metrics (Hypothetical 2-1 Spain win):")
    print(f"RPS Loss : {rps:.5f}")
    print(f"ESD Loss : {esd:.5f}")

if __name__ == '__main__':
    main()
