# `multielo-football` ⚽🏆

**Multi-Dimensional Elo Ratings and Poisson Prediction Hierarchy for International Football**

`multielo-football` is a Python library for computing multi-vector Elo rating architectures and forecasting match outcomes in national team football.

---

## 🌟 Installation

Install locally or from source:

```bash
cd multielo_package
pip install -e .
```

Or via `pip`:
```bash
pip install multielo-football
```

---

## 🚀 Quickstart Usage

```python
import multielo

# 1. Load International Match Dataset
df = multielo.load_dataset()

# 2. Compute Multi-Vector Ratings (e.g. 3-Elo Complete)
# Supported: 'fifa-sum', 'eloratings', '1elo-simple', '1elo-complete', '2elo-pure', '2elo-fast-slow', '3elo-hybrid', '3elo-complete'
df_rated = multielo.compute_ratings(df, system='3elo-complete')

# 3. Predict Match Outcome
spain_ratings = {'elo': 2279.0, 'off': 2276.6, 'def': 2412.4}
england_ratings = {'elo': 2117.7, 'off': 2341.5, 'def': 2226.9}

pred = multielo.predict(spain_ratings, england_ratings, model_specs='M32', is_neutral=True)

print(f"P(Spain Win)   : {pred['p_win_a']*100:.1f}%")
print(f"P(Draw)        : {pred['p_draw']*100:.1f}%")
print(f"P(England Win) : {pred['p_win_b']*100:.1f}%")
print(f"Most Likely Score: {pred['most_likely_score'][0]} - {pred['most_likely_score'][1]}")

# 4. Evaluation Metrics
rps = multielo.compute_rps(pred['p_win_a'], pred['p_draw'], pred['p_win_b'], outcome='H')
esd = multielo.compute_esd(pred['score_matrix'], actual_g_a=2, actual_g_b=1)
print(f"RPS Loss: {rps:.5f} | ESD Loss: {esd:.5f}")
```

---

## 📊 Supported Rating Architectures

| Rating System | Description | Parameters |
| :--- | :--- | :--- |
| `fifa-sum` | Official FIFA/Coca-Cola World Ranking (SUM formula) | Standard FIFA |
| `eloratings` | World Football Elo Ratings (`elorating.net` benchmark) | Standard Elo |
| `1elo-simple` | Parametrized single-scale Elo | 4 |
| `1elo-complete` | Parametrized complete tier-weighted single-scale Elo | 10 |
| `2elo-pure` / `2elo-style` | Decoupled Offensive & Defensive style ratings | 4 |
| `2elo-fast-slow` | Dual-timescale Fast+Slow outcome ratings | 8 |
| `3elo-hybrid` | Overall outcome + Decoupled style ratings | 8 |
| `3elo-complete` | Complete multi-vector outcome + Decoupled style ratings | 11 |

---

## 🛠️ Downloading Kaggle Dataset

Download the latest international football match dataset directly:

```python
import multielo

# Download latest results.csv via kagglehub / kaggle API
path = multielo.download_dataset()
```

---

## 📜 Citation

If you use `multielo-football` in your research, please cite:

```bibtex
@article{rennocosta_csato_2026_multielo,
  title={Comparison of Elo-based prediction models for national team football},
  author={Renn{\'o}-Costa, C{\'e}sar and Csat{\'o}, L{\'a}szl{\'o}},
  journal={Working Paper},
  year={2026}
}
```
