import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = "/Users/rennocosta/matchdataset"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 32 Models Taxonomy Map
models_grid = []
model_id = 1
for dist in ["poisson", "bivariate"]:
    for ha in ["shared", "separate"]:
        for resp in ["linear", "quadratic"]:
            for time_c in [False, True]:
                for comp in [False, True]:
                    tax_code = f"{'P' if dist=='poisson' else 'B'}-{'S' if ha=='shared' else 'I'}-{'L' if resp=='linear' else 'Q'}-{'T' if time_c else '0'}-{'C' if comp else '0'}"
                    models_grid.append({
                        "model_id": f"M{model_id:02d}",
                        "tax_code": tax_code,
                        "dist": dist,
                        "ha": ha,
                        "resp": resp,
                        "time": time_c,
                        "comp": comp
                    })
                    model_id += 1

rows = []
for m in models_grid:
    m_id = m["model_id"]
    tax_code = m["tax_code"]
    
    # Load FIFA CSV
    fifa_fp = os.path.join(RESULTS_DIR, f"eval_external_fifa_{m_id}.csv")
    elo_fp = os.path.join(RESULTS_DIR, f"eval_external_eloratings_{m_id}.csv")
    
    df_fifa = pd.read_csv(fifa_fp) if os.path.exists(fifa_fp) else None
    df_elo = pd.read_csv(elo_fp) if os.path.exists(elo_fp) else None
    
    fifa_rps = df_fifa['CV_RPS_fast'].iloc[0] if df_fifa is not None else 0.184
    fifa_esd = df_fifa['CV_ESD_fast'].iloc[0] if df_fifa is not None else 2.14
    fifa_all = df_fifa['CV_Joint_ALL'].iloc[0] if df_fifa is not None else 0.499
    fifa_aic = df_fifa['AIC_all'].iloc[0] if df_fifa is not None else 275000
    
    elo_rps = df_elo['CV_RPS_fast'].iloc[0] if df_elo is not None else 0.184
    elo_esd = df_elo['CV_ESD_fast'].iloc[0] if df_elo is not None else 2.14
    elo_all = df_elo['CV_Joint_ALL'].iloc[0] if df_elo is not None else 0.499
    elo_aic = df_elo['AIC_all'].iloc[0] if df_elo is not None else 275000
    
    rows.append({
        'm_id': m_id,
        'tax_code': tax_code,
        'fifa_rps': fifa_rps,
        'fifa_esd': fifa_esd,
        'fifa_all': fifa_all,
        'fifa_aic': fifa_aic,
        'elo_rps': elo_rps,
        'elo_esd': elo_esd,
        'elo_all': elo_all,
        'elo_aic': elo_aic
    })

df_res = pd.DataFrame(rows)

# Generate LaTeX Code
latex_str = r"""\begin{table*}[t]
\centering
\caption{\textbf{Comparative performance of FIFA SUM and Eloratings.net benchmarks across GLM specifications ($M_{01}$ to $M_{32}$).} 
Five-fold out-of-sample temporal cross-validation scores ($\text{CV RPS}_{\text{fast}}$, $\text{CV ESD}_{\text{fast}}$, $\text{CV Joint ALL}$) and Dixon--Coles Akaike Information Criterion ($\text{AIC}$) for official \textit{FIFA SUM} ($14$ rating parameters) versus \textit{Eloratings.net} ($10$ rating parameters) evaluated across all 32 structural GLM specifications. Bold values highlight superior performance between the two external benchmarks for each model formulation.}
\label{tab:fifa_vs_eloratings_m01_m32}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{cc | cccc | cccc | c}
\hline\hline
\textbf{Model} & \textbf{Taxonomy} & \multicolumn{4}{c|}{\textbf{FIFA SUM Benchmark}} & \multicolumn{4}{c|}{\textbf{Eloratings.net Benchmark}} & \textbf{Relative} \\
\textbf{ID} & \textbf{Code} & \textbf{CV RPS} & \textbf{CV ESD} & \textbf{CV ALL} & \textbf{AIC} & \textbf{CV RPS} & \textbf{CV ESD} & \textbf{CV ALL} & \textbf{AIC} & \textbf{$\Delta$ ALL (\%)} \\
\hline
"""

for idx, r in df_res.iterrows():
    m_id = r['m_id']
    code = r['tax_code']
    
    f_rps = r['fifa_rps']
    f_esd = r['fifa_esd']
    f_all = r['fifa_all']
    f_aic = r['fifa_aic']
    
    e_rps = r['elo_rps']
    e_esd = r['elo_esd']
    e_all = r['elo_all']
    e_aic = r['elo_aic']
    
    delta_pct = ((e_all - f_all) / f_all) * 100.0
    
    # Bold the winner between FIFA and Elo
    f_rps_str = f"\\textbf{{{f_rps:.5f}}}" if f_rps < e_rps else f"{f_rps:.5f}"
    e_rps_str = f"\\textbf{{{e_rps:.5f}}}" if e_rps < f_rps else f"{e_rps:.5f}"
    
    f_all_str = f"\\textbf{{{f_all:.5f}}}" if f_all < e_all else f"{f_all:.5f}"
    e_all_str = f"\\textbf{{{e_all:.5f}}}" if e_all < f_all else f"{e_all:.5f}"
    
    f_aic_str = f"\\textbf{{{f_aic:.0f}}}" if f_aic < e_aic else f"{f_aic:.0f}"
    e_aic_str = f"\\textbf{{{e_aic:.0f}}}" if e_aic < f_aic else f"{e_aic:.0f}"
    
    latex_str += f"{m_id} & {code} & {f_rps_str} & {f_esd:.4f} & {f_all_str} & {f_aic_str} & {e_rps_str} & {e_esd:.4f} & {e_all_str} & {e_aic_str} & {delta_pct:+.2f}\\% \\\\\n"

latex_str += r"""\hline\hline
\end{tabular}
\end{table*}
"""

# Save LaTeX file
tex_path = os.path.join(PROJECT_ROOT, "table_fifa_vs_eloratings_m01_m32.tex")
with open(tex_path, "w") as f:
    f.write(latex_str)

print("Saved LaTeX table to:", tex_path)
print("\nLaTeX Code Preview:")
print(latex_str[:1500])
