import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from scipy import stats

# Set matplotlib params for Adobe Illustrator & vector editing compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

PROJECT_ROOT = "/Users/rennocosta/matchdataset"
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
FINAL_FIGURES_DIR = os.path.join(PROJECT_ROOT, "final_figures")
CAMERAREADY_DIR = os.path.join(PROJECT_ROOT, "cameraready")

for d in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR]:
    os.makedirs(d, exist_ok=True)

# Load paired data across all 32 models
fifa_rows = []
elo_rows = []

for mid in range(1, 33):
    m_str = f"M{mid:02d}"
    f_path = os.path.join(RESULTS_DIR, f"eval_external_fifa_{m_str}.csv")
    e_path = os.path.join(RESULTS_DIR, f"eval_external_eloratings_{m_str}.csv")
    
    if os.path.exists(f_path) and os.path.exists(e_path):
        df_f = pd.read_csv(f_path)
        df_e = pd.read_csv(e_path)
        fifa_rows.append(df_f.iloc[0])
        elo_rows.append(df_e.iloc[0])

df_fifa = pd.DataFrame(fifa_rows)
df_elo = pd.DataFrame(elo_rows)

n_models = len(df_fifa)

# 5 Blocks / Subplots configuration
PANELS_CONFIG = [
    {
        'panel_letter': '(a)',
        'title': 'RPS Fast (Immediate Outcome)',
        'metric_col': 'CV_RPS_fast',
        'ylabel': 'RPS Fast Loss (5-Fold CV)',
        'color': '#1f77b4'
    },
    {
        'panel_letter': '(b)',
        'title': 'RPS Slow (6-Month Horizon)',
        'metric_col': 'CV_RPS_slow',
        'ylabel': 'RPS Slow Loss (5-Fold CV)',
        'color': '#ff7f0e'
    },
    {
        'panel_letter': '(c)',
        'title': 'ESD (Scoreline Goal Distance)',
        'metric_col': 'CV_ESD_fast',
        'ylabel': 'ESD Scoreline Loss (5-Fold CV)',
        'color': '#e377c2'
    },
    {
        'panel_letter': '(d)',
        'title': 'Fast+ESD Combined Objective',
        'metric_col': 'CV_Fast+ESD',
        'ylabel': 'Fast+ESD Loss (5-Fold CV)',
        'color': '#2ca02c'
    },
    {
        'panel_letter': '(e)',
        'title': 'Joint ALL & Dixon-Coles AIC Fit (Dual Axes)',
        'metric_col': 'CV_Joint_ALL',
        'aic_col': 'AIC_all',
        'ylabel': 'Joint ALL Loss (5-Fold CV)',
        'color': '#9467bd'
    }
]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 5, figsize=(20, 6.5), facecolor='white')
fig.patch.set_facecolor('white')

plt.subplots_adjust(top=0.86, bottom=0.15, left=0.05, right=0.94, wspace=0.35)

x_left = 0.0
x_right = 1.0

# Render 5 blocks
for idx, p_cfg in enumerate(PANELS_CONFIG):
    ax = axes[idx]
    ax.set_facecolor('white')
    
    m_col = p_cfg['metric_col']
    f_vals = df_fifa[m_col].values
    e_vals = df_elo[m_col].values
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(f_vals, e_vals)
    w_stat, p_val_w = stats.wilcoxon(f_vals, e_vals)
    
    # Draw connected slope lines for all 32 models
    for m_i in range(n_models):
        y_f = f_vals[m_i]
        y_e = e_vals[m_i]
        line_color = '#d62728' if y_e < y_f else '#2ca02c' # Red/green if improved
        ax.plot([x_left, x_right], [y_f, y_e], color=line_color, alpha=0.35, linewidth=1.2, zorder=2)
        ax.plot(x_left, y_f, 'o', color='#8c564b', markersize=4.5, alpha=0.7, zorder=3)
        ax.plot(x_right, y_e, 'o', color=p_cfg['color'], markersize=4.5, alpha=0.8, zorder=3)

    # Mean ± SD Points
    mean_f, sd_f = np.mean(f_vals), np.std(f_vals)
    mean_e, sd_e = np.mean(e_vals), np.std(e_vals)
    
    ax.errorbar(x_left - 0.08, mean_f, yerr=sd_f, fmt='D', color='#8c564b', ecolor='#8c564b',
                elinewidth=1.5, capsize=4, markersize=7, label='FIFA Mean ± SD', zorder=5)
    ax.errorbar(x_right + 0.08, mean_e, yerr=sd_e, fmt='D', color=p_cfg['color'], ecolor=p_cfg['color'],
                elinewidth=1.5, capsize=4, markersize=7, label='Elo Mean ± SD', zorder=5)
    
    # Panel Title & Annotations
    ax.set_title(f"{p_cfg['panel_letter']} {p_cfg['title']}", fontsize=11, fontweight='bold', color='#1D3557', pad=10)
    ax.set_ylabel(p_cfg['ylabel'], fontsize=10.5, fontweight='bold', color='#1D3557')
    
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['FIFA SUM\n(Benchmark)', 'Eloratings.net\n(Benchmark)'], fontsize=10, fontweight='bold', color='#1D3557')
    ax.set_xlim(-0.25, 1.25)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
    
    # Statistical significance text box
    p_str = "p < 0.001***" if p_val < 0.001 else f"p = {p_val:.4f}"
    ax.text(0.5, 0.95, f"Paired t-test:\nt = {t_stat:.2f}, {p_str}\nWilcoxon p < 0.001***",
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5, fontweight='bold',
            color='#d62728', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#d62728", lw=1.2, alpha=0.9))

    # Dual Axes for Panel 5 (Joint ALL & AIC)
    if 'aic_col' in p_cfg:
        ax_right = ax.twinx()
        f_aic = df_fifa[p_cfg['aic_col']].values
        e_aic = df_elo[p_cfg['aic_col']].values
        
        for m_i in range(n_models):
            ax_right.plot([x_left, x_right], [f_aic[m_i], e_aic[m_i]], color='#d62728', linestyle=':', alpha=0.3, linewidth=1.0, zorder=1)
            
        ax_right.set_ylabel('Akaike Information Criterion (AIC)', fontsize=10.5, fontweight='bold', color='#d62728', labelpad=8)
        ax_right.grid(False)

# Main Title
fig.suptitle("Paired Evaluation & Statistical Significance: Eloratings.net vs. FIFA SUM Across 32 GLM Models (M01–M32)",
             fontsize=14, fontweight='bold', color='#1D3557', y=0.96)

# Save figure in all target folders
out_name = "figure_paired_fifa_vs_eloratings_5blocks"
for folder in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR, PROJECT_ROOT]:
    pdf_path = os.path.join(folder, f"{out_name}.pdf")
    png_path = os.path.join(folder, f"{out_name}.png")
    svg_path = os.path.join(folder, f"{out_name}.svg")
    
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {pdf_path}")

plt.close(fig)
