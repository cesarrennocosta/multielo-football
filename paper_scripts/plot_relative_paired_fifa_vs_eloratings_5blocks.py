import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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

# Load paired evaluation CSVs across all 32 models
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

# 5 Blocks / Subplots configuration: each paired with metric-specific CV and AIC
PANELS_CONFIG = [
    {
        'panel_letter': '(a)',
        'title': 'RPS Fast (Immediate)',
        'cv_col': 'CV_RPS_fast',
        'aic_col': 'AIC_fast',
        'color': '#1f77b4'
    },
    {
        'panel_letter': '(b)',
        'title': 'RPS Slow (6-Month)',
        'cv_col': 'CV_RPS_slow',
        'aic_col': 'AIC_slow',
        'color': '#ff7f0e'
    },
    {
        'panel_letter': '(c)',
        'title': 'ESD (Scoreline Distance)',
        'cv_col': 'CV_ESD_fast',
        'aic_col': 'AIC_esd',
        'color': '#e377c2'
    },
    {
        'panel_letter': '(d)',
        'title': 'Fast+ESD Combined Objective',
        'cv_col': 'CV_Fast+ESD',
        'aic_col': 'AIC_fastesd',
        'color': '#2ca02c'
    },
    {
        'panel_letter': '(e)',
        'title': 'Joint ALL Objective',
        'cv_col': 'CV_Joint_ALL',
        'aic_col': 'AIC_all',
        'color': '#9467bd'
    }
]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 5, figsize=(20, 6.5), facecolor='white')
fig.patch.set_facecolor('white')

plt.subplots_adjust(top=0.86, bottom=0.16, left=0.05, right=0.94, wspace=0.35)

x_left = 0.0
x_right = 1.0

# Render 5 blocks
for idx, p_cfg in enumerate(PANELS_CONFIG):
    ax = axes[idx]
    ax.set_facecolor('white')
    
    cv_c = p_cfg['cv_col']
    aic_c = p_cfg['aic_col']
    
    f_cv = df_fifa[cv_c].values
    e_cv = df_elo[cv_c].values
    
    f_aic = df_fifa[aic_c].values
    e_aic = df_elo[aic_c].values
    
    # Compute relative values (Ratio = Elo / FIFA, where FIFA = 1.0)
    rel_cv = e_cv / f_cv
    rel_aic = e_aic / f_aic
    
    # Left side: Single point for FIFA baseline at 1.0
    ax.plot(x_left, 1.0, 'o', color='#8c564b', markersize=10, zorder=6, label='FIFA SUM (Baseline = 1.0)')
    
    # Right side: 32 points for Eloratings.net relative CV values
    for m_i in range(n_models):
        r_cv_val = rel_cv[m_i]
        line_color = p_cfg['color'] if r_cv_val <= 1.0 else '#d62728'
        
        # Connect single FIFA point (0, 1.0) to (1.0, rel_cv)
        ax.plot([x_left, x_right], [1.0, r_cv_val], color=line_color, alpha=0.35, linewidth=1.2, zorder=2)
        ax.plot(x_right, r_cv_val, 'o', color=p_cfg['color'], markersize=4.5, alpha=0.8, zorder=4)

    # Plot metric-specific AIC relative values on Right Y-Axis (Dual Axes)
    ax_right = ax.twinx()
    ax_right.plot(x_left, 1.0, 'v', color='#d62728', markersize=9, zorder=6, label='FIFA AIC Baseline (1.0)')
    
    for m_i in range(n_models):
        r_aic_val = rel_aic[m_i]
        ax_right.plot([x_left, x_right], [1.0, r_aic_val], color='#d62728', linestyle=':', alpha=0.3, linewidth=1.0, zorder=1)
        ax_right.plot(x_right, r_aic_val, 'v', color='#d62728', markersize=4.0, alpha=0.7, zorder=4)

    # Calculate Mean Relative Reduction
    mean_rel_cv = np.mean(rel_cv)
    mean_rel_aic = np.mean(rel_aic)
    
    # Statistical tests
    t_stat_cv, p_val_cv = stats.ttest_rel(f_cv, e_cv)
    t_stat_aic, p_val_aic = stats.ttest_rel(f_aic, e_aic)
    
    # Mean ± SD Relative Point
    ax.errorbar(x_right + 0.08, mean_rel_cv, yerr=np.std(rel_cv), fmt='D', color=p_cfg['color'],
                ecolor=p_cfg['color'], elinewidth=1.5, capsize=4, markersize=7, zorder=5)
    
    ax_right.errorbar(x_right + 0.16, mean_rel_aic, yerr=np.std(rel_aic), fmt='D', color='#d62728',
                      ecolor='#d62728', elinewidth=1.5, capsize=4, markersize=7, zorder=5)

    # Panel Title & Y-Axes Labels
    ax.set_title(f"{p_cfg['panel_letter']} {p_cfg['title']}", fontsize=11, fontweight='bold', color='#1D3557', pad=10)
    
    if idx == 0:
        ax.set_ylabel('Relative 5-CV Loss (FIFA Baseline = 1.0)', fontsize=10.5, fontweight='bold', color='#1D3557')
    else:
        ax.set_ylabel('')
        
    if idx == 4:
        ax_right.set_ylabel('Relative Specific AIC (FIFA Baseline = 1.0)', fontsize=10.5, fontweight='bold', color='#d62728', labelpad=8)
    else:
        ax_right.set_ylabel('')
    
    # Format X-axis
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['FIFA SUM\n(Baseline = 1.0)', 'Eloratings.net\n(Relative values)'], fontsize=9.5, fontweight='bold', color='#1D3557')
    ax.set_xlim(-0.25, 1.30)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
    ax_right.grid(False)

    # Y-axis scaling around 1.0
    cv_min, cv_max = np.min(rel_cv), np.max(rel_cv)
    ax.set_ylim(min(0.95, cv_min - 0.02), max(1.02, cv_max + 0.02))
    
    aic_min, aic_max = np.min(rel_aic), np.max(rel_aic)
    ax_right.set_ylim(min(0.95, aic_min - 0.02), max(1.02, aic_max + 0.02))

    # Statistical significance text box
    p_cv_str = "p < 0.001***" if p_val_cv < 0.001 else f"p = {p_val_cv:.4f}"
    p_aic_str = "p < 0.001***" if p_val_aic < 0.001 else f"p = {p_val_aic:.4f}"
    
    ax.text(0.5, 0.95, f"Relative 5-CV: {mean_rel_cv:.4f} ({p_cv_str})\nRelative AIC: {mean_rel_aic:.4f} ({p_aic_str})",
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5, fontweight='bold',
            color='#d62728', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#d62728", lw=1.2, alpha=0.9))

# Main Title
fig.suptitle("Relative Evaluation Across 5 Metrics: Eloratings.net Relative Performance vs. FIFA SUM Baseline (=1.0)",
             fontsize=14, fontweight='bold', color='#1D3557', y=0.96)

# Save figure in all target folders
out_name = "figure_relative_paired_fifa_vs_eloratings_5blocks"
for folder in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR, PROJECT_ROOT]:
    pdf_path = os.path.join(folder, f"{out_name}.pdf")
    png_path = os.path.join(folder, f"{out_name}.png")
    svg_path = os.path.join(folder, f"{out_name}.svg")
    
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {pdf_path}")

plt.close(fig)
