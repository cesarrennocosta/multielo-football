import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

# Set matplotlib params for Adobe Illustrator & vector editing compatibility with Arial Font
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

PROJECT_ROOT = "/Users/rennocosta/matchdataset"
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
FINAL_FIGURES_DIR = os.path.join(PROJECT_ROOT, "final_figures")
CAMERAREADY_DIR = os.path.join(PROJECT_ROOT, "cameraready")

for d in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR]:
    os.makedirs(d, exist_ok=True)

# Load audited paired evaluation CSVs across all 32 models
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

# Metric details for 5 columns
METRICS_CONFIG = [
    {
        'col_title': 'RPS Fast',
        'cv_col': 'CV_RPS_fast',
        'aic_col': 'AIC_fast',
        'color': '#1f77b4'
    },
    {
        'col_title': 'RPS Slow',
        'cv_col': 'CV_RPS_slow',
        'aic_col': 'AIC_slow',
        'color': '#ff7f0e'
    },
    {
        'col_title': 'ESD Scoreline',
        'cv_col': 'CV_ESD_fast',
        'aic_col': 'AIC_esd',
        'color': '#e377c2'
    },
    {
        'col_title': 'Fast+ESD Combined',
        'cv_col': 'CV_Fast+ESD',
        'aic_col': 'AIC_fastesd',
        'color': '#2ca02c'
    },
    {
        'col_title': 'Joint ALL Objective',
        'cv_col': 'CV_Joint_ALL',
        'aic_col': 'AIC_all',
        'color': '#9467bd'
    }
]

# Grid of 2 rows x 5 columns on Landscape A4 Page (11.69 in x 8.27 in)
fig, axes = plt.subplots(2, 5, figsize=(11.69, 8.27), facecolor='white')
fig.patch.set_facecolor('white')

plt.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.94, wspace=0.35, hspace=0.36)

x_left = 0.0
x_right = 1.0

# ----------------------------------------------------
# ROW 1: 5-Fold Cross Validation Loss (Relative to FIFA = 1.0)
# ----------------------------------------------------
for col_idx, m_cfg in enumerate(METRICS_CONFIG):
    ax = axes[0, col_idx]
    ax.set_facecolor('white')
    
    cv_c = m_cfg['cv_col']
    f_cv = df_fifa[cv_c].values
    e_cv = df_elo[cv_c].values
    
    rel_cv = e_cv / f_cv
    mean_rel_cv = np.mean(rel_cv)
    t_stat_cv, p_val_cv = stats.ttest_rel(f_cv, e_cv)
    
    # Left: FIFA Single Baseline Point at 1.0
    ax.plot(x_left, 1.0, 'o', color='#8c564b', markersize=8.0, zorder=6)
    
    # Right: 32 Eloratings points
    for m_i in range(n_models):
        r_val = rel_cv[m_i]
        line_col = m_cfg['color'] if r_val <= 1.0 else '#d62728'
        ax.plot([x_left, x_right], [1.0, r_val], color=line_col, alpha=0.4, linewidth=1.1, zorder=2)
        ax.plot(x_right, r_val, 'o', color=m_cfg['color'], markersize=4.5, alpha=0.85, zorder=4)

    # Mean ± SD Error Bar
    ax.errorbar(x_right + 0.08, mean_rel_cv, yerr=np.std(rel_cv), fmt='D', color=m_cfg['color'],
                ecolor=m_cfg['color'], elinewidth=1.5, capsize=3.5, markersize=6.0, zorder=5)

    # Scaling & Formatting
    cv_min, cv_max = np.min(rel_cv), np.max(rel_cv)
    cv_range = max(0.003, cv_max - cv_min)
    ax.set_ylim(cv_min - 0.15 * cv_range, cv_max + 0.25 * cv_range)
    ax.ticklabel_format(useOffset=False)

    # Subplot Title (13pt Arial)
    panel_letter = chr(97 + col_idx) # (a) to (e)
    ax.set_title(f"({panel_letter}) {m_cfg['col_title']}", fontsize=12.5, fontweight='bold', color='#1D3557', pad=6, fontfamily='Arial')

    # Y-Axis Legend (18pt Arial on Column 0)
    if col_idx == 0:
        ax.set_ylabel('Relative 5-CV Loss', fontsize=18, fontweight='bold', color='#1D3557', fontfamily='Arial', labelpad=10)
    else:
        ax.set_ylabel('')

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['FIFA Baseline\n(=1.0)', 'Eloratings.net\n(Relative)'], fontsize=9.5, fontweight='bold', color='#1D3557', fontfamily='Arial')
    ax.set_xlim(-0.20, 1.25)
    ax.tick_params(axis='both', which='major', labelsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')

    # Statistical significance box
    p_cv_str = "p < 0.001***" if p_val_cv < 0.001 else f"p = {p_val_cv:.4f}"
    ax.text(0.5, 0.93, f"Rel 5-CV: {mean_rel_cv:.4f}\n({p_cv_str})",
            transform=ax.transAxes, ha='center', va='top', fontsize=8.0, fontweight='bold', fontfamily='Arial',
            color='#1D3557', bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#1D3557", lw=1.0, alpha=0.9))

# ----------------------------------------------------
# ROW 2: Metric-Specific Dixon-Coles AIC (Relative to FIFA = 1.0)
# ----------------------------------------------------
for col_idx, m_cfg in enumerate(METRICS_CONFIG):
    ax = axes[1, col_idx]
    ax.set_facecolor('white')
    
    aic_c = m_cfg['aic_col']
    f_aic = df_fifa[aic_c].values
    e_aic = df_elo[aic_c].values
    
    rel_aic = e_aic / f_aic
    mean_rel_aic = np.mean(rel_aic)
    t_stat_aic, p_val_aic = stats.ttest_rel(f_aic, e_aic)
    
    # Left: FIFA Single Baseline Point at 1.0
    ax.plot(x_left, 1.0, 'v', color='#d62728', markersize=8.0, zorder=6)
    
    # Right: 32 Eloratings points for AIC
    for m_i in range(n_models):
        r_val = rel_aic[m_i]
        ax.plot([x_left, x_right], [1.0, r_val], color='#d62728', linestyle=':', alpha=0.4, linewidth=1.1, zorder=2)
        ax.plot(x_right, r_val, 'v', color='#d62728', markersize=4.5, alpha=0.85, zorder=4)

    # Mean ± SD Error Bar
    ax.errorbar(x_right + 0.08, mean_rel_aic, yerr=np.std(rel_aic), fmt='D', color='#d62728',
                ecolor='#d62728', elinewidth=1.5, capsize=3.5, markersize=6.0, zorder=5)

    # Scaling & Formatting
    aic_min, aic_max = np.min(rel_aic), np.max(rel_aic)
    aic_range = max(0.003, aic_max - aic_min)
    ax.set_ylim(aic_min - 0.15 * aic_range, aic_max + 0.25 * aic_range)
    ax.ticklabel_format(useOffset=False)

    # Subplot Title (13pt Arial)
    panel_letter = chr(102 + col_idx) # (f) to (j)
    ax.set_title(f"({panel_letter}) AIC ({m_cfg['col_title']})", fontsize=12.5, fontweight='bold', color='#d62728', pad=6, fontfamily='Arial')
    
    # Y-Axis Legend (18pt Arial on Column 0)
    if col_idx == 0:
        ax.set_ylabel('Relative Specific AIC', fontsize=18, fontweight='bold', color='#d62728', fontfamily='Arial', labelpad=10)
    else:
        ax.set_ylabel('')

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['FIFA Baseline\n(=1.0)', 'Eloratings.net\n(Relative)'], fontsize=10, fontweight='bold', color='#1D3557', fontfamily='Arial')
    ax.set_xlim(-0.20, 1.25)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')

    # Statistical significance box
    p_aic_str = "p < 0.001***" if p_val_aic < 0.001 else f"p = {p_val_aic:.4f}"
    ax.text(0.5, 0.93, f"Rel AIC: {mean_rel_aic:.4f}\n({p_aic_str})",
            transform=ax.transAxes, ha='center', va='top', fontsize=8.0, fontweight='bold', fontfamily='Arial',
            color='#d62728', bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#d62728", lw=1.0, alpha=0.9))

# Main Title (18pt Arial)
fig.suptitle("Audited Relative Evaluation Across 5 Objectives: Eloratings.net vs. FIFA SUM Baseline (=1.0) Across 32 Models (M01–M32)",
             fontsize=18, fontweight='bold', color='#1D3557', y=0.965, fontfamily='Arial')

# Save figure in all target folders
out_name = "figure_grid_2x5_relative_fifa_vs_eloratings_a4"
for folder in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR, PROJECT_ROOT]:
    pdf_path = os.path.join(folder, f"{out_name}.pdf")
    png_path = os.path.join(folder, f"{out_name}.png")
    svg_path = os.path.join(folder, f"{out_name}.svg")
    
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {pdf_path}")

plt.close(fig)
