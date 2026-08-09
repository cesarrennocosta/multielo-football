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

PANELS_CONFIG = [
    {
        'panel_letter': '(a)',
        'title': 'RPS Fast',
        'cv_col': 'CV_RPS_fast',
        'aic_col': 'AIC_fast',
        'color': '#1f77b4'
    },
    {
        'panel_letter': '(b)',
        'title': 'RPS Slow',
        'cv_col': 'CV_RPS_slow',
        'aic_col': 'AIC_slow',
        'color': '#ff7f0e'
    },
    {
        'panel_letter': '(c)',
        'title': 'ESD Scoreline',
        'cv_col': 'CV_ESD_fast',
        'aic_col': 'AIC_esd',
        'color': '#e377c2'
    },
    {
        'panel_letter': '(d)',
        'title': 'Fast+ESD Combined',
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

# Landscape A4 Page Dimensions: Width = 11.69 inches, Height = 8.27 inches
fig, axes = plt.subplots(1, 5, figsize=(11.69, 8.27), facecolor='white')
fig.patch.set_facecolor('white')

plt.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.90, wspace=0.38)

x_left = 0.0
x_right = 1.0

# Render 5 side-by-side panels across Landscape A4 page
for idx, p_cfg in enumerate(PANELS_CONFIG):
    ax = axes[idx]
    ax.set_facecolor('white')
    
    cv_c = p_cfg['cv_col']
    aic_c = p_cfg['aic_col']
    
    f_cv = df_fifa[cv_c].values
    e_cv = df_elo[cv_c].values
    
    f_aic = df_fifa[aic_c].values
    e_aic = df_elo[aic_c].values
    
    # Compute relative values (FIFA Baseline = 1.0)
    rel_cv = e_cv / f_cv
    rel_aic = e_aic / f_aic
    
    # Left side: Single reference point for FIFA baseline at 1.0
    ax.plot(x_left, 1.0, 'o', color='#8c564b', markersize=8.5, zorder=6, label='FIFA Baseline (=1.0)')
    
    # Right side: 32 points for Eloratings.net relative CV values
    for m_i in range(n_models):
        r_cv_val = rel_cv[m_i]
        line_color = p_cfg['color'] if r_cv_val <= 1.0 else '#d62728'
        ax.plot([x_left, x_right], [1.0, r_cv_val], color=line_color, alpha=0.35, linewidth=1.1, zorder=2)
        ax.plot(x_right, r_cv_val, 'o', color=p_cfg['color'], markersize=4.5, alpha=0.85, zorder=4)

    # Dedicated Right Y-Axis for Specific AIC - Scaled BELOW CV scale to eliminate visual overlap completely
    ax_right = ax.twinx()
    ax_right.plot(x_left, 1.0, 'v', color='#d62728', markersize=8.5, zorder=6, label='FIFA AIC Baseline (=1.0)')
    
    for m_i in range(n_models):
        r_aic_val = rel_aic[m_i]
        ax_right.plot([x_left, x_right], [1.0, r_aic_val], color='#d62728', linestyle=':', alpha=0.35, linewidth=1.0, zorder=1)
        ax_right.plot(x_right, r_aic_val, 'v', color='#d62728', markersize=4.0, alpha=0.75, zorder=4)

    # Mean Relative Values & Statistical Tests
    mean_rel_cv = np.mean(rel_cv)
    mean_rel_aic = np.mean(rel_aic)
    
    t_stat_cv, p_val_cv = stats.ttest_rel(f_cv, e_cv)
    t_stat_aic, p_val_aic = stats.ttest_rel(f_aic, e_aic)
    
    # Mean ± SD Error Bar Points
    ax.errorbar(x_right + 0.08, mean_rel_cv, yerr=np.std(rel_cv), fmt='D', color=p_cfg['color'],
                ecolor=p_cfg['color'], elinewidth=1.5, capsize=3.5, markersize=6.5, zorder=5)
    
    ax_right.errorbar(x_right + 0.16, mean_rel_aic, yerr=np.std(rel_aic), fmt='D', color='#d62728',
                      ecolor='#d62728', elinewidth=1.5, capsize=3.5, markersize=6.5, zorder=5)

    # Maximize Graph View Scale & Offset AIC below CV
    cv_min, cv_max = np.min(rel_cv), np.max(rel_cv)
    cv_range = max(0.003, cv_max - cv_min)
    ax.set_ylim(cv_min - 0.15 * cv_range, cv_max + 0.35 * cv_range)
    
    aic_min, aic_max = np.min(rel_aic), np.max(rel_aic)
    aic_range = max(0.003, aic_max - aic_min)
    # Offset AIC scale significantly below CV lines
    ax_right.set_ylim(aic_min - 0.65 * aic_range, aic_max + 0.10 * aic_range)

    # Disable scalar offset formatting (+1)
    ax.ticklabel_format(useOffset=False)
    ax_right.ticklabel_format(useOffset=False)

    # Subplot Caption / Title (14-points Arial font)
    ax.set_title(f"{p_cfg['panel_letter']} {p_cfg['title']}", fontsize=14, fontweight='bold', color='#1D3557', pad=12, fontfamily='Arial')
    
    # 18-points Arial Axis Legends (ONLY ON OUTERMOST PANELS)
    if idx == 0:
        ax.set_ylabel('Relative 5-CV Loss', fontsize=18, fontweight='bold', color='#1D3557', fontfamily='Arial', labelpad=12)
    else:
        ax.set_ylabel('')
        
    if idx == 4:
        ax_right.set_ylabel('Relative Specific AIC', fontsize=18, fontweight='bold', color='#d62728', fontfamily='Arial', labelpad=12)
    else:
        ax_right.set_ylabel('')
    
    # Format X-axis (12-points Arial)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['FIFA Baseline\n(=1.0)', 'Eloratings.net\n(Relative)'], fontsize=12, fontweight='bold', color='#1D3557', fontfamily='Arial')
    ax.set_xlim(-0.22, 1.30)
    
    # Tick label sizes (11-points Arial)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax_right.tick_params(axis='both', which='major', labelsize=11)
    
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
    ax_right.grid(False)

    # Statistical Significance Text Box (Arial font)
    p_cv_str = "p < 0.001***" if p_val_cv < 0.001 else f"p = {p_val_cv:.4f}"
    p_aic_str = "p < 0.001***" if p_val_aic < 0.001 else f"p = {p_val_aic:.4f}"
    
    ax.text(0.5, 0.88, f"Rel 5-CV: {mean_rel_cv:.4f} ({p_cv_str})\nRel AIC: {mean_rel_aic:.4f} ({p_aic_str})",
            transform=ax.transAxes, ha='center', va='top', fontsize=8.0, fontweight='bold', fontfamily='Arial',
            color='#d62728', bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#d62728", lw=1.2, alpha=0.9))

# Main Title (18-points Arial)
fig.suptitle("Relative Evaluation Across 5 Objectives: Eloratings.net vs. FIFA SUM Baseline (=1.0) Across 32 Models (M01–M32)",
             fontsize=18, fontweight='bold', color='#1D3557', y=0.96, fontfamily='Arial')

# Save figure in all target folders
out_name = "figure_relative_paired_fifa_vs_eloratings_a4"
for folder in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR, PROJECT_ROOT]:
    pdf_path = os.path.join(folder, f"{out_name}.pdf")
    png_path = os.path.join(folder, f"{out_name}.png")
    svg_path = os.path.join(folder, f"{out_name}.svg")
    
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {pdf_path}")

plt.close(fig)
