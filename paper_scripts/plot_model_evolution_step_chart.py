import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set matplotlib params for Adobe Illustrator & vector editing compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

PROJECT_ROOT = "/Users/rennocosta/matchdataset"
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
FINAL_FIGURES_DIR = os.path.join(PROJECT_ROOT, "final_figures")
CAMERAREADY_DIR = os.path.join(PROJECT_ROOT, "cameraready")

for d in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR]:
    os.makedirs(d, exist_ok=True)

# Progression of architectures from simplest to most complex
model_progression = [
    'FIFA SUM',
    'eloratings.net',
    '1-Elo Simple',
    '1-Elo Complete',
    '2-Elo Fast-Slow',
    '2-Elo (O+D)',
    '3-Elo Hybrid',
    '3-Elo Complete\n(BEST)',
    '4-Elo Multi-Scale'
]

x_indices = np.arange(len(model_progression))

# Data values from Optuna M32 ALL optimization
all_joint = [0.45360, 0.44629, 0.44805, 0.44292, 0.44804, 0.43772, 0.43687, 0.43680, 0.43717]
fast_esd  = [0.28147, 0.27685, 0.27779, 0.27469, 0.27777, 0.27096, 0.27065, 0.27055, 0.27074]
rps_fast  = [0.17428, 0.17067, 0.17145, 0.16911, 0.17145, 0.16768, 0.16724, 0.16721, 0.16735]
rps_slow  = [0.17213, 0.16943, 0.17026, 0.16823, 0.17027, 0.16676, 0.16622, 0.16625, 0.16643]
esd_score = [2.1439,  2.1238,  2.1267,  2.1116,  2.1265,  2.0657,  2.0682,  2.0669,  2.0678]

# Setup 2-panel step/line plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True, facecolor='white')
fig.patch.set_facecolor('white')

plt.subplots_adjust(top=0.92, bottom=0.12, hspace=0.18, left=0.08, right=0.92)

# --- Panel A: Joint Loss Evolution (ALL Joint & Fast+ESD) ---
ax1.set_facecolor('white')

# Plot step lines
ax1.step(x_indices, all_joint, where='mid', color='#9467bd', linewidth=2.5, label='ALL Joint Loss', linestyle='-')
ax1.plot(x_indices, all_joint, 'o', color='#9467bd', markersize=7, zorder=5)

ax1.step(x_indices, fast_esd, where='mid', color='#2ca02c', linewidth=2.5, label='Fast+ESD Combined Loss', linestyle='--')
ax1.plot(x_indices, fast_esd, 's', color='#2ca02c', markersize=6, zorder=5)

# Highlight Best Point (3-Elo Complete)
best_idx = 7
ax1.plot(best_idx, all_joint[best_idx], '*', color='#d62728', markersize=16, zorder=10, label='3-Elo Complete (BEST)')
ax1.plot(best_idx, fast_esd[best_idx], '*', color='#d62728', markersize=16, zorder=10)

# Vertical shade separating 1D models from 2D/3D Decoupled models
ax1.axvspan(-0.4, 4.4, color='#ffe6cc', alpha=0.3, zorder=1)
ax1.axvspan(4.4, 8.4, color='#d5e8d4', alpha=0.3, zorder=1)

ax1.text(2.0, 0.4525, "Scalar 1D Models\n(Outcome Only)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#d95f02')
ax1.text(6.4, 0.4525, "Decoupled Multi-Vector Models\n(Offense + Defense Style)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#276419')

# Annotate Best Point
ax1.annotate("BEST Performance\nALL Loss = 0.43680", xy=(best_idx, all_joint[best_idx]),
             xytext=(best_idx - 2.4, all_joint[best_idx] - 0.025),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=7),
             fontsize=9.5, fontweight='bold', color='#d62728',
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#d62728", lw=1.5))

ax1.set_ylabel("Joint Loss Value", fontsize=12, fontweight='bold', color='#1D3557', labelpad=8)
ax1.set_title("Panel A: Model Complexity Progression & Joint Loss Minimization (Model M32 Architecture)", fontsize=13, fontweight='bold', color='#1D3557', pad=12)
ax1.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10, prop={'weight': 'bold'})

# --- Panel B: Individual Metric Evolution (RPS Fast, RPS Slow, ESD Scoreline) ---
ax2.set_facecolor('white')

# Left Y-axis: RPS Scores
ax2.step(x_indices, rps_fast, where='mid', color='#1f77b4', linewidth=2.2, label='RPS Fast (Immediate)', linestyle='-')
ax2.plot(x_indices, rps_fast, 'o', color='#1f77b4', markersize=6, zorder=5)

ax2.step(x_indices, rps_slow, where='mid', color='#ff7f0e', linewidth=2.2, label='RPS Slow (6-Month)', linestyle='--')
ax2.plot(x_indices, rps_slow, '^', color='#ff7f0e', markersize=6, zorder=5)

ax2.set_ylabel("Ranked Probability Score (RPS)", fontsize=12, fontweight='bold', color='#1D3557', labelpad=8)
ax2.grid(True, linestyle=':', alpha=0.6, color='#cccccc')

# Right Y-axis: ESD Scoreline Distance
ax2_right = ax2.twinx()
ax2_right.step(x_indices, esd_score, where='mid', color='#e377c2', linewidth=2.2, label='ESD Scoreline Distance', linestyle=':')
ax2_right.plot(x_indices, esd_score, 'D', color='#e377c2', markersize=5.5, zorder=5)
ax2_right.set_ylabel("Scoreline Distance (ESD)", fontsize=12, fontweight='bold', color='#e377c2', labelpad=8)
ax2_right.grid(False)

# Combine legends for Panel B
lines_left, labels_left = ax2.get_legend_handles_labels()
lines_right, labels_right = ax2_right.get_legend_handles_labels()
ax2.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', prop={'weight': 'bold'})

ax2.set_title("Panel B: Individual Metric Evolution (RPS Fast, RPS Slow, and Scoreline ESD)", fontsize=13, fontweight='bold', color='#1D3557', pad=12)

# X-Axis Ticks & Labels
ax2.set_xticks(x_indices)
ax2.set_xticklabels(model_progression, fontsize=10.5, fontweight='bold', color='#1D3557')
ax2.set_xlabel("Model Architectural Progression (Increasing Complexity →)", fontsize=12, fontweight='bold', color='#1D3557', labelpad=10)

# Save figure in all target folders
out_name = "figure_model_evolution_step_chart"
for folder in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR, PROJECT_ROOT]:
    pdf_path = os.path.join(folder, f"{out_name}.pdf")
    png_path = os.path.join(folder, f"{out_name}.png")
    svg_path = os.path.join(folder, f"{out_name}.svg")
    
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {pdf_path}")

plt.close(fig)
