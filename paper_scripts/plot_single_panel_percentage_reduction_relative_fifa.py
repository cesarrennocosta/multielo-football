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
sys.path.insert(0, PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
FINAL_FIGURES_DIR = os.path.join(PROJECT_ROOT, "final_figures")
CAMERAREADY_DIR = os.path.join(PROJECT_ROOT, "cameraready")

for d in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR]:
    os.makedirs(d, exist_ok=True)

# 1. Load Data from update_heatmap_m32_pure
from update_heatmap_m32_pure import load_m32_data, TAGS, TAG_LABELS

d_store = load_m32_data()

fifa_idx = TAGS.index('fifa')
fifa_vals = {
    'RPS_fast': float(d_store['RPS_fast']['cv_mean'][fifa_idx, 0]),
    'RPS_slow': float(d_store['RPS_slow']['cv_mean'][fifa_idx, 0]),
    'ESD_fast': float(d_store['ESD_fast']['cv_mean'][fifa_idx, 0]),
    'Fast+ESD': float(d_store['fastesd']['cv_mean'][fifa_idx, 0]),
    'Joint_ALL': float(d_store['all']['cv_mean'][fifa_idx, 0]),
    'AIC': float(d_store['all']['aic'][fifa_idx, 0])
}

systems_data = []
for i, tag in enumerate(TAGS):
    label = TAG_LABELS[tag].split('(')[0].strip()
    if label == "2-Elo":
        label = "2-Elo (O+D) Pure"
    elif label == "FIFA SUM Ref":
        label = "FIFA SUM"
    elif label == "Eloratings.net Ref":
        label = "Eloratings.net"
        
    rf = float(d_store['RPS_fast']['cv_mean'][i, 0])
    rs = float(d_store['RPS_slow']['cv_mean'][i, 0])
    ef = float(d_store['ESD_fast']['cv_mean'][i, 0])
    fesd = float(d_store['fastesd']['cv_mean'][i, 0])
    all_loss = float(d_store['all']['cv_mean'][i, 0])
    aic_val = float(d_store['all']['aic'][i, 0])
    
    systems_data.append({
        'tag': tag,
        'label': label,
        'RPS_fast_pct': (rf / fifa_vals['RPS_fast']) * 100.0,
        'RPS_slow_pct': (rs / fifa_vals['RPS_slow']) * 100.0,
        'ESD_fast_pct': (ef / fifa_vals['ESD_fast']) * 100.0,
        'Fast+ESD_pct': (fesd / fifa_vals['Fast+ESD']) * 100.0,
        'Joint_ALL_pct': (all_loss / fifa_vals['Joint_ALL']) * 100.0,
        'AIC_pct': (aic_val / fifa_vals['AIC']) * 100.0,
        'RPS_fast_raw': rf,
        'RPS_slow_raw': rs,
        'ESD_fast_raw': ef,
        'Fast+ESD_raw': fesd,
        'Joint_ALL_raw': all_loss,
        'AIC_raw': aic_val
    })

df = pd.DataFrame(systems_data)

# Sort models by Joint ALL Loss in DESCENDING Order (Highest/Worst at x=0 -> Lowest/Best at x=8)
df = df.sort_values('Joint_ALL_raw', ascending=False).reset_index(drop=True)

# Compute raw min/max dictionary
metrics_raw = ['Joint_ALL', 'Fast+ESD', 'ESD_fast', 'RPS_fast', 'RPS_slow', 'AIC']
min_max_dict = {}
for m in metrics_raw:
    min_max_dict[m] = {'min': np.min(df[f"{m}_raw"].values), 'max': np.max(df[f"{m}_raw"].values)}

# 2. Build Single-Panel Percentage Relative Plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(15, 8.5), facecolor='white')
ax.set_facecolor('white')

plt.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.92)

x_indices = np.arange(len(df))
model_labels = df['label'].tolist()

# Metric configurations relative to FIFA SUM reference
METRICS_CONFIG = [
    {
        'col': 'Joint_ALL_pct',
        'label': f"Joint ALL (5-Fold CV)  [min: {min_max_dict['Joint_ALL']['min']:.5f}, ref: {fifa_vals['Joint_ALL']:.5f}]",
        'color': '#9467bd', 'marker': 'p', 'linestyle': '-', 'linewidth': 2.8, 'markersize': 8.5
    },
    {
        'col': 'AIC_pct',
        'label': f"AIC (Joint Likelihood Fit)  [min: {min_max_dict['AIC']['min']:.0f}, ref: {fifa_vals['AIC']:.0f}]",
        'color': '#d62728', 'marker': 'v', 'linestyle': '-.', 'linewidth': 2.5, 'markersize': 8.0
    },
    {
        'col': 'Fast+ESD_pct',
        'label': f"Fast+ESD Combined (5-Fold CV)  [min: {min_max_dict['Fast+ESD']['min']:.5f}, ref: {fifa_vals['Fast+ESD']:.5f}]",
        'color': '#2ca02c', 'marker': 's', 'linestyle': '--', 'linewidth': 2.4, 'markersize': 7.5
    },
    {
        'col': 'ESD_fast_pct',
        'label': f"ESD Scoreline (5-Fold CV)  [min: {min_max_dict['ESD_fast']['min']:.4f}, ref: {fifa_vals['ESD_fast']:.4f}]",
        'color': '#e377c2', 'marker': 'D', 'linestyle': ':', 'linewidth': 2.2, 'markersize': 6.5
    },
    {
        'col': 'RPS_fast_pct',
        'label': f"RPS Fast (5-Fold CV)  [min: {min_max_dict['RPS_fast']['min']:.5f}, ref: {fifa_vals['RPS_fast']:.5f}]",
        'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.0, 'markersize': 6.5
    },
    {
        'col': 'RPS_slow_pct',
        'label': f"RPS Slow (5-Fold CV)  [min: {min_max_dict['RPS_slow']['min']:.5f}, ref: {fifa_vals['RPS_slow']:.5f}]",
        'color': '#ff7f0e', 'marker': '^', 'linestyle': '-.', 'linewidth': 2.0, 'markersize': 6.5
    }
]

# Plot each metric line as % relative to FIFA SUM reference
for m_cfg in METRICS_CONFIG:
    y_pct = df[m_cfg['col']].values
    ax.plot(x_indices, y_pct, color=m_cfg['color'], marker=m_cfg['marker'],
            linestyle=m_cfg['linestyle'], linewidth=m_cfg['linewidth'],
            markersize=m_cfg['markersize'], label=m_cfg['label'], zorder=4)

# Highlight Best Point (3-Elo Complete on far right x=8)
best_x = len(df) - 1
best_y = df.loc[best_x, 'Joint_ALL_pct']

ax.plot(best_x, best_y, '*', color='#d62728', markersize=20, zorder=10, label='3-Elo Complete (BEST)')

# Annotate Best Model Point on Far Right
ax.annotate(f"BEST 5-Fold CV Performance\n3-Elo Complete ({best_y:.2f}% of FIFA SUM)", xy=(best_x, best_y),
             xytext=(best_x - 3.2, best_y + 0.9),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=7),
             fontsize=10.5, fontweight='bold', color='#d62728',
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#d62728", lw=1.5))

# Reference Line at 100% (FIFA SUM)
ax.axhline(100.0, color='#8c564b', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
ax.text(0.1, 100.15, "100% FIFA SUM Baseline", fontsize=9.5, fontweight='bold', color='#8c564b')

# Formatting Y-Axis & X-Axis
ax.set_ylim(95.2, 100.8)
ax.set_ylabel("Loss & Information Divergence (% Relative to FIFA SUM Reference)", fontsize=12, fontweight='bold', color='#1D3557', labelpad=10)

# Format Y-axis ticks with percentage signs
ax.set_yticks(np.arange(95.5, 101.0, 0.5))
ax.set_yticklabels([f"{y:.1f}%" for y in np.arange(95.5, 101.0, 0.5)], fontsize=10, fontweight='bold', color='#333333')

ax.set_xticks(x_indices)
ax.set_xticklabels(model_labels, rotation=25, ha='right', fontsize=11, fontweight='bold', color='#1D3557')
ax.set_xlabel("Rating System Architectures (Sorted from Highest/Worst to Lowest/Best Joint ALL CV Loss →)", fontsize=12, fontweight='bold', color='#1D3557', labelpad=10)

ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', prop={'weight': 'bold', 'size': 9.5})

ax.set_title("Relative Percentage Loss & Information Reduction Across 9 Rating Architectures (Model M32)", fontsize=13.5, fontweight='bold', color='#1D3557', pad=14)

# Save figure to all target folders
out_name = "figure_single_panel_percentage_reduction_relative_fifa"
for folder in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR, PROJECT_ROOT]:
    pdf_path = os.path.join(folder, f"{out_name}.pdf")
    png_path = os.path.join(folder, f"{out_name}.png")
    svg_path = os.path.join(folder, f"{out_name}.svg")
    
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {pdf_path}")

plt.close(fig)
