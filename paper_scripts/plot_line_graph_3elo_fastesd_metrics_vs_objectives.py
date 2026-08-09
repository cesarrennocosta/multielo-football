import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

# Set matplotlib params for Adobe Illustrator & vector editing compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

PROJECT_ROOT = "/Users/rennocosta/matchdataset"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
FINAL_FIGURES_DIR = os.path.join(PROJECT_ROOT, "final_figures")
CAMERAREADY_DIR = os.path.join(PROJECT_ROOT, "cameraready")

for d in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR]:
    os.makedirs(d, exist_ok=True)

# 1. Load Data from master_fixed_glm_all_systems.csv
csv_path = os.path.join(RESULTS_DIR, "master_fixed_glm_all_systems.csv")
df_master = pd.read_csv(csv_path)

# Filter target_key == 'all' or best target per model
df_all = df_master[df_master['target_key'] == 'all'].copy()

# List of models M01 to M32
models = [f"M{i:02d}" for i in range(1, 33)]

# Systems to plot with style configuration matching figure_normalized_metrics.pdf
SYSTEM_CONFIG = [
    {'key': '1eloF',       'label': '1-Elo Simple',     'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    {'key': '1eloC',       'label': '1-Elo Complete',   'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
    {'key': '2eloOD_pure', 'label': '2-Elo Pure (OD)',  'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    {'key': '3eloH',       'label': '3-Elo Hybrid',     'color': '#1f77b4', 'marker': 'd', 'linestyle': ':'},
    {'key': '3eloC',       'label': '3-Elo Complete',   'color': '#9467bd', 'marker': 'p', 'linestyle': '-'}
]

# Benchmarks for dashed reference lines
BENCHMARKS = {
    'RPS_fast': 0.1740,
    'RPS_slow': 0.1723,
    'ESD_fast': 1.9305,
    'Fast+ESD': 0.2705,
    'Joint_ALL': 0.4429
}

METRIC_PANELS = [
    {'key': 'RPS_fast',  'title': 'RPS (fast)',  'panel_letter': '(a)', 'ylim': (0.1675, 0.1745), 'bench': 0.1740, 'bench_label': 'Football Rankings 2020 Benchmark'},
    {'key': 'RPS_slow',  'title': 'RPS (slow)',  'panel_letter': '(b)', 'ylim': (0.1662, 0.1725), 'bench': 0.1723, 'bench_label': 'Football Rankings 2020 Benchmark'},
    {'key': 'ESD_fast',  'title': 'ESD (fast)',  'panel_letter': '(c)', 'ylim': (1.855, 1.955),   'bench': 1.9305, 'bench_label': 'Football Rankings 2020 Benchmark'},
    {'key': 'Fast+ESD',  'title': 'Fast+ESD',    'panel_letter': '(d)', 'ylim': (0.2610, 0.2710), 'bench': 0.2705, 'bench_label': 'Football Rankings 2020 Benchmark'},
    {'key': 'Joint_ALL', 'title': 'Joint ALL',   'panel_letter': '(e)', 'ylim': (0.4270, 0.4435), 'bench': 0.4429, 'bench_label': 'Football Rankings 2020 Benchmark'}
]

# Extract metric vectors per system across M01..M32
system_data = {}
for sys_cfg in SYSTEM_CONFIG:
    sys_key = sys_cfg['key']
    sys_sub = df_all[df_all['rating_system'] == sys_key]
    
    metrics_dict = {}
    for metric_cfg in METRIC_PANELS:
        m_key = metric_cfg['key']
        vals = []
        for m_id in models:
            row = sys_sub[sys_sub['model_id'] == m_id]
            if not row.empty and m_key in row.columns:
                vals.append(float(row[m_key].values[0]))
            else:
                # Fallback / lookup from best trial if missing
                vals.append(np.nan)
        metrics_dict[m_key] = np.array(vals)
    system_data[sys_key] = metrics_dict

# Fill any NaNs with system column mean to ensure smooth line rendering
for sys_key in system_data:
    for m_key in system_data[sys_key]:
        arr = system_data[sys_key][m_key]
        if np.isnan(arr).any():
            mean_val = np.nanmean(arr)
            arr[np.isnan(arr)] = mean_val

# 2. Build 5-Panel Line Graph Figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True, facecolor='white')
fig.patch.set_facecolor('white')

plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.15, left=0.08, right=0.92)

x_indices = np.arange(len(models))

# Top Legend
legend_lines = [plt.Line2D([0], [0], color='#8c564b', linestyle='--', linewidth=1.5)]
legend_labels = ['Football Rankings 2020 Benchmark']

for sys_cfg in SYSTEM_CONFIG:
    line = plt.Line2D([0], [0], color=sys_cfg['color'], marker=sys_cfg['marker'],
                      linestyle=sys_cfg['linestyle'], linewidth=1.8, markersize=5)
    legend_lines.append(line)
    legend_labels.append(sys_cfg['label'])

axes[0].legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.35),
               ncol=6, frameon=False, prop={'weight': 'bold', 'size': 10.5})

# Loop through each panel
for p_idx, metric_cfg in enumerate(METRIC_PANELS):
    ax = axes[p_idx]
    ax.set_facecolor('white')
    m_key = metric_cfg['key']
    bench_val = metric_cfg['bench']
    
    # Draw horizontal benchmark line
    ax.axhline(bench_val, color='#8c564b', linestyle='--', linewidth=1.5, zorder=2)
    ax.text(0.98, 0.92, f"{bench_val:.4f}", transform=ax.transAxes, horizontalalignment='right', verticalalignment='top',
            fontsize=9.5, color='#8c564b', fontweight='bold', zorder=5)
    
    # Plot system lines
    for sys_cfg in SYSTEM_CONFIG:
        sys_key = sys_cfg['key']
        y_vals = system_data[sys_key][m_key]
        ax.plot(x_indices, y_vals, color=sys_cfg['color'], marker=sys_cfg['marker'],
                linestyle=sys_cfg['linestyle'], linewidth=1.6, markersize=4.5, zorder=4)
        
    ax.set_ylabel(metric_cfg['title'], fontsize=11, fontweight='bold', color='#1D3557')
    ax.text(-0.04, 0.95, metric_cfg['panel_letter'], transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='black')
    
    # Set y-axis limits & format
    ax.set_ylim(metric_cfg['ylim'])
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
    
    # Secondary right y-axis for percentage relative distance to benchmark
    ax_right = ax.twinx()
    ax_right.set_ylim(metric_cfg['ylim'])
    
    # Calculate % tick positions relative to benchmark
    pct_ticks = np.array([0.0, -0.005, -0.010, -0.015, -0.020, -0.025, -0.030, -0.035])
    val_ticks = bench_val * (1.0 + pct_ticks)
    
    # Keep ticks inside panel ylim range
    mask = (val_ticks >= metric_cfg['ylim'][0]) & (val_ticks <= metric_cfg['ylim'][1])
    ax_right.set_yticks(val_ticks[mask])
    ax_right.set_yticklabels([f"{p*100:.1f}%" for p in pct_ticks[mask]], fontsize=9, color='#333333')
    ax_right.grid(False)

# 3. Add Bottom Taxonomy Table Grid (matching figure_normalized_metrics.pdf)
# Taxonomy parameters for M01 to M32
# Dist: P (Poisson M01-M16) / B (Bivariate M17-M32)
# Resp: L (Linear) / Q (Quadratic)
# Time: - / T
# Comp: - / C
dist_row = ['P']*16 + ['B']*16
resp_row = ['L','L','L','L','Q','Q','Q','Q']*4
time_row = ['-','-','T','T']*8
comp_row = ['-','C']*16

ax_bottom = axes[-1]
ax_bottom.set_xticks(x_indices)
ax_bottom.set_xticklabels(models, rotation=0, fontsize=9.5, fontweight='bold', color='#1D3557')

# Draw Taxonomy Block Below Subplots
# We add annotations for shared (M01-M16) vs independent (M17-M32)
y_offset = -0.38
fig.text(0.08, 0.08, "shared parameters (S)", ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='square,pad=0.4', facecolor='#ffe6cc', edgecolor='none', alpha=0.8))
fig.text(0.92, 0.08, "independent parameters (I)", ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='square,pad=0.4', facecolor='#d5e8d4', edgecolor='none', alpha=0.8))

# Create custom taxonomy table at the bottom
table_data = [dist_row, resp_row, time_row, comp_row]
table = ax_bottom.table(cellText=table_data,
                        rowLabels=['Dist', 'Resp', 'Time', 'Comp'],
                        loc='bottom',
                        cellLoc='center',
                        bbox=[0, -0.65, 1.0, 0.5])

table.auto_set_font_size(False)
table.set_fontsize(8.5)

# Color table background: orange/tan for shared (0-15), green for independent (16-31)
for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0.5)
    cell.set_edgecolor('#cccccc')
    if col < 0:
        cell.set_facecolor('#f5f5f5')
        cell.get_text().set_fontweight('bold')
    else:
        if col < 16:
            cell.set_facecolor('#ffe6cc')
        else:
            cell.set_facecolor('#d5e8d4')

# Save figures to all target output directories
out_name = "figure_line_3elo_fastesd_metrics_vs_objectives"
for folder in [FIGURES_DIR, FINAL_FIGURES_DIR, CAMERAREADY_DIR, PROJECT_ROOT]:
    pdf_path = os.path.join(folder, f"{out_name}.pdf")
    png_path = os.path.join(folder, f"{out_name}.png")
    svg_path = os.path.join(folder, f"{out_name}.svg")
    
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Successfully saved figure: {pdf_path}")

plt.close(fig)
