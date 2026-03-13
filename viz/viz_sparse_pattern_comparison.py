import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================

# Plot styling
plt.style.use('default')
plt.rcParams['font.size'] = 14

# Output configuration
OUTPUT_CONFIG = {
    'figs_folder': './figs/sparse_pattern_comparison',
    'file_prefix': 'HA_sparse_vs_max_comparison',
    'dpi': 300,
    'format': 'pdf'
}

# Colors for the two methods
METHOD_COLORS = {
    'HA_sparse_prefill': '#2196F3',  # Blue
    'HA_flatten_max_prefill': '#FF9800'  # Orange
}

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_sparse_pattern_data():
    """Load the sparse pattern comparison data."""
    try:
        data = np.load('sparse_compare.npy', allow_pickle=True).item()
        return data
    except FileNotFoundError:
        print("Error: sparse_compare.npy file not found!")
        return None

def extract_setting_label(item, method_type):
    """Extract a readable label for the parameter setting."""
    if method_type == 'sparse':
        start_val = item.get('starts_val', item.get('stats_val', 0))
        sparse_ratio = item.get('sparse_ratio', item.get('sparse_ratio', 0))
        return f"({start_val}, {sparse_ratio})"
    else:  # max method
        start_val = item.get('stats_val', item.get('starts_val', 0))
        sparse_ratio = item.get('sparse_ratio', 0)
        return f"({start_val}, {sparse_ratio})"

def prepare_comparison_data(data):
    """Prepare data for visualization comparison."""
    if not data:
        return None, None, None

    # Extract sparse and max method data
    sparse_data = data.get('HA_sparse_prefill', [])
    max_data = data.get('HA_flatten_max_prefill', [])

    if not sparse_data or not max_data:
        print("Error: Missing data for either sparse or max method!")
        return None, None, None

    # Ensure we have matching settings
    settings = []
    sparse_stats = []
    max_stats = []

    for i in range(min(len(sparse_data), len(max_data))):
        # Extract setting labels
        setting_label = extract_setting_label(sparse_data[i], 'sparse')
        settings.append(setting_label)

        # Extract stats
        sparse_stat = sparse_data[i]['stats'][0] if sparse_data[i]['stats'] else None
        max_stat = max_data[i]['stats'][0] if max_data[i]['stats'] else None

        sparse_stats.append(sparse_stat)
        max_stats.append(max_stat)

    return settings, sparse_stats, max_stats

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def create_combined_stacked_plot(settings, sparse_stats, max_stats):
    """Create a combined stacked plot showing compute (1 layer), compute (36 layers), and prepare times."""

    # Extract data for plotting
    prepare_times_sparse = [stat['prepare_mean_ms'] for stat in sparse_stats]
    compute_times_sparse = [stat['compute_mean_ms'] for stat in sparse_stats]
    compute_times_36_sparse = [comp * 36 for comp in compute_times_sparse]

    prepare_times_max = [stat['prepare_mean_ms'] for stat in max_stats]
    compute_times_max = [stat['compute_mean_ms'] for stat in max_stats]
    compute_times_36_max = [comp * 36 for comp in compute_times_max]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(settings))
    width = 0.35

    # Stacked bars for sparse method
    # Bottom: compute time (1 layer) - full color
    bars1 = ax.bar(x - width/2, compute_times_sparse, width,
                   label='HA Sparse - Compute (1 layer)',
                   color=METHOD_COLORS['HA_sparse_prefill'], alpha=1.0)

    # Middle: compute time (additional 35 layers for total 36) - hatched
    compute_additional_sparse = [compute_times_36_sparse[i] - compute_times_sparse[i] for i in range(len(compute_times_36_sparse))]
    bars2 = ax.bar(x - width/2, compute_additional_sparse, width,
                   bottom=compute_times_sparse,
                   label='HA Sparse - Compute (36 layers)',
                   color=METHOD_COLORS['HA_sparse_prefill'], alpha=0.8,
                   hatch='///', edgecolor='black', linewidth=0.5)

    # Top: prepare time - different hatch
    ax.bar(x - width/2, prepare_times_sparse, width,
           bottom=compute_times_36_sparse,
           label='HA Sparse - Prepare',
           color=METHOD_COLORS['HA_sparse_prefill'], alpha=0.6,
           hatch='\\\\\\\\', edgecolor='black', linewidth=0.5)

    # Stacked bars for max method
    # Bottom: compute time (1 layer) - full color
    ax.bar(x + width/2, compute_times_max, width,
           label='HA Flatten Max - Compute (1 layer)',
           color=METHOD_COLORS['HA_flatten_max_prefill'], alpha=1.0)

    # Middle: compute time (additional 35 layers for total 36) - hatched
    compute_additional_max = [compute_times_36_max[i] - compute_times_max[i] for i in range(len(compute_times_36_max))]
    ax.bar(x + width/2, compute_additional_max, width,
           bottom=compute_times_max,
           label='HA Flatten Max - Compute (36 layers)',
           color=METHOD_COLORS['HA_flatten_max_prefill'], alpha=0.8,
           hatch='///', edgecolor='black', linewidth=0.5)

    # Top: prepare time - different hatch
    ax.bar(x + width/2, prepare_times_max, width,
           bottom=compute_times_36_max,
           label='HA Flatten Max - Prepare',
           color=METHOD_COLORS['HA_flatten_max_prefill'], alpha=0.6,
           hatch='\\\\\\\\', edgecolor='black', linewidth=0.5)

    # Configure plot
    ax.set_xlabel('Parameter Settings (Max Sparsity per Head, Overall Sparsity)')
    ax.set_ylabel('Time (ms)')
    # ax.set_title('HA Sparse vs HA Flatten Max Performance Comparison\n(Stacked: Compute (1 layer) + Compute (36 layers) + Prepare)')
    ax.set_xticks(x)
    ax.set_xticklabels(settings)

    # Create custom legend with grouped entries
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=METHOD_COLORS['HA_sparse_prefill'], alpha=1.0, label='HA Sparse - Compute (1 layer)'),
        plt.Rectangle((0,0),1,1, facecolor=METHOD_COLORS['HA_sparse_prefill'], alpha=0.8, hatch='///', edgecolor='black', label='HA Sparse - Compute (36 layers)'),
        plt.Rectangle((0,0),1,1, facecolor=METHOD_COLORS['HA_sparse_prefill'], alpha=0.6, hatch='\\\\\\\\', edgecolor='black', label='HA Sparse - Prepare'),
        plt.Rectangle((0,0),1,1, facecolor=METHOD_COLORS['HA_flatten_max_prefill'], alpha=1.0, label='HA Flatten Max - Compute (1 layer)'),
        plt.Rectangle((0,0),1,1, facecolor=METHOD_COLORS['HA_flatten_max_prefill'], alpha=0.8, hatch='///', edgecolor='black', label='HA Flatten Max - Compute (36 layers)'),
        plt.Rectangle((0,0),1,1, facecolor=METHOD_COLORS['HA_flatten_max_prefill'], alpha=0.6, hatch='\\\\\\\\', edgecolor='black', label='HA Flatten Max - Prepare'),
    ]
    ax.legend(handles=legend_elements, ncol=3, bbox_to_anchor=(0.5, -0.125), loc='upper center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# ==============================================================================
# OUTPUT FUNCTIONS
# ==============================================================================

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_CONFIG['figs_folder']):
        os.makedirs(OUTPUT_CONFIG['figs_folder'])

def save_figure(fig, suffix):
    """Save figure to file."""
    filename = f"{OUTPUT_CONFIG['file_prefix']}_{suffix}.{OUTPUT_CONFIG['format']}"
    filepath = os.path.join(OUTPUT_CONFIG['figs_folder'], filename)
    fig.savefig(filepath, dpi=OUTPUT_CONFIG['dpi'], bbox_inches='tight')
    return filepath

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main function to create the comparison visualization."""
    print("HA Sparse vs HA Flatten Max Performance Comparison")
    print("=" * 60)

    # Load data
    data = load_sparse_pattern_data()
    if not data:
        return

    # Prepare data for plotting
    settings, sparse_stats, max_stats = prepare_comparison_data(data)
    if not settings or not sparse_stats or not max_stats:
        print("Error: Failed to prepare data for plotting!")
        return

    print(f"Found {len(settings)} parameter settings:")
    for i, setting in enumerate(settings):
        sparse_prep = sparse_stats[i]['prepare_mean_ms']
        sparse_comp = sparse_stats[i]['compute_mean_ms']
        sparse_comp_36 = sparse_comp * 36
        sparse_total = sparse_prep + sparse_comp_36

        max_prep = max_stats[i]['prepare_mean_ms']
        max_comp = max_stats[i]['compute_mean_ms']
        max_comp_36 = max_comp * 36
        max_total = max_prep + max_comp_36

        print(f"  {i+1}. {setting}")
        print(f"     Sparse: Prepare={sparse_prep:.3f}ms, Compute(1 layer)={sparse_comp:.3f}ms, Compute(36 layers)={sparse_comp_36:.1f}ms, Total={sparse_total:.1f}ms")
        print(f"     Max:    Prepare={max_prep:.3f}ms, Compute(1 layer)={max_comp:.3f}ms, Compute(36 layers)={max_comp_36:.1f}ms, Total={max_total:.1f}ms")

    # Ensure output directory exists
    ensure_output_directory()

    # Create stacked comparison plot
    print("\nCreating stacked comparison plot...")

    fig = create_combined_stacked_plot(settings, sparse_stats, max_stats)
    filepath = save_figure(fig, "stacked_comparison")
    print(f"Saved: {filepath}")

    print(f"\nFigure saved to {OUTPUT_CONFIG['figs_folder']}/")
    plt.show()

if __name__ == "__main__":
    import os
    main()