import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================

# Plot styling
plt.style.use('default')
plt.rcParams['font.size'] = 36

# Data source configuration
DATA_SOURCE_CONFIG = {
    'source': 'LS',  # Options: 'LS', 'LA', 'HS', 'HA'  
}

# Method configurations
METHOD_CONFIGS = {
    'LS': {
        'methods': ['LS_rewrite_prefill', 'LS_sparse_prefill', 'LS_flatten_sum_prefill'],
        'display_names': ['LS Rewrite Prefill', 'LS Sparse Prefill', 'LS Flatten Sum Prefill'],
        'abbreviations': {
            'LS_rewrite_prefill': 'LS-Rewrite',
            'LS_sparse_prefill': 'LS-Sparse',
            'LS_flatten_sum_prefill': 'LS-Sum'
        },
        'colors': ['#006994', '#ff6b6b', '#8b0000']
    },
    'LA': {
        'methods': ['LA_flatten_max_prefill', 'LA_sparse_prefill',],
        'display_names': ['LA Flatten Max Prefill', 'LA Sparse Prefill'],
        'abbreviations': {
            'LA_flatten_max_prefill': 'LA-Max', 
            'LA_sparse_prefill': 'LA-Sparse'
        },
        'colors': ['#006994', '#ff6b6b']
    },
    'HS': {
        'methods': ['HS_flatten_max_prefill', 'HS_sparse_prefill', 'HS_flatten_sum_prefill'],
        'display_names': ['HS Flatten Max Prefill', 'HS Sparse Prefill', 'HS Flatten Sum Prefill'],
        'abbreviations': {
            'HS_flatten_max_prefill': 'HS-Max',
            'HS_sparse_prefill': 'HS-Sparse',
            'HS_flatten_sum_prefill': 'HS-Sum'
        },
        'colors': ['#006994', '#ff6b6b', '#8b0000']
    }, 
    'HA': {
        'methods': ['HA_flatten_max_prefill', 'HA_sparse_prefill'],
        'display_names': ['HA Flatten Max Prefill', 'HA Sparse Prefill'],
        'abbreviations': {
            'HA_flatten_max_prefill': 'HA-Max',
            'HA_sparse_prefill': 'HA-Sparse'
        },
        'colors': ['#006994', '#ff6b6b']
    }
}

# Statistics configurations
STATS_CONFIGS = {
    'LS': {
        'stats_keys': {
            'LS_rewrite_prefill': 'LS_rewrite_prefill',
            'LS_sparse_prefill': 'LS_sparse_prefill',
            'LS_flatten_sum_prefill': 'LS_flatten_sum_prefill'
        }
    },
    'LA': {
        'stats_keys': {
            'LA_flatten_max_prefill': 'LA_flatten_max_prefill',
            'LA_sparse_prefill': 'LA_sparse_prefill'
        }
    },
    'HS': {
        'stats_keys': {
            'HS_flatten_max_prefill': 'HS_flatten_max_prefill',
            'HS_sparse_prefill': 'HS_sparse_prefill',
            'HS_flatten_sum_prefill': 'HS_flatten_sum_prefill'
        }
    }, 
    'HA': {
        'stats_keys': {
            'HA_flatten_max_prefill': 'HA_flatten_max_prefill',
            'HA_sparse_prefill': 'HA_sparse_prefill'
        }
    }
}

# Layers configuration
LAYERS_CONFIG = {
    'total_layers': 36,
    'compute_suffix': '(36 layers)'
}

# Broken axis configuration
BROKEN_AXIS_CONFIG = {
    'enabled': True,
    'method_to_break': None,  # Auto-detect if None
    'height_ratio': [1, 3],
    'hspace': 0.02
}

# Output configurations
OUTPUT_CONFIGS = {
    'LS': {
        'figs_folder': './figs/backend_performance',
        'naming_scheme': 'LS',
        'file_prefix': 'LS_backend_performance',
        'dpi': 300,
        'format': 'pdf'
    },
    'LA': {
        'figs_folder': './figs/backend_performance',
        'naming_scheme': 'LA',
        'file_prefix': 'LA_backend_performance',
        'dpi': 300,
        'format': 'pdf'
    },
    'HS': {
        'figs_folder': './figs/backend_performance',
        'naming_scheme': 'HS',
        'file_prefix': 'HS_backend_performance',
        'dpi': 300,
        'format': 'pdf'
    }, 
    'HA': {
        'figs_folder': './figs/backend_performance',
        'naming_scheme': 'HA',
        'file_prefix': 'HA_backend_performance',
        'dpi': 300,
        'format': 'pdf'
    }
}

# Debug configuration
DEBUG_SINGLE_FIGURE = False
DEBUG_NUM_SEQS = 8

# ==============================================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================================

def load_statistics():
    """Load backend performance statistics from numpy files."""
    source = DATA_SOURCE_CONFIG['source']

    if source == 'LS':
        LS_stats = np.load('LS_backend_stats.npy', allow_pickle=True).item()
        stats_dict = {
            'LS_rewrite_prefill': LS_stats['LS_rewrite_prefill'],
            'LS_sparse_prefill': LS_stats['LS_sparse_prefill'],
            'LS_flatten_sum_prefill': LS_stats['LS_flatten_sum_prefill']
        }
    elif source == 'LA':
        LA_stats = np.load('LA_backend_stats.npy', allow_pickle=True).item()
        stats_dict = {
            'LA_sparse_prefill': LA_stats['LA_sparse_prefill'],
            'LA_flatten_max_prefill': LA_stats['LA_flatten_max_prefill']
        }
    elif source == 'HS':
        HS_stats = np.load('HS_backend_stats.npy', allow_pickle=True).item()
        stats_dict = {
            'HS_flatten_max_prefill': HS_stats['HS_flatten_max_prefill'],
            'HS_sparse_prefill': HS_stats['HS_sparse_prefill'],
            'HS_flatten_sum_prefill': HS_stats['HS_flatten_sum_prefill']
        }
    elif source == 'HA':
        HA_stats = np.load('HA_backend_stats.npy', allow_pickle=True).item()
        stats_dict = {
            'HA_flatten_max_prefill': HA_stats['HA_flatten_max_prefill'],
            'HA_sparse_prefill': HA_stats['HA_sparse_prefill']
        }
    else:
        raise ValueError(f"Unknown data source: {source}")

    return stats_dict

def filter_stats_by_num_seqs(stats_dict, num_seqs_value):
    """Filter statistics for specific number of sequences."""
    filtered_stats = {}

    for method_name, method_stats in stats_dict.items():
        filtered_stats[method_name] = [
            item for item in method_stats
            if item['num_seqs'] == num_seqs_value
        ]

    return filtered_stats

def get_sequence_lengths(filtered_stats):
    """Get unique sequence lengths from filtered statistics."""
    sequence_lengths = sorted(set(
        int(item['seq_length_mean'])
        for method_stats in filtered_stats.values()
        for item in method_stats
    ))
    return sequence_lengths

def calculate_method_totals(filtered_stats, stats_config, sequence_lengths):
    """Calculate total times (prepare + compute) for each method."""
    source = DATA_SOURCE_CONFIG['source']
    method_config = METHOD_CONFIGS[source]
    method_totals = {}

    for method in method_config['methods']:
        stat_key = find_stat_key_for_method(method, stats_config)

        if stat_key and stat_key in filtered_stats:
            totals = []
            for item in filtered_stats[stat_key]:
                total = item['prepare_mean_ms'] + item['compute_mean_ms']
                totals.append(total)
            method_totals[method] = totals

    return method_totals

def find_stat_key_for_method(method, stats_config):
    """Find the corresponding stat key for a method."""
    method_normalized = method.replace(' ', '_').lower()

    for file_key, display_key in stats_config['stats_keys'].items():
        if method_normalized == file_key.lower() or method_normalized == display_key.lower():
            return display_key

    return None

# ==============================================================================
# BROKEN AXIS CALCULATIONS
# ==============================================================================

def calculate_broken_axis_ranges(prepare_times, compute_times_total_layers, sequence_lengths):
    """Calculate y-axis ranges for broken axis visualization."""
    source = DATA_SOURCE_CONFIG['source']
    method_config = METHOD_CONFIGS[source]

    # Calculate 36-layer totals for all methods
    all_method_totals = []
    for method in method_config['methods']:
        if method in prepare_times and method in compute_times_total_layers:
            for i in range(len(sequence_lengths)):
                total_36_layers = prepare_times[method][i] + compute_times_total_layers[method][i]
                all_method_totals.append(total_36_layers)

    if not all_method_totals:
        return 0, 0, 0, 0

    max_total_value = max(all_method_totals)
    print(f"Max 36-layer total value: {max_total_value:.2f}")

    # Find the break point - use the maximum value excluding the highest method
    method_averages = {}
    for method in method_config['methods']:
        if method in prepare_times and method in compute_times_total_layers:
            totals = []
            for i in range(len(sequence_lengths)):
                total_36_layers = prepare_times[method][i] + compute_times_total_layers[method][i]
                totals.append(total_36_layers)
            method_averages[method] = np.mean(totals) if totals else 0

    # Sort methods by average to find the highest
    sorted_methods = sorted(method_averages.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_methods) >= 2:
        highest_method_name = sorted_methods[0][0]

        # Calculate max value of second highest method
        second_highest_max = 0
        for method in method_config['methods']:
            if method != highest_method_name and method in prepare_times and method in compute_times_total_layers:
                for i in range(len(sequence_lengths)):
                    total_36_layers = prepare_times[method][i] + compute_times_total_layers[method][i]
                    second_highest_max = max(second_highest_max, total_36_layers)

        # Set break point above second highest method
        break_point = second_highest_max * 1.2
        print(f"Break point: {break_point:.2f}")

        # Configure ranges: lower subplot covers 0 to break_point, upper covers break_point to max_total_value*1.1
        ax1_ylim_bottom = break_point
        ax1_ylim_top = max_total_value * 1.1
        ax2_ylim_bottom = 0
        ax2_ylim_top = break_point
    else:
        # Fallback: use same range for both
        ax1_ylim_bottom = 0
        ax1_ylim_top = max_total_value * 1.1
        ax2_ylim_bottom = 0
        ax2_ylim_top = max_total_value * 1.1

    return ax1_ylim_bottom, ax1_ylim_top, ax2_ylim_bottom, ax2_ylim_top

def determine_method_to_break(method_totals):
    """Determine which method should be placed in the upper subplot."""
    if BROKEN_AXIS_CONFIG['method_to_break']:
        return BROKEN_AXIS_CONFIG['method_to_break']

    # Auto-detect: find method with highest average total time
    avg_totals = {}
    for method, totals in method_totals.items():
        if totals:
            avg_totals[method] = np.mean(totals)

    if avg_totals:
        method_to_break = max(avg_totals, key=avg_totals.get)
        print(f"Breaking axis for method: {method_to_break}")
        return method_to_break

    return None

# ==============================================================================
# DATA PREPARATION FUNCTIONS
# ==============================================================================

def prepare_plot_data(filtered_stats, stats_config, sequence_lengths):
    """Prepare prepare_times and compute_times data for plotting."""
    source = DATA_SOURCE_CONFIG['source']
    method_config = METHOD_CONFIGS[source]
    prepare_times = {}
    compute_times = {}

    for method in method_config['methods']:
        stat_key = find_stat_key_for_method(method, stats_config)

        if stat_key is None:
            print(f"Warning: No stat key found for method {method}")
            prepare_times[method] = [0] * len(sequence_lengths)
            compute_times[method] = [0] * len(sequence_lengths)
            continue

        # Create lookup dict for this method
        data_lookup = {
            int(item['seq_length_mean']): item
            for item in filtered_stats[stat_key]
        }

        prepare_times[method] = [
            data_lookup[seq_len]['prepare_mean_ms']
            if seq_len in data_lookup else 0
            for seq_len in sequence_lengths
        ]

        compute_times[method] = [
            data_lookup[seq_len]['compute_mean_ms']
            if seq_len in data_lookup else 0
            for seq_len in sequence_lengths
        ]

    return prepare_times, compute_times

def compute_total_layers_times(compute_times):
    """Compute times for total number of layers."""
    source = DATA_SOURCE_CONFIG['source']
    method_config = METHOD_CONFIGS[source]
    total_layers = LAYERS_CONFIG['total_layers']
    compute_times_total_layers = {
        method: [compute_times[method][i] * total_layers for i in range(len(compute_times[method]))]
        for method in method_config['methods']
    }
    return compute_times_total_layers

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def create_figure(use_broken_axis):
    """Create matplotlib figure with appropriate subplot structure."""
    if use_broken_axis:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 10),
                                       gridspec_kw={'height_ratios': BROKEN_AXIS_CONFIG['height_ratio'],
                                                   'hspace': BROKEN_AXIS_CONFIG['hspace']})
        return fig, ax1, ax2
    else:
        fig, ax = plt.subplots(figsize=(28, 10))
        return fig, ax, None

def plot_broken_axis_data(ax1, ax2, method, prepare_times, compute_times, compute_times_total_layers,
                         x, width, colors, i, highest_method):
    """Plot data for broken axis visualization."""
    offset = (i - 1) * width
    method_handles = []

    if method == highest_method:
        # Plot highest method in upper subplot (ax1)
        # Compute time for 1 layer (bottom layer)
        bar1 = ax1.bar(x + offset, compute_times[method], width,
                      color=colors[i], alpha=1.0)
        # Compute time for additional layers (middle layer)
        bar2 = ax1.bar(x + offset, np.array(compute_times_total_layers[method]) - np.array(compute_times[method]), width,
                      bottom=compute_times[method],
                      color=colors[i], alpha=0.7, hatch='///', edgecolor='black', linewidth=0.5)
        # Prepare time (top layer)
        bar3 = ax1.bar(x + offset, prepare_times[method], width,
                      bottom=compute_times_total_layers[method],
                      color=colors[i], alpha=0.4, hatch='\\\\\\\\', edgecolor='black', linewidth=0.5)

        method_handles.extend([bar1[0], bar2[0], bar3[0]])

    # Plot ALL methods in lower subplot (ax2)
    # Compute time for 1 layer (bottom layer)
    bar1 = ax2.bar(x + offset, compute_times[method], width,
                  color=colors[i], alpha=1.0)
    # Compute time for additional layers (middle layer)
    bar2 = ax2.bar(x + offset, compute_times_total_layers[method], width,
                  bottom=compute_times[method],
                  color=colors[i], alpha=0.7, hatch='///', edgecolor='black', linewidth=0.5)
    # Prepare time (top layer)
    bottom_total_layers = [compute_times_total_layers[method][j] for j in range(len(compute_times[method]))]
    bar3 = ax2.bar(x + offset, prepare_times[method], width,
                  bottom=bottom_total_layers,
                  color=colors[i], alpha=0.4, hatch='\\\\\\\\', edgecolor='black', linewidth=0.5)

    method_handles.extend([bar1[0], bar2[0], bar3[0]])

    return method_handles

def plot_regular_axis_data(ax, method, prepare_times, compute_times, compute_times_total_layers,
                         x, width, colors, i):
    """Plot data for regular (non-broken) axis visualization."""
    offset = (i - 1) * width

    # Compute time for 1 layer (bottom layer)
    bar1 = ax.bar(x + offset, compute_times[method], width,
                 color=colors[i], alpha=1.0)

    # Compute time for additional layers (middle layer)
    bar2 = ax.bar(x + offset, np.array(compute_times_total_layers[method]) - np.array(compute_times[method]), width,
                 bottom=compute_times[method],
                 color=colors[i], alpha=0.7, hatch='///', edgecolor='black', linewidth=0.5)

    # Prepare time (top layer)
    bar3 = ax.bar(x + offset, prepare_times[method], width,
                 bottom=compute_times_total_layers[method],
                 color=colors[i], alpha=0.4, hatch='\\\\\\\\', edgecolor='black', linewidth=0.5)

    return [bar1[0], bar2[0], bar3[0]]

def configure_broken_axis_subplots(ax1, ax2, x, sequence_lengths, width, ax1_ylim_bottom, ax1_ylim_top,
                                  ax2_ylim_bottom, ax2_ylim_top, handles, labels):
    """Configure broken axis subplots with proper styling."""
    # Set y-axis limits
    ax1.set_ylim(ax1_ylim_bottom, ax1_ylim_top)
    ax2.set_ylim(ax2_ylim_bottom, ax2_ylim_top)

    # Remove spines between subplots
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Add diagonal lines to indicate break
    d = 0.015
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # Set labels and ticks
    # Remove y-axis label from upper subplot
    ax1.set_ylabel('')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(seq_len) for seq_len in sequence_lengths])

    # Set x-axis limits
    x_min, x_max = x[0] - width * 1.5, x[-1] + width * 1.5
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    # Add grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

def configure_regular_axis(ax, x, sequence_lengths, width, handles, labels):
    """Configure regular axis subplot with proper styling."""
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    
    # Set x-axis limits
    x_min, x_max = x[0] - width * 1.5, x[-1] + width * 1.5
    ax.set_xlim(x_min, x_max)

    ax.set_xticklabels([str(seq_len) for seq_len in sequence_lengths])
    ax.grid(True, alpha=0.3)

# ==============================================================================
# OUTPUT FUNCTIONS
# ==============================================================================

def generate_filename(num_seqs_value):
    """Generate output filename."""
    source = DATA_SOURCE_CONFIG['source']
    output_config = OUTPUT_CONFIGS[source]
    base_name = f"{output_config['file_prefix']}_{num_seqs_value}seqs"
    extension = output_config['format']
    return os.path.join(output_config['figs_folder'], f"{base_name}.{extension}")

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    source = DATA_SOURCE_CONFIG['source']
    output_config = OUTPUT_CONFIGS[source]
    if not os.path.exists(output_config['figs_folder']):
        os.makedirs(output_config['figs_folder'])

def save_figure(fig, num_seqs_value):
    """Save figure to file."""
    source = DATA_SOURCE_CONFIG['source']
    output_config = OUTPUT_CONFIGS[source]
    filename = generate_filename(num_seqs_value)
    fig.savefig(filename, dpi=output_config['dpi'], bbox_inches='tight')
    plt.close(fig)
    return filename

# ==============================================================================
# MAIN VISUALIZATION FUNCTION
# ==============================================================================

def create_figure_for_num_seqs(stats_dict, num_seqs_value):
    """Create visualization for a specific number of sequences."""
    source = DATA_SOURCE_CONFIG['source']
    method_config = METHOD_CONFIGS[source]
    stats_config = STATS_CONFIGS[source]

    print(f"\nCreating figure for num_seqs = {num_seqs_value}")

    # Filter data
    filtered_stats = filter_stats_by_num_seqs(stats_dict, num_seqs_value)
    sequence_lengths = get_sequence_lengths(filtered_stats)

    if not sequence_lengths:
        print(f"No data available for num_seqs = {num_seqs_value}")
        return

    print(f"Sequence lengths: {sequence_lengths}")

    # Calculate method totals for broken axis determination
    method_totals = calculate_method_totals(filtered_stats, stats_config, sequence_lengths)

    # Determine if broken axis should be used
    use_broken_axis = BROKEN_AXIS_CONFIG['enabled'] and len(method_totals) > 0
    print(f"Broken axis enabled: {use_broken_axis}")

    # Create figure
    fig, ax1, ax2 = create_figure(use_broken_axis)

    # Prepare data
    prepare_times, compute_times = prepare_plot_data(filtered_stats, stats_config, sequence_lengths)
    compute_times_total_layers = compute_total_layers_times(compute_times)

    # Calculate broken axis ranges if needed
    if use_broken_axis:
        ax1_ylim_bottom, ax1_ylim_top, ax2_ylim_bottom, ax2_ylim_top = calculate_broken_axis_ranges(
            prepare_times, compute_times_total_layers, sequence_lengths
        )

    # Determine which method to break
    method_to_break_upper = determine_method_to_break(method_totals) if use_broken_axis else None

    # Plot setup
    width = 0.25
    x = np.arange(len(sequence_lengths))
    handles_by_method = []
    labels_by_method = []

    # Debug: Verify data integrity
    print(f"Data verification for num_seqs = {num_seqs_value}:")
    for method in method_config['methods'][:3]:  # Check first 3 methods
        if method in prepare_times and method in compute_times:
            for i, seq_len in enumerate(sequence_lengths[:3]):  # Check first 3 sequence lengths
                prep = prepare_times[method][i]
                comp = compute_times[method][i]
                total = prep + comp
                print(f"  {method} - seq_len {seq_len}: prepare={prep:.3f}, compute={comp:.3f}, total={total:.3f}")

    # Plot data for each method
    for i, method in enumerate(method_config['methods']):
        abbrev = method_config['abbreviations'][method]
        method_handles = []

        if use_broken_axis:
            handles = plot_broken_axis_data(
                ax1, ax2, method, prepare_times, compute_times, compute_times_total_layers,
                x, width, method_config['colors'], i, method_to_break_upper
            )
        else:
            handles = plot_regular_axis_data(
                ax1 if ax2 is None else ax1, method, prepare_times, compute_times,
                compute_times_total_layers, x, width, method_config['colors'], i
            )

        method_handles.extend(handles)

        method_labels = [
            f'{abbrev} - Compute (1 layer)',
            f'{abbrev} - Compute {LAYERS_CONFIG["compute_suffix"]}',
            f'{abbrev} - Prepare'
        ]

        handles_by_method.append(method_handles)
        labels_by_method.append(method_labels)

    # Reorder legend handles and labels to group by component type
    handles = []
    labels = []
    for component_idx in range(3):
        for method_idx in range(len(method_config['methods'])):
            if len(handles_by_method[method_idx]) > component_idx:
                handles.append(handles_by_method[method_idx][component_idx])
                labels.append(labels_by_method[method_idx][component_idx])

    # Configure subplots
    if use_broken_axis:
        configure_broken_axis_subplots(
            ax1, ax2, x, sequence_lengths, width, ax1_ylim_bottom, ax1_ylim_top,
            ax2_ylim_bottom, ax2_ylim_top, handles, labels
        )
    else:
        configure_regular_axis(
            ax1 if ax2 is None else ax1, x, sequence_lengths, width, handles, labels
        )

    legend_ax = ax1 if use_broken_axis else (ax1 if ax2 is None else ax1)

    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.1), loc='upper center',
              ncol=3, columnspacing=0.8, handletextpad=0.4, frameon=True)
    
    # Add statistics text
    stats_text = f"Number of requests:{num_seqs_value}\nAll Layer Performance (ms):\n"
    for method in method_config['methods']:
        stat_key = find_stat_key_for_method(method, stats_config)
        if stat_key and stat_key in filtered_stats and filtered_stats[stat_key]:
            # Calculate averages across all sequence lengths
            avg_prepare = np.mean([item['prepare_mean_ms'] for item in filtered_stats[stat_key]])
            avg_compute = np.mean([item['compute_mean_ms'] for item in filtered_stats[stat_key]])
            # Multiply compute time by total layers for all-layer performance
            total_compute_time = avg_compute * LAYERS_CONFIG['total_layers']
            abbrev = method_config['abbreviations'][method]
            stats_text += f"{abbrev}: {avg_prepare:.2f} (prepare avg), {total_compute_time:.2f} (compute {LAYERS_CONFIG['total_layers']} layers avg)\n"

    text_ax = ax2 if use_broken_axis else (ax1 if ax2 is None else ax1)
    text_ax.text(0.02, 0.98, stats_text, transform=text_ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.20)

    # Print detailed statistics
    print(f"\n=== Detailed Statistics for {num_seqs_value} Sequences ===")
    for method in method_config['methods']:
        stat_key = find_stat_key_for_method(method, stats_config)
        print(f"\n{method} (stat_key={stat_key}):")
        if stat_key and stat_key in filtered_stats:
            print(f"  Found {len(filtered_stats[stat_key])} entries")
            for item in filtered_stats[stat_key][:3]:  # Show first 3 entries
                seq_len = int(item['seq_length_mean'])
                prep = item['prepare_mean_ms']
                comp = item['compute_mean_ms']
                total = prep + comp
                print(f"  Seq Length {seq_len}: Prepare={prep:.2f}ms, Compute={comp:.2f}ms, Total={total:.2f}ms")
        else:
            print(f"  No statistics available for this method (stat_key={stat_key}, in_filtered={stat_key in filtered_stats if stat_key else 'N/A'})")

    # Save figure
    filename = save_figure(fig, num_seqs_value)
    print(f"Saved: {filename}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main function to run the visualization."""
    source = DATA_SOURCE_CONFIG['source']
    print(f"{source} Backend Performance Visualization")
    print("=" * 50)

    # Load statistics
    stats_dict = load_statistics()

    # Get available num_seqs values
    all_num_seqs = sorted(set(
        item['num_seqs']
        for method_stats in stats_dict.values()
        for item in method_stats
    ))

    source = DATA_SOURCE_CONFIG['source']
    method_config = METHOD_CONFIGS[source]
    output_config = OUTPUT_CONFIGS[source]

    print(f"Available num_seqs values: {all_num_seqs}")
    print(f"Using methods: {method_config['methods']}")

    # Ensure output directory exists
    ensure_output_directory()

    # Create figures
    if DEBUG_SINGLE_FIGURE:
        if DEBUG_NUM_SEQS in all_num_seqs:
            create_figure_for_num_seqs(stats_dict, DEBUG_NUM_SEQS)
        else:
            print(f"DEBUG_NUM_SEQS {DEBUG_NUM_SEQS} not found. Available: {all_num_seqs}")
    else:
        for num_seqs in all_num_seqs:
            create_figure_for_num_seqs(stats_dict, num_seqs)

    print(f"\nAll figures saved to {output_config['figs_folder']}/")
    print(f"Generated files:")
    for num_seqs in all_num_seqs if not DEBUG_SINGLE_FIGURE else [DEBUG_NUM_SEQS]:
        if num_seqs in all_num_seqs or (DEBUG_SINGLE_FIGURE and num_seqs == DEBUG_NUM_SEQS):
            filename = generate_filename(num_seqs)
            if os.path.exists(filename):
                print(f"  - {filename}")

if __name__ == "__main__":
    main()