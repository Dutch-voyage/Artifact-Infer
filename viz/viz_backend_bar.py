import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION SECTION - Customize names and stats types here
# ==============================================================================

# Method name configurations - pick one of these options
METHOD_CONFIGS = {
    'abbrev': {
        'methods': ['Paged Decode', 'Paged Prefill', 'Ragged Prefill'],
        'abbreviations': {'Paged Decode': 'PD', 'Paged Prefill': 'PP', 'Ragged Prefill': 'RP'},
        'colors': ['#4CAF50', '#2196F3', '#FF9800']
    },
    'full': {
        'methods': ['Paged Decode', 'Paged Prefill', 'Ragged Prefill'],
        'abbreviations': {'Paged Decode': 'Paged Decode', 'Paged Prefill': 'Paged Prefill', 'Ragged Prefill': 'Ragged Prefill'},
        'colors': ['#4CAF50', '#2196F3', '#FF9800']
    },
    'short': {
        'methods': ['Decode', 'Prefill', 'Ragged'],
        'abbreviations': {'Decode': 'Decode', 'Prefill': 'Prefill', 'Ragged': 'Ragged'},
        'colors': ['#4CAF50', '#2196F3', '#FF9800']
    },
    'compact': {
        'methods': ['Decode', 'Prefill', 'Ragged'],
        'abbreviations': {'Decode': 'D', 'Prefill': 'P', 'Ragged': 'R'},
        'colors': ['#4CAF50', '#2196F3', '#FF9800']
    }
}

# Stats type configurations - pick one of these options
STATS_CONFIGS = {
    'current': {
        'file_name': 'backend_stats.npy',
        'stats_keys': {
            'paged_decode': 'paged_decode',    # file key -> display key
            'paged_prefill': 'paged_prefill',
            'sparse_prefill': 'ragged_prefill'  # file key 'sparse_prefill' -> display as 'ragged_prefill'
        }
    },
    'extended': {
        'file_name': 'backend_stats_extended.npy',
        'stats_keys': {
            'paged_decode': 'paged_decode',
            'paged_prefill': 'paged_prefill',
            'ragged_prefill': 'ragged_prefill',
            'flash_decode': 'flash_decode',
            'triton_decode': 'triton_decode',
            'custom_sparse': 'custom_sparse'
        }
    },
    'alternative': {
        'file_name': 'alternative_stats.npy',
        'stats_keys': {
            'baseline_decode': 'baseline',
            'optimized_decode': 'optimized',
            'sparse_decode': 'sparse',
            'ragged_decode': 'ragged'
        }
    }
}

# Layers configuration
LAYERS_CONFIG = {
    'total_layers': 36,
    'compute_suffix': f'(36 layers)',
    'annotation_text': 'Solid: Single layer compute\nHatched area: 35 additional layers\n(36 layers total)'
}

# ==============================================================================
# USER SELECTION - Pick your preferred configuration
# ==============================================================================

# Choose your method naming scheme: 'abbrev', 'full', 'short', 'compact'
METHOD_CHOICE = 'abbrev'

# Choose your stats type: 'current', 'extended', 'alternative'
STATS_CHOICE = 'current'

# Load chosen configurations
method_config = METHOD_CONFIGS[METHOD_CHOICE]
stats_config = STATS_CONFIGS[STATS_CHOICE]

# Extract configurations
methods = method_config['methods']
method_abbrev = method_config['abbreviations']
colors = method_config['colors']

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 36

# Load the backend stats using the selected configuration
backend_stats = np.load(stats_config['file_name'], allow_pickle=True).item()

print("Available methods in file:", list(backend_stats.keys()))
print("Using methods:", list(stats_config['stats_keys'].keys()))

# Extract data for each method using the configurable keys
stats_dict = {}
for file_key, display_key in stats_config['stats_keys'].items():
    if file_key in backend_stats:
        # Convert list to dictionary indexed by sequence length for easier access
        stats_dict[display_key] = {
            int(item['seq_length_mean']): item for item in backend_stats[file_key]
        }
    else:
        print(f"Warning: {file_key} not found in stats file")

# Get all unique sequence lengths and sort them from all available stats
all_stats_keys = list(stats_dict.keys())
sequence_lengths = sorted(set(
    [seq_len for stat_key in all_stats_keys for seq_len in stats_dict[stat_key].keys()]
))

print(f"Sequence lengths: {sequence_lengths}")

# Create single figure
fig, ax = plt.subplots(figsize=(28, 10))

width = 0.25
x = np.arange(len(sequence_lengths))

# Prepare data for stacked bars using configurable method names
prepare_times = {}
compute_times = {}

for method in methods:
    stat_key = None
    # Find the corresponding stat key for this method using multiple strategies
    for file_key, display_key in stats_config['stats_keys'].items():
        # Try exact match
        if method.replace(' ', '_').lower() == display_key:
            stat_key = display_key
            break
        # Try partial match
        if any(part in display_key for part in method.lower().split()):
            stat_key = display_key
            break
        # Try reverse match
        if any(part in method.lower() for part in display_key.replace('_', ' ').split()):
            stat_key = display_key
            break

    if stat_key is None:
        print(f"Warning: No stat key found for method {method}")
        prepare_times[method] = [0] * len(sequence_lengths)
        compute_times[method] = [0] * len(sequence_lengths)
        continue

    prepare_times[method] = [
        stats_dict[stat_key][seq_len]['prepare_mean_ms']
        if seq_len in stats_dict[stat_key] else 0
        for seq_len in sequence_lengths
    ]
    compute_times[method] = [
        stats_dict[stat_key][seq_len]['compute_mean_ms']
        if seq_len in stats_dict[stat_key] else 0
        for seq_len in sequence_lengths
    ]

# Additional layers computation time using configurable total
total_layers = LAYERS_CONFIG['total_layers']
compute_times_total_layers = {
    method: [compute_times[method][i] * total_layers for i in range(len(sequence_lengths))]
    for method in methods
}

# Plot stacked bars
for i, method in enumerate(methods):
    offset = (i - 1) * width
    abbrev = method_abbrev[method]

    # Prepare time (bottom layer)
    ax.bar(x + offset, prepare_times[method], width,
          label=f'{abbrev} - Prepare', color=colors[i], alpha=0.7)

    # Compute time for 1 layer (second layer)
    ax.bar(x + offset, compute_times[method], width,
          bottom=prepare_times[method],
          label=f'{abbrev} - Compute (1 layer)', color=colors[i], alpha=0.9)

    # Compute time for additional layers as a dotted/hatched area on top
    bottom_total_layers = [prepare_times[method][j] + compute_times[method][j]
                          for j in range(len(sequence_lengths))]

    ax.bar(x + offset, compute_times_total_layers[method], width,
          bottom=bottom_total_layers,
          color=colors[i], alpha=0.4, hatch='///', edgecolor='black', linewidth=0.5,
          label=f'{abbrev} - Compute {LAYERS_CONFIG["compute_suffix"]}')

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Time (ms)')
# ax.set_title('Backend Performance: Prepare Time and Compute Time (36 Layers Stacked)')
ax.set_xticks(x)
ax.set_xticklabels([str(seq_len) for seq_len in sequence_lengths])
ax.legend(bbox_to_anchor=(0.5, -0.13), loc='upper center', ncol=3)
ax.grid(True, alpha=0.3)

# # Add annotations to explain the dotted area
# ax.text(0.02, 0.98, 'Solid: Single layer compute\nHatched area: 35 additional layers\n(36 layers total)',
#         transform=ax.transAxes, verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Add some performance statistics
stats_text = "All Layer Performance (ms):\n"
for method in methods:
    # Find the corresponding stat key for this method using multiple strategies
    stat_key = None
    for file_key, display_key in stats_config['stats_keys'].items():
        # Try exact match
        if method.replace(' ', '_').lower() == display_key:
            stat_key = display_key
            break
        # Try partial match
        if any(part in display_key for part in method.lower().split()):
            stat_key = display_key
            break
        # Try reverse match
        if any(part in method.lower() for part in display_key.replace('_', ' ').split()):
            stat_key = display_key
            break

    if stat_key and stat_key in stats_dict:
        avg_prepare = np.mean([stats_dict[stat_key][seq_len]['prepare_mean_ms']
                              for seq_len in stats_dict[stat_key]])
        avg_compute = np.mean([stats_dict[stat_key][seq_len]['compute_mean_ms']
                              for seq_len in stats_dict[stat_key]])
        stats_text += f"{method_abbrev[method]}: {avg_prepare:.2f} (prepare), {avg_compute * total_layers:.2f} (compute)\n"

stats_text = stats_text.strip("\n")

ax.text(0.02, 0.68, stats_text, transform=ax.transAxes,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Adjust layout to minimize empty space and make room for bottom legend
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.2)
plt.savefig('backend_performance_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# Print detailed statistics
print("\n=== Detailed Statistics ===")
for method in methods:
    # Find the corresponding stat key for this method using multiple strategies
    stat_key = None
    for file_key, display_key in stats_config['stats_keys'].items():
        # Try exact match
        if method.replace(' ', '_').lower() == display_key:
            stat_key = display_key
            break
        # Try partial match
        if any(part in display_key for part in method.lower().split()):
            stat_key = display_key
            break
        # Try reverse match
        if any(part in method.lower() for part in display_key.replace('_', ' ').split()):
            stat_key = display_key
            break

    print(f"\n{method}:")
    if stat_key and stat_key in stats_dict:
        for seq_len in sequence_lengths:
            if seq_len in stats_dict[stat_key]:
                prep = stats_dict[stat_key][seq_len]['prepare_mean_ms']
                comp = stats_dict[stat_key][seq_len]['compute_mean_ms']
                total = prep + comp
                print(f"  Seq Length {seq_len}: Prepare={prep:.2f}ms, Compute={comp:.2f}ms, Total={total:.2f}ms")
            else:
                print(f"  Seq Length {seq_len}: No data available")
    else:
        print(f"  No statistics available for this method")