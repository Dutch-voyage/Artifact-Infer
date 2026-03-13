import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.style.use('default')
plt.rcParams['font.size'] = 36

# Configuration
CONFIG = {
    'data_path': 'HA_backend_stats.npy',
    'output_dir': 'figs/product/',
    'dpi': 300,
    'fig_format': 'png'
}

num_merge_steps = 1
num_seqs = 32
seq_length = 2048

# Load data
HA_stats = np.load(CONFIG['data_path'], allow_pickle=True).item()

stats_sparse = HA_stats["HA_sparse_prefill"]
stats_max = HA_stats["HA_flatten_max_prefill"]

stats_sparse = [item for item in stats_sparse if item["num_seqs"] == num_seqs]
stats_max = [item for item in stats_max if item["num_seqs"] == num_seqs]

data_index = seq_length // 2048 - 1

# Memory allocation (in GB)
base_mem = num_seqs * seq_length * 1/8 * 144 / 1024 / 1024
double_mem = num_seqs * seq_length * 1/8 * 288 / 1024 / 1024

# Extract latency data (in ms)
# HA_Sparse method: no rewrite phase
sparse_prepare = stats_sparse[data_index]["prepare_mean_ms"]
sparse_compute = stats_sparse[data_index]["compute_mean_ms"] * 36

# HA_Max method: has rewrite phase
max_rewrite = stats_max[data_index]["rewrite_mean_ms"]
max_prepare = stats_max[data_index]["prepare_mean_ms"] - stats_max[data_index]["rewrite_mean_ms"]
max_compute = stats_max[data_index]["compute_mean_ms"] * 36

step_mem_increase = num_seqs * 144 / 1024 / 1024 # KB per step

def solve(a, b, c):
    determinant = b**2 - 4*a*c
    if determinant < 0:
        return None  # No real solutions
    sqrt_det = np.sqrt(determinant)
    x1 = (-b + sqrt_det) / (2 * a)
    x2 = (-b - sqrt_det) / (2 * a)
    return x1, x2

def calculate_boundary():
    """Calculate the number of decoding steps when HA_Sparse and HA_Max integrals are equal"""
    
    # Pre-initialization integrals (KB·ms)
    sparse_pre_integral = base_mem * (sparse_prepare + sparse_compute)
    max_pre_integral = double_mem * max_rewrite + base_mem * (max_prepare + max_compute)
    # Per-step memory increase (KB) - both methods add new_seq_lengths * 144KB
    # Assuming new_seq_lengths = 1 for each decoding step
    
    # Per-step time costs (ms) - normalized to 1 unit of seq length
    sparse_step_time = sparse_prepare + sparse_compute
    max_step_time =  max_prepare + max_compute
    
    # Per-step integrals (KB·ms)
    sparse_step_integral = step_mem_increase * sparse_step_time
    max_step_integral = step_mem_increase * max_step_time

    a = step_mem_increase / 2 * (sparse_step_time - max_step_time)
    b = (step_mem_increase / 2 + base_mem) * (sparse_step_time - max_step_time)
    c =  sparse_pre_integral - max_pre_integral

    boundary_steps_float, _ = solve(a, b, c)

    # Convert to integer for complete decoding steps (round up to next whole step)
    if boundary_steps_float is not None and boundary_steps_float > 0:
        boundary_steps = int(np.ceil(boundary_steps_float))
    else:
        boundary_steps = boundary_steps_float

    return boundary_steps, boundary_steps_float, sparse_step_integral, max_step_integral, step_mem_increase

def create_subfigures():
    """Create single figure with two subplots side by side"""

    # Calculate boundary first
    boundary_steps, boundary_steps_float, sparse_step_integral, max_step_integral, step_mem_increase = calculate_boundary()
    print(f"Boundary calculation:")
    print(f"  Sparse per-step integral: {sparse_step_integral:.2e} KB·ms")
    print(f"  Max per-step integral: {max_step_integral:.2e} KB·ms")
    print(f"  Boundary at {boundary_steps_float:.1f} decoding steps (rounded to {boundary_steps} complete steps)")
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

    # Color scheme
    colors = {
        'sparse_prepare': '#FF6B6B',  # Red
        'sparse_compute': '#FFA07A',  # Light Salmon
        'max_rewrite': '#4ECDC4',     # Teal
        'max_prepare': '#45B7D1',     # Blue
        'max_compute': '#96CEB4'      # Green
    }

    num_steps_to_show = 10
    step_height = step_mem_increase
    sparse_step_time = sparse_prepare + sparse_compute
    max_step_time = max_prepare + max_compute
    sparse_x_end = sparse_prepare + sparse_compute
    max_x_end = max_rewrite + max_prepare + max_compute

    # ==================== LEFT SUBPLOT: HA_Sparse ====================
    # HA_Sparse phases
    sparse_phases = [
        {
            'name': 'Prepare',
            'x': 0, 'y': 0,
            'width': sparse_prepare,
            'height': base_mem,
            'color': colors['sparse_prepare'],
            'integral': base_mem * 1024 * sparse_prepare
        },
        {
            'name': 'Compute',
            'x': sparse_prepare, 'y': 0,
            'width': sparse_compute,
            'height': base_mem,
            'color': colors['sparse_compute'],
            'integral': base_mem * 1024 * sparse_compute
        }
    ]

    # Draw HA_Sparse pre-initialization phases
    for phase in sparse_phases:
        rect = mpatches.Rectangle(
            (phase['x'], phase['y']),
            phase['width'],
            phase['height'],
            facecolor='none',
            edgecolor=phase['color'],
            linewidth=2,
            linestyle='-',
            hatch='//',
            alpha=1.0
        )
        ax1.add_patch(rect)

    # Add HA_Sparse decoding step squares
    for step in range(0, num_steps_to_show, num_merge_steps):
        sparse_y_offset = base_mem + step * step_height
        sparse_rect = mpatches.Rectangle(
            (sparse_x_end + step * sparse_step_time, 0),
            sparse_step_time * num_merge_steps,
            sparse_y_offset,
            facecolor='none',
            edgecolor=colors['sparse_compute'],
            linewidth=1,
            linestyle='-',
            hatch='//',
            alpha=1.0
        )
        ax1.add_patch(sparse_rect)

    # HA_Sparse total
    sparse_total = sum(phase['integral'] for phase in sparse_phases)

    # HA_Sparse formatting
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Memory (GB)')
    ax1.set_title('(a) HA_Sparse', fontsize=36, pad=15)

    # HA_Sparse axis limits
    max_x_limit_sparse = sparse_x_end + num_steps_to_show * sparse_step_time + 0.5
    max_y_limit_sparse = base_mem + num_steps_to_show * step_height
    ax1.set_xlim(0, max_x_limit_sparse)
    ax1.set_ylim(0, max_y_limit_sparse)

    # HA_Sparse legend
    legend_elements_sparse = [
        mpatches.Patch(facecolor='none', edgecolor=colors['sparse_prepare'], hatch='//',
                      label='Prepare'),
        mpatches.Patch(facecolor='none', edgecolor=colors['sparse_compute'], hatch='//',
                      label='Compute'),
    ]
    ax1.legend(handles=legend_elements_sparse, loc='upper left', fancybox=True, shadow=True, fontsize=24)

    # ==================== RIGHT SUBPLOT: HA_Max ====================
    # HA_Max phases
    max_phases = [
        {
            'name': 'Rewrite',
            'x': 0, 'y': 0,
            'width': max_rewrite,
            'height': double_mem,
            'color': colors['max_rewrite'],
            'integral': double_mem * 1024 * max_rewrite
        },
        {
            'name': 'Prepare',
            'x': max_rewrite, 'y': 0,
            'width': max_prepare,
            'height': base_mem,
            'color': colors['max_prepare'],
            'integral': double_mem * 1024 * max_prepare
        },
        {
            'name': 'Compute',
            'x': max_rewrite + max_prepare, 'y': 0,
            'width': max_compute,
            'height': base_mem,
            'color': colors['max_compute'],
            'integral': base_mem * 1024 * max_compute
        }
    ]

    # Draw HA_Max pre-initialization phases
    for phase in max_phases:
        rect = mpatches.Rectangle(
            (phase['x'], phase['y']),
            phase['width'],
            phase['height'],
            facecolor='none',
            edgecolor=phase['color'],
            linewidth=2,
            linestyle='--',
            hatch='\\\\',
            alpha=1.0
        )
        ax2.add_patch(rect)

    # Add HA_Max decoding step squares
    for step in range(0, num_steps_to_show, num_merge_steps):
        max_y_offset = base_mem + step * step_height
        max_rect = mpatches.Rectangle(
            (max_x_end + step * max_step_time, 0),
            max_step_time * num_merge_steps,
            max_y_offset,
            facecolor='none',
            edgecolor=colors['max_compute'],
            linewidth=1,
            linestyle='--',
            hatch='\\\\',
        )
        ax2.add_patch(max_rect)

    # HA_Max total
    max_total = sum(phase['integral'] for phase in max_phases)

    # HA_Max formatting
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Memory (GB)')
    ax2.set_title('(b) HA_Max', fontsize=36, pad=15)

    # HA_Max axis limits
    max_x_limit_max = max_x_end + num_steps_to_show * max_step_time + 0.5
    max_y_limit_max = max(double_mem + num_steps_to_show * step_height, base_mem + num_steps_to_show * step_height)
    ax2.set_xlim(0, max_x_limit_max)
    ax2.set_ylim(0, max_y_limit_max)

    # HA_Max legend
    legend_elements_max = [
        mpatches.Patch(facecolor='none', edgecolor=colors['max_rewrite'], hatch='\\\\',
                      label='Rewrite', linestyle='--'),
        mpatches.Patch(facecolor='none', edgecolor=colors['max_prepare'], hatch='\\\\',
                      label='Prepare', linestyle='--'),
        mpatches.Patch(facecolor='none', edgecolor=colors['max_compute'], hatch='\\\\',
                      label='Compute', linestyle='--')
    ]
    ax2.legend(handles=legend_elements_max, loc='upper left', fancybox=True, shadow=True, fontsize=24)

    plt.tight_layout()

    # Save figure
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    output_path = os.path.join(CONFIG['output_dir'], f'{num_seqs}_{seq_length}.{CONFIG["fig_format"]}')
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', format=CONFIG['fig_format'])
    plt.show()
    print(f"Figure saved to: {output_path}")

    return sparse_total, max_total, boundary_steps_float


def create_simple_squares():
    """Create single figure with two subplots comparing both methods"""

    print("=" * 80)
    print("Generating figure with two subplots...")
    print("=" * 80)
    print()

    # Create figure with two subplots
    sparse_total, max_total, boundary_steps_float = create_subfigures()

    # Print comparison results
    print()
    print("=" * 80)
    print(f"Summary - Sequence Length {seq_length}, {num_seqs} Sequences")
    print("=" * 80)
    print(f"HA_Sparse Pre-initialization Total: {sparse_total:.2e} KB·ms")
    print(f"HA_Max Pre-initialization Total: {max_total:.2e} KB·ms")
    print()
    efficiency = (1 - sparse_total / max_total) * 100
    print(f"HA_Sparse is {efficiency:.1f}% more efficient than HA_Max in pre-initialization")
    print(f"Memory growth per decoding step: {step_mem_increase:.3f} KB")
    print(f"HA_Sparse time per step: {sparse_prepare + sparse_compute:.3f} ms")
    print(f"HA_Max time per step: {max_prepare + max_compute:.3f} ms")
    print()
    print(f"Boundary at {boundary_steps_float:.1f} decoding steps")
    print(f"Output directory: {CONFIG['output_dir']}")
    
if __name__ == "__main__":
    create_simple_squares()