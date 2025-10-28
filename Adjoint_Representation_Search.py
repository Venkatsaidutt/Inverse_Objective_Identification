import meep as mp
import numpy as np
from scipy.linalg import null_space
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

# ============================================
# PART 1: FORWARD SIMULATION
# ============================================

sx, sy = 10, 3
resolution = 40
wvg_width = 0.5
Si = mp.Medium(epsilon=12.0)

geometry = [mp.Block(size=mp.Vector3(sx, wvg_width, 0),
                     center=mp.Vector3(0, 0, 0),
                     material=Si)]

fcen = 1/1.55
df = 0.1*fcen

sources = [mp.EigenModeSource(
    src=mp.GaussianSource(fcen, fwidth=df),
    center=mp.Vector3(-4, 0, 0),
    size=mp.Vector3(0, sy, 0),
    eig_band=1
)]

cell = mp.Vector3(sx, sy, 0)
pml_layers = [mp.PML(1.0)]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    boundary_layers=pml_layers
)

design_region = mp.Volume(center=mp.Vector3(0, 0, 0), 
                          size=mp.Vector3(4, 1.0, 0))

sim.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ez, mp.Vector3(4, 0, 0), 1e-3))

Ex = sim.get_array(center=design_region.center, size=design_region.size, component=mp.Ex)
Ey = sim.get_array(center=design_region.center, size=design_region.size, component=mp.Ey)
Ez = sim.get_array(center=design_region.center, size=design_region.size, component=mp.Ez)

grid_shape = Ex.shape
N_points = Ex.size

print(f"Grid shape: {grid_shape}")
print(f"Total points: {N_points}")

E_fwd_matrix = np.column_stack([Ex.flatten(), Ey.flatten(), Ez.flatten()])
e_fwd_flat = E_fwd_matrix.flatten()

# ============================================
# PART 2: DESIGN REGION MASK
# ============================================

y_coords = np.linspace(-design_region.size.y/2, design_region.size.y/2, grid_shape[1])
design_mask = np.abs(y_coords) < wvg_width/2

design_mask_2d = np.ones(grid_shape, dtype=bool)
design_mask_2d[:, ~design_mask] = False

design_mask_3N = np.repeat(design_mask_2d.flatten(), 3)

print(f"Design region: {np.sum(design_mask_2d)} points ({100*np.sum(design_mask_2d)/N_points:.1f}%)")

# ============================================
# PART 3: WEIGHTED CONSTRAINT & NULL SPACE
# ============================================

weights = np.ones(len(e_fwd_flat))
weights[~design_mask_3N] = 0

e_fwd_weighted = e_fwd_flat * weights

A_constraint = e_fwd_weighted.reshape(1, -1)

print(f"\nComputing null space...")
null_basis = null_space(A_constraint)

n_modes = null_basis.shape[1]
print(f"Null space dimension: {n_modes}")

# ============================================
# PART 4: DIVERGENCE-FREE FILTER
# ============================================

def compute_divergence_2d(E_field_3d, dx):
    Ex, Ey, Ez = E_field_3d[:,:,0], E_field_3d[:,:,1], E_field_3d[:,:,2]
    dEx_dx = np.gradient(Ex, dx, axis=0)
    dEy_dy = np.gradient(Ey, dx, axis=1)
    return dEx_dx + dEy_dy

def check_divergence_free(e_adj_flat, grid_shape, dx, tolerance=0.1):
    E_adj_3d = e_adj_flat.reshape(grid_shape + (3,))
    div_E = compute_divergence_2d(E_adj_3d, dx)
    div_norm = np.linalg.norm(div_E)
    field_norm = np.linalg.norm(E_adj_3d)
    return (div_norm / field_norm) < tolerance, div_norm / field_norm

dx = 1.0 / resolution

print(f"\nFiltering divergence-free modes...")

valid_mode_indices = []
batch_size = 1000

for batch_idx in range((n_modes + batch_size - 1) // batch_size):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, n_modes)
    
    for i in range(start_idx, end_idx):
        is_valid, _ = check_divergence_free(null_basis[:, i], grid_shape, dx, 0.1)
        if is_valid:
            valid_mode_indices.append(i)
    
    print(f"  Batch {batch_idx+1}: checked {end_idx}/{n_modes}, valid: {len(valid_mode_indices)}")

print(f"\n✓ Valid divergence-free modes: {len(valid_mode_indices)}")

# ============================================
# PART 5: SOURCE LOCALIZATION - ALL MODES
# ============================================

def analyze_source_localization(E_adj_3d, omega, epsilon, dx, grid_shape):
    """Compute source and analyze localization metrics"""
    Ez_adj = E_adj_3d[:,:,2]
    
    # Maxwell inverse (Fourier space)
    Ez_k = np.fft.fft2(Ez_adj)
    kx = 2*np.pi*np.fft.fftfreq(grid_shape[0], d=dx)
    ky = 2*np.pi*np.fft.fftfreq(grid_shape[1], d=dx)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    laplacian_k = -(Kx**2 + Ky**2)
    k_squared = omega**2 * epsilon
    
    # Regularized inversion
    J_k = (laplacian_k * Ez_k + k_squared * Ez_k) / (1j * omega + 1e-6)
    J_z = np.fft.ifft2(J_k)
    
    J_mag = np.abs(J_z)
    
    # Peak strength
    J_max = np.max(J_mag)
    
    # Spatial spread
    com_y, com_x = center_of_mass(J_mag)
    
    y_indices, x_indices = np.meshgrid(range(grid_shape[1]), range(grid_shape[0]), indexing='ij')
    
    var_x = np.sum(J_mag * (x_indices.T - com_x)**2) / (np.sum(J_mag) + 1e-12)
    var_y = np.sum(J_mag * (y_indices.T - com_y)**2) / (np.sum(J_mag) + 1e-12)
    
    spread_x = np.sqrt(var_x) * dx
    spread_y = np.sqrt(var_y) * dx
    
    # Localization ratio
    threshold_90 = 0.9 * J_max
    localized_energy = np.sum(J_mag[J_mag > threshold_90])
    total_energy = np.sum(J_mag) + 1e-12
    localization_ratio = localized_energy / total_energy
    
    return {
        'J_max': J_max,
        'spread_x': spread_x,
        'spread_y': spread_y,
        'localization_ratio': localization_ratio,
        'center_of_mass': (com_x * dx, com_y * dx)
    }

omega = 2 * np.pi * fcen
epsilon_Si = 12.0

print(f"\n" + "="*70)
print(f"SOURCE LOCALIZATION ANALYSIS - ALL {len(valid_mode_indices)} MODES")
print("="*70)

# Process ALL valid modes
print(f"\nAnalyzing all {len(valid_mode_indices)} divergence-free modes...")
print("(This will take a few minutes)")

all_J_max = []
all_spread_x = []
all_spread_y = []
all_loc_ratio = []

update_interval = 200

for idx_count, mode_idx in enumerate(valid_mode_indices):
    e_adj = null_basis[:, mode_idx]
    E_adj_3d = e_adj.reshape(grid_shape + (3,))
    
    analysis = analyze_source_localization(E_adj_3d, omega, epsilon_Si, dx, grid_shape)
    
    all_J_max.append(analysis['J_max'])
    all_spread_x.append(analysis['spread_x'])
    all_spread_y.append(analysis['spread_y'])
    all_loc_ratio.append(analysis['localization_ratio'])
    
    if (idx_count + 1) % update_interval == 0:
        print(f"  Progress: {idx_count + 1}/{len(valid_mode_indices)} "
              f"({100*(idx_count+1)/len(valid_mode_indices):.1f}%)")

print(f"\n✓ Source analysis complete for all modes")

# Convert to arrays
all_J_max = np.array(all_J_max)
all_spread_x = np.array(all_spread_x)
all_spread_y = np.array(all_spread_y)
all_loc_ratio = np.array(all_loc_ratio)

# ============================================
# PART 6: FILTERING BY LOCALIZATION
# ============================================

print(f"\n" + "="*70)
print("FILTERING RESULTS - COMPLETE DATASET")
print("="*70)

# Criteria
J_max_threshold = 0.1 * np.max(all_J_max)
spread_min = 0.1  # microns
spread_max = 2.0  # microns
loc_ratio_min = 0.2

strong_mask = all_J_max > J_max_threshold
spread_x_mask = (all_spread_x > spread_min) & (all_spread_x < spread_max)
spread_y_mask = (all_spread_y > spread_min) & (all_spread_y < spread_max)
localized_mask = all_loc_ratio > loc_ratio_min

# Combined filter
good_modes_mask = strong_mask & spread_x_mask & spread_y_mask & localized_mask

n_good = np.sum(good_modes_mask)

print(f"Total divergence-free modes: {len(valid_mode_indices)}")
print(f"\nIndividual criteria:")
print(f"  Strong source (>{J_max_threshold:.2e}): {np.sum(strong_mask)} ({100*np.sum(strong_mask)/len(valid_mode_indices):.1f}%)")
print(f"  Good spread X ({spread_min}-{spread_max} μm): {np.sum(spread_x_mask)} ({100*np.sum(spread_x_mask)/len(valid_mode_indices):.1f}%)")
print(f"  Good spread Y ({spread_min}-{spread_max} μm): {np.sum(spread_y_mask)} ({100*np.sum(spread_y_mask)/len(valid_mode_indices):.1f}%)")
print(f"  High localization (>{loc_ratio_min}): {np.sum(localized_mask)} ({100*np.sum(localized_mask)/len(valid_mode_indices):.1f}%)")
print(f"\n✓ Combined (all criteria): {n_good} modes ({100*n_good/len(valid_mode_indices):.1f}%)")

# ============================================
# PART 7: STATISTICAL PLOTS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of J_max
axes[0,0].hist(np.log10(all_J_max + 1e-12), bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0,0].axvline(np.log10(J_max_threshold), color='red', linestyle='--', linewidth=2.5, 
                  label=f'Threshold (10%)')
axes[0,0].set_xlabel('log₁₀(J_max)', fontsize=12)
axes[0,0].set_ylabel('Count', fontsize=12)
axes[0,0].set_title(f'Source Strength Distribution (N={len(all_J_max)})', 
                    fontsize=13, fontweight='bold')
axes[0,0].legend(fontsize=11)
axes[0,0].grid(alpha=0.3)

# Scatter: spread vs strength
sc = axes[0,1].scatter(all_spread_x, all_J_max, c=all_loc_ratio, 
                      cmap='viridis', alpha=0.5, s=20)
axes[0,1].axhline(J_max_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Strength threshold')
axes[0,1].axvline(spread_min, color='orange', linestyle='--', linewidth=2, alpha=0.7)
axes[0,1].axvline(spread_max, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Spread bounds')
axes[0,1].set_xlabel('Spread in x (μm)', fontsize=12)
axes[0,1].set_ylabel('J_max', fontsize=12)
axes[0,1].set_yscale('log')
axes[0,1].set_title('Strength vs Spread', fontsize=13, fontweight='bold')
axes[0,1].legend(fontsize=10, loc='upper right')
cbar = plt.colorbar(sc, ax=axes[0,1])
cbar.set_label('Localization ratio', fontsize=11)
axes[0,1].grid(alpha=0.3)

# Histogram of localization ratio
axes[1,0].hist(all_loc_ratio, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1,0].axvline(loc_ratio_min, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Threshold ({loc_ratio_min})')
axes[1,0].set_xlabel('Localization ratio', fontsize=12)
axes[1,0].set_ylabel('Count', fontsize=12)
axes[1,0].set_title('Localization Distribution', fontsize=13, fontweight='bold')
axes[1,0].legend(fontsize=11)
axes[1,0].grid(alpha=0.3)

# 2D histogram: spread_x vs spread_y
h = axes[1,1].hist2d(all_spread_x, all_spread_y, bins=30, cmap='Blues', cmin=1)
axes[1,1].axhline(spread_min, color='red', linestyle='--', linewidth=2, alpha=0.8)
axes[1,1].axhline(spread_max, color='red', linestyle='--', linewidth=2, alpha=0.8)
axes[1,1].axvline(spread_min, color='red', linestyle='--', linewidth=2, alpha=0.8)
axes[1,1].axvline(spread_max, color='red', linestyle='--', linewidth=2, alpha=0.8)
axes[1,1].set_xlabel('Spread in x (μm)', fontsize=12)
axes[1,1].set_ylabel('Spread in y (μm)', fontsize=12)
axes[1,1].set_title('Spatial Spread Distribution', fontsize=13, fontweight='bold')
plt.colorbar(h[3], ax=axes[1,1], label='Count')

plt.tight_layout()
plt.savefig('complete_source_statistics.png', dpi=200, bbox_inches='tight')
print(f"\n✓ Statistics saved: complete_source_statistics.png")


print(f"\n" + "="*70)
print(" FILTERING - REDUCING TO ACTIONABLE SET")
print("="*70)

# Get indices of good modes
good_mode_indices = np.array(valid_mode_indices)[good_modes_mask]

print(f"Starting with {len(good_mode_indices)} well-localized modes")

# ============================================
# FILTER 7: FINER SPATIAL CLUSTERING
# ============================================

print(f"\n--- Filter 7: Spatial Clustering ---")

# Extract center of mass for ALL good modes (we need this)
print(f"Extracting centers of mass for {len(good_mode_indices)} modes...")

centers_of_mass = []
J_max_values = []

update_interval = 500

for count, idx in enumerate(good_mode_indices):
    e_adj = null_basis[:, idx]
    E_adj_3d = e_adj.reshape(grid_shape + (3,))
    Ez_adj = E_adj_3d[:,:,2]
    
    # Compute source
    Ez_k = np.fft.fft2(Ez_adj)
    kx = 2*np.pi*np.fft.fftfreq(grid_shape[0], d=dx)
    ky = 2*np.pi*np.fft.fftfreq(grid_shape[1], d=dx)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    laplacian_k = -(Kx**2 + Ky**2)
    k_squared = omega**2 * epsilon_Si
    
    J_k = (laplacian_k * Ez_k + k_squared * Ez_k) / (1j * omega + 1e-6)
    J_z = np.fft.ifft2(J_k)
    J_mag = np.abs(J_z)
    
    com_y, com_x = center_of_mass(J_mag)
    centers_of_mass.append([com_x * dx, com_y * dx])
    J_max_values.append(np.max(J_mag))
    
    if (count + 1) % update_interval == 0:
        print(f"  Processed {count + 1}/{len(good_mode_indices)}...")

centers_of_mass = np.array(centers_of_mass)
J_max_values = np.array(J_max_values)

print(f"✓ Centers extracted")

# Finer clustering (smaller eps for single waveguide)
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.05, min_samples=1)  # 0.05 micron radius (much finer)
cluster_labels = clustering.fit_predict(centers_of_mass)

n_clusters = len(set(cluster_labels))
print(f"Found {n_clusters} spatial clusters (eps=0.05 μm)")

# Keep strongest mode from each cluster
clustered_mode_indices = []
clustered_strengths = []

for cluster_id in set(cluster_labels):
    cluster_mask = cluster_labels == cluster_id
    cluster_members = good_mode_indices[cluster_mask]
    member_strengths = J_max_values[cluster_mask]
    
    # Keep strongest
    strongest_local_idx = np.argmax(member_strengths)
    strongest_idx = cluster_members[strongest_local_idx]
    
    clustered_mode_indices.append(strongest_idx)
    clustered_strengths.append(member_strengths[strongest_local_idx])

print(f"After clustering: {len(clustered_mode_indices)} representative modes")

# ============================================
# FILTER 8: SPATIAL DISTRIBUTION ANALYSIS
# ============================================

print(f"\n--- Filter 8: Spatial Distribution Analysis ---")

# For a straight waveguide, we expect sources at:
# - Left boundary (input reflection)
# - Right boundary (output transmission)  
# - Or distributed along length (absorption, phase)

# Analyze x-position distribution
clustered_centers = centers_of_mass[np.isin(good_mode_indices, clustered_mode_indices)]

x_positions = clustered_centers[:, 0]
y_positions = clustered_centers[:, 1]

# Convert to absolute coordinates
x_abs = x_positions + design_region.center.x - design_region.size.x/2

# Categorize by position
left_boundary = design_region.center.x - design_region.size.x/2
right_boundary = design_region.center.x + design_region.size.x/2

boundary_threshold = 0.3  # Within 0.3 μm of boundary

left_modes = []
right_modes = []
interior_modes = []

for i, (idx, x) in enumerate(zip(clustered_mode_indices, x_abs)):
    if x < (left_boundary + boundary_threshold):
        left_modes.append(idx)
    elif x > (right_boundary - boundary_threshold):
        right_modes.append(idx)
    else:
        interior_modes.append(idx)

print(f"Spatial distribution:")
print(f"  Left boundary (<{boundary_threshold} μm): {len(left_modes)}")
print(f"  Right boundary (>{design_region.size.x - boundary_threshold} μm): {len(right_modes)}")
print(f"  Interior (middle): {len(interior_modes)}")

# Combine boundary modes (these are likely the physical FOMs)
boundary_modes = left_modes + right_modes

print(f"\nKeeping all categories for completeness: {len(clustered_mode_indices)} modes")

# ============================================
# FILTER 9: PATTERN DIVERSITY
# ============================================

print(f"\n--- Filter 9: Pattern Diversity (Remove Near-Duplicates) ---")

# Extract source patterns for clustered modes
source_patterns = []

print(f"Extracting patterns for {len(clustered_mode_indices)} modes...")

for count, idx in enumerate(clustered_mode_indices):
    e_adj = null_basis[:, idx]
    E_adj_3d = e_adj.reshape(grid_shape + (3,))
    Ez_adj = E_adj_3d[:,:,2]
    
    # Compute source
    Ez_k = np.fft.fft2(Ez_adj)
    kx = 2*np.pi*np.fft.fftfreq(grid_shape[0], d=dx)
    ky = 2*np.pi*np.fft.fftfreq(grid_shape[1], d=dx)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    laplacian_k = -(Kx**2 + Ky**2)
    k_squared = omega**2 * epsilon_Si
    
    J_k = (laplacian_k * Ez_k + k_squared * Ez_k) / (1j * omega + 1e-6)
    J_z = np.fft.ifft2(J_k)
    
    # Flatten and normalize
    J_flat = J_z.flatten()
    J_normalized = J_flat / (np.linalg.norm(J_flat) + 1e-12)
    
    source_patterns.append(J_normalized)

source_patterns = np.array(source_patterns)

# Compute pairwise correlations
correlations = np.abs(source_patterns @ source_patterns.conj().T)

# Remove highly similar patterns
similarity_threshold = 0.90  # Keep if less than 90% similar
unique_indices = [0]  # Always keep first

for i in range(1, len(clustered_mode_indices)):
    is_unique = True
    for j in unique_indices:
        if correlations[i, j] > similarity_threshold:
            is_unique = False
            break
    
    if is_unique:
        unique_indices.append(i)

final_mode_indices = [clustered_mode_indices[i] for i in unique_indices]

print(f"Unique patterns (correlation < {similarity_threshold}): {len(final_mode_indices)}")

# ============================================
# FINAL VISUALIZATION
# ============================================

print(f"\n" + "="*70)
print(f"VISUALIZING FINAL {len(final_mode_indices)} FOM CANDIDATES")
print("="*70)

n_viz = min(12, len(final_mode_indices))

if n_viz > 0:
    ncols = min(6, n_viz)
    nrows = (n_viz + ncols - 1) // ncols
    
    fig, axes = plt.subplots(2*nrows, ncols, figsize=(3*ncols, 4*nrows))
    axes = axes.reshape(2*nrows, ncols)  # Ensure 2D
    
    for plot_idx in range(n_viz):
        row = 2 * (plot_idx // ncols)
        col = plot_idx % ncols
        
        mode_idx = final_mode_indices[plot_idx]
        
        # Reconstruct
        e_adj = null_basis[:, mode_idx]
        E_adj_3d = e_adj.reshape(grid_shape + (3,))
        Ez_adj = E_adj_3d[:,:,2]
        
        Ez_k = np.fft.fft2(Ez_adj)
        kx = 2*np.pi*np.fft.fftfreq(grid_shape[0], d=dx)
        ky = 2*np.pi*np.fft.fftfreq(grid_shape[1], d=dx)
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        
        laplacian_k = -(Kx**2 + Ky**2)
        k_squared = omega**2 * epsilon_Si
        
        J_k = (laplacian_k * Ez_k + k_squared * Ez_k) / (1j * omega + 1e-6)
        J_z = np.fft.ifft2(J_k)
        J_mag = np.abs(J_z)
        
        # Adjoint field
        ax1 = axes[row, col]
        im1 = ax1.imshow(np.abs(Ez_adj).T, cmap='hot', origin='lower', aspect='auto')
        ax1.set_title(f'FOM {plot_idx+1}: |E_adj|', fontsize=9, fontweight='bold')
        ax1.axis('off')
        
        # Source
        ax2 = axes[row+1, col]
        im2 = ax2.imshow(J_mag.T, cmap='viridis', origin='lower', aspect='auto')
        
        com_y, com_x = center_of_mass(J_mag)
        ax2.plot(com_y, com_x, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='white')
        ax2.set_title(f'|J| source', fontsize=9)
        ax2.axis('off')
    
    # Hide unused subplots
    for plot_idx in range(n_viz, nrows * ncols):
        row = 2 * (plot_idx // ncols)
        col = plot_idx % ncols
        axes[row, col].axis('off')
        axes[row+1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('final_FOM_candidates.png', dpi=200, bbox_inches='tight')
    print(f"✓ Visualization saved: final_FOM_candidates.png")

# ============================================
# ULTIMATE SUMMARY
# ============================================

print(f"\n" + "="*70)
print("COMPLETE FILTERING PIPELINE SUMMARY")
print("="*70)
print(f"  Stage 0: Total DOF (3N)               = {len(e_fwd_flat)}")
print(f"  Stage 1: Null space (gradient)        = {n_modes} ({100*n_modes/len(e_fwd_flat):.1f}%)")
print(f"  Stage 2: Divergence-free              = {len(valid_mode_indices)} ({100*len(valid_mode_indices)/n_modes:.1f}%)")
print(f"  Stage 3: Strong & localized            = {np.sum(good_modes_mask)} ({100*np.sum(good_modes_mask)/len(valid_mode_indices):.1f}%)")
print(f"  Stage 4: Spatial clustering (0.05μm)   = {len(clustered_mode_indices)} ({100*len(clustered_mode_indices)/np.sum(good_modes_mask):.1f}%)")
print(f"  Stage 5: Pattern uniqueness (90%)      = {len(final_mode_indices)} ({100*len(final_mode_indices)/len(clustered_mode_indices):.1f}%)")
print(f"\n  ════════════════════════════════════════════════")
print(f"  ➤ FINAL ACTIONABLE FOM CANDIDATES: {len(final_mode_indices)}")
print(f"  ════════════════════════════════════════════════")
print("="*70)

print(f"\nFor single-mode waveguide, these represent:")
print(f"  - Power transmission/reflection")
print(f"  - Different coupling configurations")
print(f"  - Absorption/loss patterns")
print(f"  - Phase/dispersion optimization")
print("="*70)
