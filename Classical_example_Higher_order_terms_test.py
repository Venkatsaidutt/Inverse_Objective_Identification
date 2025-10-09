#!/usr/bin/env python3
"""
Enhanced Inverse Objective Identification with:
- Multiple k_pec values (0.2, 0.3, 0.4, 0.5)
- Higher-order exponential terms (E^2, E^4, E^6, E^8)
- Perturbation robustness analysis
- Comprehensive visualization
"""

import numpy as np
import os
import meep as mp
import matplotlib.pyplot as plt
import scipy.linalg
from datetime import datetime
import time

mp.verbosity(0)

# ============================================================================
# ENHANCED PARAMETERS
# ============================================================================
K_PEC_VALUES = [0.2, 0.3, 0.4, 0.5]  # Multiple shielding strengths
HIGHER_ORDER_POWERS = [2, 4, 6, 8]    # E^2n functionals
BASE_DIR = "Enhanced_Inverse_Analysis"
os.makedirs(BASE_DIR, exist_ok=True)

wvl = 1.55
freq = 1 / wvl
resolution = 30
pml_th = 1.5

cell_x = 16.0
cell_y = 12.0

cage_outer_width = 4.0
cage_outer_height = 4.0
wall_thickness = 0.1
cage_inner_width = cage_outer_width - 2*wall_thickness
cage_inner_height = cage_outer_height - 2*wall_thickness

def epsilon_from_k_stable(k):
    """Stable epsilon mapping from FOM_Buffer"""
    alpha = 5
    k_safe = np.clip(k, 0.1, 0.92)
    return 1.0 + 50 * np.exp(alpha * k_safe / (1 - k_safe + 1e-3))

# ============================================================================
# PERTURBATION CONFIGURATIONS (from reference)
# ============================================================================
perturbations = {
    'corner_block': {
        'center': (cage_outer_width/2 - 0.1, cage_outer_height/2 - 0.1),
        'size': (0.15, 0.15),
        'description': 'Corner block',
        'color': 'red'
    },
    'wall_extension': {
        'center': (-cage_outer_width/2 - 0.075, 0),
        'size': (0.15, 0.3),
        'description': 'Wall extension',
        'color': 'orange'
    },
    'top_addition': {
        'center': (0, cage_outer_height/2 + 0.075),
        'size': (0.3, 0.15),
        'description': 'Top addition',
        'color': 'purple'
    }
}

# ============================================================================
# ENHANCED OBJECTIVE FUNCTIONS WITH HIGHER-ORDER TERMS
# ============================================================================
def compute_higher_order_objectives(Ex_vals, Ey_vals, Ez_vals, Hx_vals, Hy_vals, Hz_vals, power=4):
    """
    Compute F = exp(-alpha * |E|^(2*power)) for different powers
    From FOM_Buffer Section 3.3: higher-order intensity functionals
    """
    eps_reg = 1e-25
    E_intensity = np.abs(Ex_vals)**2 + np.abs(Ey_vals)**2 + np.abs(Ez_vals)**2 + eps_reg
    
    # Adaptive alpha based on power to maintain numerical stability
    if power == 1:  # E^2
        alpha = 1e5
    elif power == 2:  # E^4
        alpha = 1e6
    elif power == 3:  # E^6
        alpha = 1e7
    elif power == 4:  # E^8
        alpha = 1e8
    else:
        alpha = 1e6
    
    # F = exp(-alpha * |E|^(2n))
    E_power_n = E_intensity ** power
    obj = np.exp(-alpha * E_power_n)
    
    # Derivatives using chain rule
    dobj_dE_intensity = -alpha * power * (E_intensity ** (power - 1)) * obj
    
    dF_dEx = 2 * dobj_dE_intensity * Ex_vals
    dF_dEy = 2 * dobj_dE_intensity * Ey_vals
    dF_dEz = 2 * dobj_dE_intensity * Ez_vals
    dF_dHx = np.zeros_like(Hx_vals)
    dF_dHy = np.zeros_like(Hy_vals)
    dF_dHz = np.zeros_like(Hz_vals)
    
    derivatives = [dF_dEx, dF_dEy, dF_dEz, dF_dHx, dF_dHy, dF_dHz]
    
    return obj, derivatives

# ============================================================================
# GEOMETRY CREATION
# ============================================================================
def create_geometry_with_perturbation(eps_pec, perturbation_config=None):
    """Create MEEP geometry with optional perturbation"""
    
    geometry = [
        mp.Block(center=mp.Vector3(-cage_outer_width/2 + wall_thickness/2, 0),
                 size=mp.Vector3(wall_thickness, cage_outer_height),
                 material=mp.Medium(epsilon=eps_pec)),
        mp.Block(center=mp.Vector3(cage_outer_width/2 - wall_thickness/2, 0),
                 size=mp.Vector3(wall_thickness, cage_outer_height),
                 material=mp.Medium(epsilon=eps_pec)),
        mp.Block(center=mp.Vector3(0, -cage_outer_height/2 + wall_thickness/2),
                 size=mp.Vector3(cage_inner_width, wall_thickness),
                 material=mp.Medium(epsilon=eps_pec)),
        mp.Block(center=mp.Vector3(0, cage_outer_height/2 - wall_thickness/2),
                 size=mp.Vector3(cage_inner_width, wall_thickness),
                 material=mp.Medium(epsilon=eps_pec))
    ]
    
    if perturbation_config is not None:
        cx, cy = perturbation_config['center']
        sx, sy = perturbation_config['size']
        
        geometry.append(mp.Block(
            center=mp.Vector3(cx, cy),
            size=mp.Vector3(sx, sy),
            material=mp.Medium(epsilon=eps_pec)
        ))
    
    return geometry

# ============================================================================
# DEFINE ANALYSIS REGIONS
# ============================================================================
def define_three_analysis_regions(x_grid, y_grid):
    """Define three 4x4 μm analysis regions from FOM_Buffer"""
    region_size = 2.0  # Half-width of 4x4 μm region
    
    regions = {}
    
    # Region 1: INSIDE Faraday cage (center)
    inside_mask = (
        (np.abs(x_grid - 0.0) <= region_size) & 
        (np.abs(y_grid - 0.0) <= region_size)
    )
    regions['inside_cage'] = inside_mask
    
    # Region 2: OUTSIDE cage (left side)
    left_center_x = -6.0
    outside_left_mask = (
        (np.abs(x_grid - left_center_x) <= region_size) & 
        (np.abs(y_grid - 0.0) <= region_size)
    )
    regions['outside_left'] = outside_left_mask
    
    # Region 3: OUTSIDE cage (right side)  
    right_center_x = 6.0
    outside_right_mask = (
        (np.abs(x_grid - right_center_x) <= region_size) & 
        (np.abs(y_grid - 0.0) <= region_size)
    )
    regions['outside_right'] = outside_right_mask
    
    return regions

# ============================================================================
# FULL ANALYSIS PIPELINE
# ============================================================================
def run_full_analysis_multi_order(geometry, k_pec, label, powers=HIGHER_ORDER_POWERS):
    """
    Run complete analysis with multiple higher-order functionals
    Each power corresponds to F = exp(-alpha * |E|^(2*power))
    """
    
    start_time = time.time()
    
    sources = [
        mp.Source(mp.ContinuousSource(frequency=freq),
                  center=mp.Vector3(-cell_x/2 + 3.0, 0),
                  size=mp.Vector3(0, cell_y - 2*pml_th),
                  component=mp.Ex, amplitude=2.0)
    ]
    
    # Forward simulation
    print(f"  Running forward simulation for k_pec={k_pec}, {label}...")
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y),
        boundary_layers=[mp.PML(pml_th)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        force_complex_fields=True,
        eps_averaging=True
    )
    
    sim.run(until=400)
    
    # Extract fields
    Ex_array = sim.get_array(component=mp.Ex, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
    Ey_array = sim.get_array(component=mp.Ey, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
    Ez_array = sim.get_array(component=mp.Ez, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
    Hx_array = sim.get_array(component=mp.Hx, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
    Hy_array = sim.get_array(component=mp.Hy, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
    Hz_array = sim.get_array(component=mp.Hz, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
    
    x_array = np.linspace(-cell_x/2, cell_x/2, Ex_array.shape[0])
    y_array = np.linspace(-cell_y/2, cell_y/2, Ex_array.shape[1])
    x_grid, y_grid = np.meshgrid(x_array, y_array, indexing='ij')
    
    # Define regions
    regions = define_three_analysis_regions(x_grid, y_grid)
    
    # Flatten
    Ex_flat = Ex_array.ravel()
    Ey_flat = Ey_array.ravel()
    Ez_flat = Ez_array.ravel()
    Hx_flat = Hx_array.ravel()
    Hy_flat = Hy_array.ravel()
    Hz_flat = Hz_array.ravel()
    
    sim.reset_meep()
    
    # Adjoint simulations for each region AND each power
    region_names = ['inside_cage', 'outside_left', 'outside_right']
    num_functionals = len(region_names) * len(powers)
    
    print(f"  Running {num_functionals} adjoint simulations ({len(region_names)} regions × {len(powers)} powers)...")
    
    adj_fields_dict = {}  # Key: (region, power)
    omega2 = (2 * np.pi * freq) ** 2
    
    for region_name in region_names:
        mask = regions[region_name]
        indices = np.where(mask.ravel())[0]
        
        for power in powers:
            print(f"    Adjoint: {region_name}, E^{2*power}...")
            
            # Sample adjoint sources
            max_sources = 500
            sample_step = max(1, len(indices) // max_sources)
            
            adj_sources = []
            x_flat = x_grid.ravel()
            y_flat = y_grid.ravel()
            
            for k in range(0, len(indices), sample_step):
                idx = indices[k]
                
                # Compute objective derivative for this power
                Ex_sample = Ex_flat[idx:idx+1]
                Ey_sample = Ey_flat[idx:idx+1]
                Ez_sample = Ez_flat[idx:idx+1]
                Hx_sample = Hx_flat[idx:idx+1]
                Hy_sample = Hy_flat[idx:idx+1]
                Hz_sample = Hz_flat[idx:idx+1]
                
                _, obj_derivs = compute_higher_order_objectives(
                    Ex_sample, Ey_sample, Ez_sample, 
                    Hx_sample, Hy_sample, Hz_sample, 
                    power=power
                )
                
                dF_dEx = obj_derivs[0][0]
                dF_dEy = obj_derivs[1][0]
                dF_dEz = obj_derivs[2][0]
                
                for comp, deriv in zip([mp.Ex, mp.Ey, mp.Ez], [dF_dEx, dF_dEy, dF_dEz]):
                    if abs(deriv) > 1e-20:
                        adj_sources.append(mp.Source(
                            mp.ContinuousSource(frequency=freq),
                            center=mp.Vector3(x_flat[idx], y_flat[idx]),
                            component=comp,
                            amplitude=float(np.real(deriv))
                        ))
            
            if len(adj_sources) == 0:
                # Fallback source
                center_x = 0.0 if region_name == 'inside_cage' else (-6.0 if 'left' in region_name else 6.0)
                adj_sources = [mp.Source(mp.ContinuousSource(frequency=freq),
                                        center=mp.Vector3(center_x, 0), size=mp.Vector3(),
                                        component=mp.Ex, amplitude=1e-12)]
            
            # Run adjoint
            sim_adj = mp.Simulation(
                cell_size=mp.Vector3(cell_x, cell_y),
                boundary_layers=[mp.PML(pml_th)],
                geometry=geometry,
                sources=adj_sources,
                resolution=resolution,
                force_complex_fields=True,
                eps_averaging=True
            )
            
            sim_adj.run(until=400)
            
            Ex_adj = sim_adj.get_array(component=mp.Ex, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
            Ey_adj = sim_adj.get_array(component=mp.Ey, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
            Ez_adj = sim_adj.get_array(component=mp.Ez, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
            
            adj_fields_dict[(region_name, power)] = (Ex_adj, Ey_adj, Ez_adj)
            sim_adj.reset_meep()
    
    # Build sensitivity matrix M: shape (num_points, num_functionals)
    print(f"  Building sensitivity matrix...")
    num_points = len(Ex_flat)
    M_matrix = np.zeros((num_points, num_functionals), dtype=complex)
    
    func_idx = 0
    for region_name in region_names:
        for power in powers:
            Ex_adj, Ey_adj, Ez_adj = adj_fields_dict[(region_name, power)]
            
            for i in range(num_points):
                E_fwd = np.array([Ex_flat[i], Ey_flat[i], Ez_flat[i]])
                E_adj = np.array([Ex_adj[i], Ey_adj[i], Ez_adj[i]])
                
                E_overlap = np.vdot(E_adj, E_fwd)
                M_matrix[i, func_idx] = omega2 * E_overlap
            
            func_idx += 1
    
    # SVD
    print(f"  Performing SVD...")
    U, S, Vh = scipy.linalg.svd(M_matrix, full_matrices=False)
    s_star = Vh.conj().T[:, -1]
    s_normalized = s_star / np.linalg.norm(s_star)
    s_result = np.abs(s_normalized)
    
    # Reshape s_result into (regions, powers)
    s_matrix = s_result.reshape(len(region_names), len(powers))
    
    # Optimality score
    optimality = 1 - S[-1]/S[0] if S[0] > 0 else 0
    condition_number = S[0]/S[-1] if S[-1] > 0 else np.inf
    
    elapsed = time.time() - start_time
    
    # Field statistics
    E_mag = np.sqrt(np.abs(Ex_array)**2 + np.abs(Ey_array)**2)
    mean_E_inside = np.mean(E_mag[regions['inside_cage']])
    mean_E_outside = 0.5 * (np.mean(E_mag[regions['outside_left']]) + np.mean(E_mag[regions['outside_right']]))
    
    print(f"  ✓ Complete: O={optimality:.4f}, κ={condition_number:.2e}, time={elapsed:.1f}s")
    
    return {
        's_matrix': s_matrix,  # Shape: (3 regions, N powers)
        's_vector': s_result,
        'M_matrix': M_matrix,
        'singular_values': S,
        'optimality': optimality,
        'condition_number': condition_number,
        'forward_fields': (Ex_array, Ey_array, Ez_array),
        'regions': regions,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'mean_E_inside': mean_E_inside,
        'mean_E_outside': mean_E_outside,
        'time': elapsed
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("="*80)
print("ENHANCED INVERSE OBJECTIVE IDENTIFICATION")
print("Multiple k_pec + Higher-Order Functionals + Perturbation Analysis")
print("="*80)
print()

results_by_kpec = {}

for k_pec in K_PEC_VALUES:
    eps_pec = epsilon_from_k_stable(k_pec)
    
    print(f"\n{'='*80}")
    print(f"ANALYZING k_pec = {k_pec:.1f} (ε = {eps_pec:.1f})")
    print(f"{'='*80}")
    
    # Original structure
    geometry_orig = create_geometry_with_perturbation(eps_pec, None)
    result_orig = run_full_analysis_multi_order(geometry_orig, k_pec, "Original")
    
    # Perturbations
    pert_results = {}
    for pert_name, pert_config in perturbations.items():
        print(f"\n  Analyzing perturbation: {pert_config['description']}")
        geometry_pert = create_geometry_with_perturbation(eps_pec, pert_config)
        result_pert = run_full_analysis_multi_order(geometry_pert, k_pec, pert_config['description'])
        pert_results[pert_name] = result_pert
    
    results_by_kpec[k_pec] = {
        'original': result_orig,
        'perturbations': pert_results
    }

# ============================================================================
# COMPREHENSIVE VISUALIZATION
# ============================================================================
print(f"\n{'='*80}")
print("GENERATING COMPREHENSIVE PLOTS")
print(f"{'='*80}")

# Plot 1: Weight matrices for each k_pec (original)
fig1, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()

region_names = ['Inside', 'Left', 'Right']
power_labels = [f'E^{2*p}' for p in HIGHER_ORDER_POWERS]

for idx, k_pec in enumerate(K_PEC_VALUES):
    ax = axes[idx]
    s_matrix = results_by_kpec[k_pec]['original']['s_matrix']
    
    im = ax.imshow(s_matrix.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(region_names)
    ax.set_yticks(range(len(HIGHER_ORDER_POWERS)))
    ax.set_yticklabels(power_labels)
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Functional Power', fontsize=12, fontweight='bold')
    
    eps_pec = epsilon_from_k_stable(k_pec)
    cond = results_by_kpec[k_pec]['original']['condition_number']
    opt = results_by_kpec[k_pec]['original']['optimality']
    
    ax.set_title(f'k_pec={k_pec:.1f}, ε={eps_pec:.1f}\nκ={cond:.2e}, O={opt:.4f}', 
                fontsize=13, fontweight='bold')
    
    # Annotate values
    for i in range(3):
        for j in range(len(HIGHER_ORDER_POWERS)):
            text = ax.text(i, j, f'{s_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

plt.colorbar(im, ax=axes, label='Weight Coefficient', fraction=0.046, pad=0.04)
plt.suptitle('Inverse Objective Identification: Weight Coefficients Across k_pec Values', 
            fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(BASE_DIR, 'weight_matrices_multi_kpec.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: weight_matrices_multi_kpec.png")

# Plot 2: Perturbation comparison for k_pec=0.3
fig2, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()

k_pec_test = 0.3
test_results = results_by_kpec[k_pec_test]

# Original
ax = axes[0]
s_matrix = test_results['original']['s_matrix']
im = ax.imshow(s_matrix.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(region_names)
ax.set_yticks(range(len(HIGHER_ORDER_POWERS)))
ax.set_yticklabels(power_labels)
ax.set_title('Original Structure', fontsize=13, fontweight='bold')
for i in range(3):
    for j in range(len(HIGHER_ORDER_POWERS)):
        ax.text(i, j, f'{s_matrix[i, j]:.3f}', ha="center", va="center", 
               color="black", fontsize=10, fontweight='bold')

# Perturbations
for idx, (pert_name, pert_config) in enumerate(list(perturbations.items())[:3]):
    ax = axes[idx + 1]
    s_matrix = test_results['perturbations'][pert_name]['s_matrix']
    im = ax.imshow(s_matrix.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(region_names)
    ax.set_yticks(range(len(HIGHER_ORDER_POWERS)))
    ax.set_yticklabels(power_labels)
    ax.set_title(f'{pert_config["description"].title()}', fontsize=13, fontweight='bold')
    for i in range(3):
        for j in range(len(HIGHER_ORDER_POWERS)):
            ax.text(i, j, f'{s_matrix[i, j]:.3f}', ha="center", va="center", 
                   color="black", fontsize=10, fontweight='bold')

plt.colorbar(im, ax=axes, label='Weight Coefficient', fraction=0.046, pad=0.04)
plt.suptitle(f'Perturbation Analysis for k_pec={k_pec_test}', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(BASE_DIR, f'perturbation_comparison_kpec_{k_pec_test}.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: perturbation_comparison_kpec_{k_pec_test}.png")

# Plot 3: Summary metrics
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Condition number vs k_pec
ax = axes[0, 0]
cond_nums = [results_by_kpec[k]['original']['condition_number'] for k in K_PEC_VALUES]
ax.semilogy(K_PEC_VALUES, cond_nums, 'o-', linewidth=3, markersize=12, color='darkblue')
ax.set_xlabel('k_pec', fontsize=12, fontweight='bold')
ax.set_ylabel('Condition Number κ(M)', fontsize=12, fontweight='bold')
ax.set_title('Matrix Conditioning vs Shielding Strength', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Optimality score vs k_pec
ax = axes[0, 1]
opt_scores = [results_by_kpec[k]['original']['optimality'] for k in K_PEC_VALUES]
ax.plot(K_PEC_VALUES, opt_scores, 's-', linewidth=3, markersize=12, color='darkgreen')
ax.set_xlabel('k_pec', fontsize=12, fontweight='bold')
ax.set_ylabel('Optimality Score O', fontsize=12, fontweight='bold')
ax.set_title('Optimality Score vs Shielding Strength', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

# Field suppression
ax = axes[1, 0]
E_in = [results_by_kpec[k]['original']['mean_E_inside'] for k in K_PEC_VALUES]
E_out = [results_by_kpec[k]['original']['mean_E_outside'] for k in K_PEC_VALUES]
width = 0.035
x = np.array(K_PEC_VALUES)
ax.bar(x - width/2, E_in, width, label='Inside Cage', color='cyan', edgecolor='black', linewidth=2)
ax.bar(x + width/2, E_out, width, label='Outside Cage', color='orange', edgecolor='black', linewidth=2)
ax.set_xlabel('k_pec', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean |E|', fontsize=12, fontweight='bold')
ax.set_title('Field Suppression by Shielding', fontsize=13, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Dominant functional power
ax = axes[1, 1]
for region_idx, region_name in enumerate(region_names):
    dominant_powers = []
    for k_pec in K_PEC_VALUES:
        s_matrix = results_by_kpec[k_pec]['original']['s_matrix']
        dominant_power_idx = np.argmax(s_matrix[region_idx, :])
        dominant_powers.append(2 * HIGHER_ORDER_POWERS[dominant_power_idx])
    ax.plot(K_PEC_VALUES, dominant_powers, 'o-', linewidth=2, markersize=10, label=region_name)

ax.set_xlabel('k_pec', fontsize=12, fontweight='bold')
ax.set_ylabel('Dominant Functional Power', fontsize=12, fontweight='bold')
ax.set_title('Dominant E^n Power by Region', fontsize=13, fontweight='bold')
ax.set_yticks([2, 4, 6, 8])
ax.set_yticklabels(['E^2', 'E^4', 'E^6', 'E^8'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle('Comprehensive Analysis Summary', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(BASE_DIR, 'summary_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: summary_metrics.png")

# ============================================================================
# SAVE DATA
# ============================================================================
np.savez(
    os.path.join(BASE_DIR, 'enhanced_results.npz'),
    results=results_by_kpec,
    k_pec_values=K_PEC_VALUES,
    powers=HIGHER_ORDER_POWERS,
    timestamp=datetime.now().isoformat()
)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"Results saved to: {BASE_DIR}/")
print(f"Generated plots:")
print(f"  • weight_matrices_multi_kpec.png")
print(f"  • perturbation_comparison_kpec_0.3.png")
print(f"  • summary_metrics.png")
print(f"  • enhanced_results.npz")
print()
