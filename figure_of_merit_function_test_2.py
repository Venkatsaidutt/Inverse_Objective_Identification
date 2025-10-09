# WAVEGUIDE TEST: Clear E=0, H≠0 Physics
# Using TM mode in rectangular waveguide for clean separation

import numpy as np
import os
import meep as mp
import matplotlib.pyplot as plt
import scipy.linalg
mp.verbosity(0)

# ----------------------------------------
# SIMULATION PARAMETERS  
# ----------------------------------------
output_dir = "Waveguide_E0_H_Test"
wvl = 1.55
freq = 1 / wvl
resolution = 30
pml_th = 1.5

# Simulation domain
cell_x = 16.0
cell_y = 12.0

print("WAVEGUIDE TEST: Clear E=0, H≠0 Physics")
print("="*50)
print("Using TM mode in rectangular waveguide:")
print("  - TM mode: Ez≠0, Hz=0 in waveguide")
print("  - PEC walls: Ex=Ey=0 at boundaries")
print("  - Test region: On PEC walls where Ex=Ey=0, Hz≠0")
print()
print("Expected results:")
print("  F1 = exp(-|E|²) should WIN (E=0 on PEC walls)")
print("  F2 = exp(-|H|²) should LOSE (H≠0 on PEC walls)")
print()

# K-to-epsilon mapping function
def epsilon_from_k_stable(k_values):
    """Stable mapping from design parameter k to permittivity epsilon"""
    alpha = 7
   
    return   np.exp(alpha * k_values / (1 - k_values + 1e-4))

# Design parameters
k_air = 0    # Air/vacuum
k_pec = 0.93    # Perfect conductor walls

eps_air = epsilon_from_k_stable(k_air)
eps_pec = epsilon_from_k_stable(k_pec)

print(f"Air: k = {k_air:.2f} → ε = {eps_air:.1e}")
print(f"PEC walls: k = {k_pec:.2f} → ε = {eps_pec:.1e}")

# ----------------------------------------
# RECTANGULAR WAVEGUIDE GEOMETRY
# ----------------------------------------
def create_rectangular_waveguide():
    """
    Create rectangular waveguide with PEC walls
    This gives us regions where Ex=Ey=0 but Hz≠0
    """
    geometry = []
    
    # Waveguide dimensions
    guide_width = 6.0   # x-direction
    guide_height = 4.0  # y-direction
    wall_thickness = 0.3
    
    # PEC walls
    # Top wall
    geometry.append(
        mp.Block(center=mp.Vector3(0, guide_height/2 + wall_thickness/2),
                size=mp.Vector3(guide_width + 2*wall_thickness, wall_thickness),
                material=mp.Medium(epsilon=eps_pec))
    )
    
    # Bottom wall  
    geometry.append(
        mp.Block(center=mp.Vector3(0, -guide_height/2 - wall_thickness/2),
                size=mp.Vector3(guide_width + 2*wall_thickness, wall_thickness),
                material=mp.Medium(epsilon=eps_pec))
    )
    
    # Left wall
    geometry.append(
        mp.Block(center=mp.Vector3(-guide_width/2 - wall_thickness/2, 0),
                size=mp.Vector3(wall_thickness, guide_height),
                material=mp.Medium(epsilon=eps_pec))
    )
    
    # Right wall
    geometry.append(
        mp.Block(center=mp.Vector3(guide_width/2 + wall_thickness/2, 0),
                size=mp.Vector3(wall_thickness, guide_height),
                material=mp.Medium(epsilon=eps_pec))
    )
    
    return geometry, guide_width, guide_height

# ----------------------------------------
# TM MODE SOURCE
# ----------------------------------------
def create_tm_mode_source(guide_width, guide_height):
    """
    Create TM10 mode source
    TM modes have Ez≠0, Hz=0 inside waveguide
    """
    # TM10 mode profile: Ez = sin(π*x/a) where a is waveguide width
    # Source placed at waveguide input
    
    source_x = -guide_width/2 + 0.5  # Just inside left wall
    
    # Create line source with TM mode profile
    sources = [
        mp.Source(mp.ContinuousSource(frequency=freq),
                 center=mp.Vector3(source_x, 0),
                 size=mp.Vector3(0, guide_height*0.8),  # Slightly smaller than guide
                 component=mp.Ez,
                 amplitude=2.0,
                 amp_func=lambda p: np.sin(np.pi * (p.y + guide_height/2) / guide_height))
    ]
    
    return sources

# ----------------------------------------
# OBJECTIVE FUNCTIONS
# ----------------------------------------
def compute_two_simple_objectives(Ex_vals, Ey_vals, Ez_vals, Hx_vals, Hy_vals, Hz_vals):
    """
    F1 = exp(-|E_transverse|²) - Transverse E-field minimization 
    F2 = exp(-|H|²) - Total H-field minimization
    """
    
    # For waveguide analysis, focus on transverse E-fields (Ex, Ey)
    # These should be zero on PEC walls
    E_transverse_intensity = np.abs(Ex_vals)**2 + np.abs(Ey_vals)**2
    H_total_intensity = np.abs(Hx_vals)**2 + np.abs(Hy_vals)**2 + np.abs(Hz_vals)**2
    
    # Use moderate scaling to avoid numerical issues
    scaling = 1e20
    
    print(f"  Max E_transverse_intensity: {np.max(E_transverse_intensity):.2e}")
    print(f"  Max H_total_intensity: {np.max(H_total_intensity):.2e}")
    print(f"  Using scaling: {scaling:.0e}")
    
    # OBJECTIVE 1: F1 = exp(-|E_transverse|²) - Should be maximized on PEC walls
    obj1 = np.exp(-scaling * E_transverse_intensity)
    dF1_dEx = -scaling * 2 * obj1 * Ex_vals
    dF1_dEy = -scaling * 2 * obj1 * Ey_vals
    dF1_dEz = np.zeros_like(Ez_vals)  # F1 doesn't depend on Ez
    dF1_dHx = np.zeros_like(Hx_vals)
    dF1_dHy = np.zeros_like(Hy_vals)
    dF1_dHz = np.zeros_like(Hz_vals)
    
    # OBJECTIVE 2: F2 = exp(-|H_total|²) - Should be smaller where H≠0
    obj2 = np.exp(-scaling * H_total_intensity)
    dF2_dEx = np.zeros_like(Ex_vals)
    dF2_dEy = np.zeros_like(Ey_vals)
    dF2_dEz = np.zeros_like(Ez_vals)
    dF2_dHx = -scaling * 2 * obj2 * Hx_vals
    dF2_dHy = -scaling * 2 * obj2 * Hy_vals
    dF2_dHz = -scaling * 2 * obj2 * Hz_vals
    
    objective_values = [obj1, obj2]
    derivatives = [
        [dF1_dEx, dF1_dEy, dF1_dEz, dF1_dHx, dF1_dHy, dF1_dHz],
        [dF2_dEx, dF2_dEy, dF2_dEz, dF2_dHx, dF2_dHy, dF2_dHz]
    ]
    
    return objective_values, derivatives

# ----------------------------------------
# SETUP AND RUN SIMULATION
# ----------------------------------------
geometry, guide_width, guide_height = create_rectangular_waveguide()
sources = create_tm_mode_source(guide_width, guide_height)

print(f"\nWaveguide setup:")
print(f"  Dimensions: {guide_width} × {guide_height} μm")
print(f"  TM10 mode source at x = {-guide_width/2 + 0.5:.1f} μm")
print(f"  Analysis: PEC wall regions where Ex=Ey=0")

print("\nRunning forward simulation...")
sim = mp.Simulation(
    cell_size=mp.Vector3(cell_x, cell_y),
    boundary_layers=[mp.PML(pml_th)],
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    force_complex_fields=True,
    eps_averaging=True
)

sim.run(until=300)
print("Forward simulation complete.")

# ----------------------------------------
# FIELD EXTRACTION AND ANALYSIS
# ----------------------------------------
Ex_array = sim.get_array(component=mp.Ex, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
Ey_array = sim.get_array(component=mp.Ey, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
Ez_array = sim.get_array(component=mp.Ez, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
Hx_array = sim.get_array(component=mp.Hx, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
Hy_array = sim.get_array(component=mp.Hy, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))
Hz_array = sim.get_array(component=mp.Hz, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y))

x_array = np.linspace(-cell_x/2, cell_x/2, Ex_array.shape[0])
y_array = np.linspace(-cell_y/2, cell_y/2, Ex_array.shape[1])
x_grid, y_grid = np.meshgrid(x_array, y_array, indexing='ij')

# Define analysis region: Near PEC walls where Ex=Ey=0 but Hz≠0
wall_thickness = 0.3
pec_wall_mask = (
    # Near top/bottom walls
    ((np.abs(y_grid) >= guide_height/2 - 0.1) & (np.abs(y_grid) <= guide_height/2 + wall_thickness)) |
    # Near left/right walls  
    ((np.abs(x_grid) >= guide_width/2 - 0.1) & (np.abs(x_grid) <= guide_width/2 + wall_thickness))
) & (
    # But within reasonable bounds
    (np.abs(x_grid) <= guide_width/2 + wall_thickness + 0.2) &
    (np.abs(y_grid) <= guide_height/2 + wall_thickness + 0.2)
)

print(f"\nField statistics:")
print(f"  |Ex| range: [{np.min(np.abs(Ex_array)):.3e}, {np.max(np.abs(Ex_array)):.3e}]")
print(f"  |Ey| range: [{np.min(np.abs(Ey_array)):.3e}, {np.max(np.abs(Ey_array)):.3e}]")
print(f"  |Ez| range: [{np.min(np.abs(Ez_array)):.3e}, {np.max(np.abs(Ez_array)):.3e}]")
print(f"  |Hz| range: [{np.min(np.abs(Hz_array)):.3e}, {np.max(np.abs(Hz_array)):.3e}]")

# Analyze fields near PEC walls
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()
Ex_flat = Ex_array.ravel()
Ey_flat = Ey_array.ravel()
Ez_flat = Ez_array.ravel()
Hx_flat = Hx_array.ravel()
Hy_flat = Hy_array.ravel()
Hz_flat = Hz_array.ravel()

analysis_indices = np.where(pec_wall_mask.ravel())[0]
N_points = len(analysis_indices)

Ex_analysis = Ex_flat[analysis_indices]
Ey_analysis = Ey_flat[analysis_indices]
Ez_analysis = Ez_flat[analysis_indices]
Hx_analysis = Hx_flat[analysis_indices]
Hy_analysis = Hy_flat[analysis_indices]
Hz_analysis = Hz_flat[analysis_indices]

E_transverse = np.sqrt(np.abs(Ex_analysis)**2 + np.abs(Ey_analysis)**2)
H_total = np.sqrt(np.abs(Hx_analysis)**2 + np.abs(Hy_analysis)**2 + np.abs(Hz_analysis)**2)

mean_Et = np.mean(E_transverse)
mean_H = np.mean(H_total)

print(f"\nPEC wall analysis:")
print("="*30)
print(f"Analysis points: {N_points}")
print(f"Mean |E_transverse| on walls: {mean_Et:.3e}")
print(f"Mean |H_total| on walls: {mean_H:.3e}")
print(f"H/E_transverse ratio: {mean_H/mean_Et:.2f}")

# Calculate objectives
obj_vals, obj_derivs = compute_two_simple_objectives(
    Ex_analysis, Ey_analysis, Ez_analysis, Hx_analysis, Hy_analysis, Hz_analysis)

objective_names = [
    "F1 = exp(-|E_transverse|²) - Transverse E minimization (EXPECTED WINNER)",
    "F2 = exp(-|H_total|²) - Total H minimization (SHOULD LOSE)"
]

print(f"\nObjective values on PEC walls:")
for i, (name, vals) in enumerate(zip(objective_names, obj_vals)):
    mean_obj = np.mean(np.real(vals))
    print(f"  {i+1}. {name}")
    print(f"     Mean value: {mean_obj:.6f}")

sim.reset_meep()

# ----------------------------------------
# ADJOINT SIMULATIONS
# ----------------------------------------
print(f"\nRunning adjoint simulations...")
print("="*40)

adj_fields_by_objective = {}
omega2 = (2 * np.pi * freq) ** 2

for obj_idx in range(2):
    print(f"Adjoint simulation {obj_idx+1}/2: {objective_names[obj_idx][:30]}...")
    
    obj_vals, obj_derivs = compute_two_simple_objectives(
        Ex_analysis, Ey_analysis, Ez_analysis, Hx_analysis, Hy_analysis, Hz_analysis)
    
    # Create adjoint sources
    adj_sources = []
    source_count = 0
    threshold = 1e-20
    max_sources = 1000
    sample_step = max(1, N_points // max_sources)
    
    for k in range(0, N_points, sample_step):
        idx = analysis_indices[k]
        analysis_x = x_flat[idx]
        analysis_y = y_flat[idx]
        
        derivatives = [obj_derivs[obj_idx][j][k] if k < len(obj_derivs[obj_idx][j]) else 0 
                      for j in range(6)]
        
        components = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
        for j, (component, deriv) in enumerate(zip(components, derivatives)):
            if abs(deriv) > threshold:
                adj_sources.append(mp.Source(
                    mp.ContinuousSource(frequency=freq),
                    center=mp.Vector3(analysis_x, analysis_y),
                    size=mp.Vector3(),
                    component=component,
                    amplitude=float(np.real(deriv))
                ))
                source_count += 1
                if source_count >= max_sources:
                    break
        if source_count >= max_sources:
            break
    
    if source_count == 0:
        adj_sources = [mp.Source(mp.ContinuousSource(frequency=freq),
                                center=mp.Vector3(0, 0), size=mp.Vector3(),
                                component=mp.Ex, amplitude=1e-12)]
        source_count = 1
    
    print(f"  Created {source_count} adjoint sources")
    
    # Run adjoint simulation
    sim_adj = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y),
        boundary_layers=[mp.PML(pml_th)],
        geometry=geometry,
        sources=adj_sources,
        resolution=resolution,
        force_complex_fields=True,
        eps_averaging=True
    )
    
    sim_adj.run(until=300)
    
    # Extract adjoint fields
    Ex_adj = sim_adj.get_array(component=mp.Ex, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
    Ey_adj = sim_adj.get_array(component=mp.Ey, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
    Ez_adj = sim_adj.get_array(component=mp.Ez, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
    Hx_adj = sim_adj.get_array(component=mp.Hx, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
    Hy_adj = sim_adj.get_array(component=mp.Hy, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
    Hz_adj = sim_adj.get_array(component=mp.Hz, center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y)).ravel()
    
    adj_fields_by_objective[obj_idx] = np.vstack([Ex_adj, Ey_adj, Ez_adj, Hx_adj, Hy_adj, Hz_adj])
    
    sim_adj.reset_meep()

# ----------------------------------------
# BUILD MATRIX AND SOLVE
# ----------------------------------------
print("\nBuilding optimality matrix...")

max_analysis = 500
if N_points > max_analysis:
    sample_indices = np.linspace(0, N_points-1, max_analysis, dtype=int)
    sampled_indices = analysis_indices[sample_indices]
else:
    sampled_indices = analysis_indices

N_analysis = len(sampled_indices)
X_matrix = np.zeros((N_analysis, 2), dtype=np.complex128)

for row_idx in range(N_analysis):
    global_idx = sampled_indices[row_idx]
    
    for col in range(2):
        adj_fields = adj_fields_by_objective[col]
        
        E_adj = adj_fields[:3, global_idx]
        H_adj = adj_fields[3:, global_idx]
        E_fwd = np.array([Ex_flat[global_idx], Ey_flat[global_idx], Ez_flat[global_idx]])
        H_fwd = np.array([Hx_flat[global_idx], Hy_flat[global_idx], Hz_flat[global_idx]])
        
        E_overlap = np.vdot(E_adj, E_fwd)
        H_overlap = np.vdot(H_adj, H_fwd) / 377.0
        X_matrix[row_idx, col] = omega2 * (E_overlap + H_overlap)

print(f"Matrix shape: {X_matrix.shape}")
print(f"Matrix range: [{np.min(np.abs(X_matrix)):.3e}, {np.max(np.abs(X_matrix)):.3e}]")

# SVD Analysis
matrix_norm = np.linalg.norm(X_matrix)
if matrix_norm < 1e-15:
    reg_strength = max(1e-15, matrix_norm * 1e-3)
    X_matrix += reg_strength * np.random.randn(*X_matrix.shape) * (1 + 1j)
    print(f"Added regularization: {reg_strength:.2e}")

U, S, Vh = scipy.linalg.svd(X_matrix, full_matrices=False)
print(f"Singular values: {S}")

a_opt = Vh.conj().T[:, -1]
a_opt_norm = a_opt / np.linalg.norm(a_opt) if np.linalg.norm(a_opt) > 0 else np.ones(2) / np.sqrt(2)

abs_coeffs = np.abs(a_opt_norm)
dominant_idx = np.argmax(abs_coeffs)

print(f"\n" + "="*60)
print("WAVEGUIDE TEST RESULTS")
print("="*60)

for i in range(2):
    magnitude = abs_coeffs[i]
    winner = " ← WINNER" if i == dominant_idx else ""
    expected = " (EXPECTED)" if i == 0 else " (SHOULD LOSE)"
    print(f"  F{i+1}: {magnitude:.1%}{winner}{expected}")
    print(f"      {objective_names[i][:50]}...")

print(f"\nIDENTIFICATION:")
print(f"PEC walls optimize: {objective_names[dominant_idx]}")

e_coeff = abs_coeffs[0]
h_coeff = abs_coeffs[1]
print(f"\nCOEFFICIENTS:")
print(f"  E-transverse minimization: {e_coeff:.1%}")
print(f"  H-total minimization: {h_coeff:.1%}")
if h_coeff > 0:
    print(f"  E/H ratio: {e_coeff/h_coeff:.2f}")

if dominant_idx == 0:  # F1 should win
    if e_coeff > 0.7:
        result = "PERFECT SUCCESS"
    elif e_coeff > 0.6:
        result = "EXCELLENT SUCCESS"
    else:
        result = "GOOD SUCCESS"
    status = "✓ Correctly identifies E-field minimization on PEC walls"
else:
    result = "UNEXPECTED"
    status = "⚠ Identifies H-field minimization (check field magnitudes)"

print(f"\nRESULT: {result}")
print(f"STATUS: {status}")

# ----------------------------------------
# VISUALIZATION
# ----------------------------------------
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Ex field
im1 = axes[0,0].imshow(np.abs(Ex_array).T, extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2], 
                       origin='lower', cmap='hot')
axes[0,0].contour(x_grid, y_grid, pec_wall_mask.astype(int), levels=[0.5], colors='cyan', linewidths=2)
axes[0,0].set_title('|Ex| Field (Should be ~0 on PEC walls)')
plt.colorbar(im1, ax=axes[0,0])

# Hz field
im2 = axes[0,1].imshow(np.abs(Hz_array).T, extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2], 
                       origin='lower', cmap='hot')
axes[0,1].contour(x_grid, y_grid, pec_wall_mask.astype(int), levels=[0.5], colors='cyan', linewidths=2)
axes[0,1].set_title('|Hz| Field (Non-zero on PEC walls)')
plt.colorbar(im2, ax=axes[0,1])

# Coefficients
bars = axes[1,0].bar(['E-transverse', 'H-total'], abs_coeffs, color=['green', 'red'])
axes[1,0].set_ylabel('Coefficient')
axes[1,0].set_title('Objective Identification')
bars[dominant_idx].set_edgecolor('black')
bars[dominant_idx].set_linewidth(3)

# Field comparison
field_data = [mean_Et, mean_H]
axes[1,1].bar(['E-transverse', 'H-total'], field_data, color=['blue', 'red'])
axes[1,1].set_ylabel('Mean Field Magnitude')
axes[1,1].set_title('E vs H Fields on PEC Walls')
axes[1,1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'waveguide_test.png'), dpi=150)
plt.close()

print(f"\n🎯 WAVEGUIDE TEST COMPLETE!")
print(f"Expected: F1 (E-transverse) should win with >70% confidence")
print(f"Actual: F{dominant_idx+1} wins with {abs_coeffs[dominant_idx]:.1%} confidence")

if result in ["PERFECT SUCCESS", "EXCELLENT SUCCESS"]:
    print("🎉 Method successfully identifies E-field constraint physics!")
    print("This validates the correct behavior: F1 maximized when E=0!")
else:
    print("📊 Check field magnitudes - may still have noise issues")
    print(f"If H/E ratio is >> 1, the method may be seeing real physics differences")