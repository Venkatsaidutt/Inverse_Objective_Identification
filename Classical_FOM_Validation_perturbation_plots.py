
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import os

# ============================================================================
# PARAMETERS (match your validation code)
# ============================================================================
OUTPUT_DIR = "Spatial_Discrimination_Study"

# Geometry
cell_x = 16.0
cell_y = 12.0
cage_outer_width = 4.0
cage_outer_height = 4.0
wall_thickness = 0.1
cage_inner_width = cage_outer_width - 2*wall_thickness
cage_inner_height = cage_outer_height - 2*wall_thickness
pml_th = 1.5

# ============================================================================
# CREATE PERTURBATION LOCATION DIAGRAM
# ============================================================================

def create_perturbation_diagram():
    """Create diagram showing where perturbations are added"""
    
    print("Creating perturbation location diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim([-cell_x/2 - 1, cell_x/2 + 1])
    ax.set_ylim([-cell_y/2 - 1, cell_y/2 + 1])
    ax.set_aspect('equal')
    
    # ========================================================================
    # Draw Faraday cage (original structure)
    # ========================================================================
    cage_color = 'darkgreen'
    cage_alpha = 0.7
    
    # Cage walls
    ax.add_patch(Rectangle((-cage_outer_width/2, -cage_outer_height/2), 
                           wall_thickness, cage_outer_height,
                           facecolor=cage_color, edgecolor='black', 
                           linewidth=2, alpha=cage_alpha, label='Faraday Cage'))
    ax.add_patch(Rectangle((cage_outer_width/2-wall_thickness, -cage_outer_height/2), 
                           wall_thickness, cage_outer_height,
                           facecolor=cage_color, edgecolor='black', 
                           linewidth=2, alpha=cage_alpha))
    ax.add_patch(Rectangle((-cage_inner_width/2, -cage_outer_height/2), 
                           cage_inner_width, wall_thickness,
                           facecolor=cage_color, edgecolor='black', 
                           linewidth=2, alpha=cage_alpha))
    ax.add_patch(Rectangle((-cage_inner_width/2, cage_outer_height/2-wall_thickness), 
                           cage_inner_width, wall_thickness,
                           facecolor=cage_color, edgecolor='black', 
                           linewidth=2, alpha=cage_alpha))
    
    # ========================================================================
    # Draw three analysis regions (dashed)
    # ========================================================================
    region_size = 2.0
    regions_setup = [
        (0, 0, 'cyan', 'Inside Cage'),
        (-6, 0, 'yellow', 'Outside Left'),
        (6, 0, 'magenta', 'Outside Right')
    ]
    
    for x_center, y_center, color, label in regions_setup:
        ax.add_patch(FancyBboxPatch((x_center-region_size, y_center-region_size), 
                                    2*region_size, 2*region_size,
                                    boxstyle="round,pad=0.05", 
                                    facecolor='none', edgecolor=color, 
                                    linewidth=2, linestyle='--', alpha=0.6))
    
    # ========================================================================
    # PERTURBATION 1: Corner block
    # ========================================================================
    corner_x = cage_outer_width/2 - 0.1
    corner_y = cage_outer_height/2 - 0.1
    corner_size_x = 0.15
    corner_size_y = 0.15
    
    corner_block = Rectangle((corner_x - corner_size_x/2, corner_y - corner_size_y/2),
                             corner_size_x, corner_size_y,
                             facecolor='red', edgecolor='darkred',
                             linewidth=3, alpha=0.8, hatch='///')
    ax.add_patch(corner_block)
    
    # Annotation for corner block
    ax.annotate('Corner Block\nPerturbation',
                xy=(corner_x, corner_y),
                xytext=(corner_x + 1.5, corner_y + 1.5),
                fontsize=11, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='darkred', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5))
    
    # Dimension arrow for corner block
    ax.annotate('', xy=(corner_x + corner_size_x/2, corner_y - corner_size_y/2 - 0.15),
                xytext=(corner_x - corner_size_x/2, corner_y - corner_size_y/2 - 0.15),
                arrowprops=dict(arrowstyle='<->', color='darkred', lw=1.5))
    ax.text(corner_x, corner_y - corner_size_y/2 - 0.35, 
            f'{corner_size_x} μm', ha='center', fontsize=9, color='darkred', fontweight='bold')
    
    # ========================================================================
    # PERTURBATION 2: Wall extension
    # ========================================================================
    wall_x = -cage_outer_width/2 - 0.075
    wall_y = 0
    wall_size_x = 0.15
    wall_size_y = 0.3
    
    wall_extension = Rectangle((wall_x - wall_size_x/2, wall_y - wall_size_y/2),
                               wall_size_x, wall_size_y,
                               facecolor='orangered', edgecolor='darkred',
                               linewidth=3, alpha=0.8, hatch='\\\\\\')
    ax.add_patch(wall_extension)
    
    # Annotation for wall extension
    ax.annotate('Wall Extension\nPerturbation',
                xy=(wall_x, wall_y),
                xytext=(wall_x - 2.0, wall_y + 2.0),
                fontsize=11, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='darkred', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5))
    
    # Dimension arrows for wall extension
    ax.annotate('', xy=(wall_x - wall_size_x/2 - 0.15, wall_y + wall_size_y/2),
                xytext=(wall_x - wall_size_x/2 - 0.15, wall_y - wall_size_y/2),
                arrowprops=dict(arrowstyle='<->', color='darkred', lw=1.5))
    ax.text(wall_x - wall_size_x/2 - 0.45, wall_y, 
            f'{wall_size_y} μm', rotation=90, va='center', fontsize=9, 
            color='darkred', fontweight='bold')
    
    # ========================================================================
    # Source
    # ========================================================================
    source_x = -cell_x/2 + 3.0
    ax.plot([source_x, source_x], [-cell_y/2+pml_th+0.5, cell_y/2-pml_th-0.5], 
           'r-', linewidth=4, label='Source')
    ax.arrow(source_x+0.3, 0, 1.5, 0, head_width=0.4, head_length=0.3, 
            fc='red', ec='red', linewidth=2)
    
    # ========================================================================
    # PML boundaries
    # ========================================================================
    pml_color = 'gray'
    ax.axvline(-cell_x/2+pml_th, color=pml_color, linestyle=':', linewidth=2, alpha=0.5)
    ax.axvline(cell_x/2-pml_th, color=pml_color, linestyle=':', linewidth=2, alpha=0.5)
    ax.axhline(-cell_y/2+pml_th, color=pml_color, linestyle=':', linewidth=2, alpha=0.5)
    ax.axhline(cell_y/2-pml_th, color=pml_color, linestyle=':', linewidth=2, alpha=0.5)
    ax.text(-cell_x/2+pml_th-0.5, cell_y/2-0.5, 'PML', 
           ha='center', fontsize=10, color=pml_color, style='italic')
    
    # ========================================================================
    # Labels and legend
    # ========================================================================
    ax.set_xlabel('x (μm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('y (μm)', fontsize=14, fontweight='bold')
    ax.set_title('Perturbation Test Locations for Robustness Analysis', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Custom legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=cage_color, edgecolor='black', 
                 linewidth=2, alpha=cage_alpha, label='Original Faraday Cage'),
        Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='darkred', 
                 linewidth=2, alpha=0.8, hatch='///', label='Corner Block (0.15×0.15 μm)'),
        Rectangle((0, 0), 1, 1, facecolor='orangered', edgecolor='darkred', 
                 linewidth=2, alpha=0.8, hatch='\\\\\\', label='Wall Extension (0.15×0.3 μm)'),
        Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', 
               label='Analysis Regions'),
        Line2D([0], [0], marker='>', color='red', markersize=12, linewidth=3,
               linestyle='-', label='Plane Wave Source'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
             framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Text box with explanation
    explanation = (
        "PERTURBATION TEST:\n\n"
        "Two material additions test weight\n"
        "vector stability:\n\n"
        "1. Corner block: 0.15×0.15 μm\n"
        "2. Wall extension: 0.15×0.3 μm\n\n"
        "Each perturbation requires:\n"
        "• Forward simulation\n"
        "• Three adjoint simulations\n"
        "• Sensitivity matrix rebuild\n"
        "• SVD analysis"
    )
    ax.text(0.98, 0.02, explanation, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, 
                    edgecolor='black', linewidth=2),
           fontfamily='monospace')
    
    plt.tight_layout()
    perturbation_file = os.path.join(OUTPUT_DIR, 'perturbation_locations.png')
    plt.savefig(perturbation_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Perturbation location diagram saved: {perturbation_file}")
    return perturbation_file

if __name__ == "__main__":
    create_perturbation_diagram()