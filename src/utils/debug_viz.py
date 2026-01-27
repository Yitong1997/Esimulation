"""
Debug Visualization Module for Hybrid Optical Simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from numpy.typing import NDArray

def fit_low_order_zernike(
    phase: NDArray, 
    x: Optional[NDArray] = None, 
    y: Optional[NDArray] = None,
    mask: Optional[NDArray] = None
) -> dict:
    """
    Fit low order Zernike polynomials to phase data.
    
    Terms: Piston, Tilt X, Tilt Y, Defocus, Astigmatism 0/45, Coma X/Y, Spherical.
    """
    if phase.ndim == 2:
        ny, nx = phase.shape
        if x is None or y is None:
            yp, xp = np.indices((ny, nx))
            # Normalize to -1..1 unit circle approx
            y = (yp - ny/2) / (ny/2)
            x = (xp - nx/2) / (nx/2)
        
        if mask is None:
            r2 = x**2 + y**2
            mask = r2 <= 1.0
            
        # Flatten
        phase_flat = phase[mask]
        x_flat = x[mask]
        y_flat = y[mask]
    else:
        # Scatter data
        if x is None or y is None:
            raise ValueError("x and y must be provided for scatter data")
        
        phase_flat = phase
        x_flat = x
        y_flat = y
        
        # Normalize if not already normalized (assuming approximate centering)
        # This is a bit risky for scatter data without knowing aperture, 
        # but for debug purposes let's assume inputs are roughly centered 
        # or just fit polynomials on whatever coordinates are given.
        # Ideally, x/y should be normalized coordinates.
        if mask is not None:
            phase_flat = phase_flat[mask]
            x_flat = x_flat[mask]
            y_flat = y_flat[mask]

    # Zernike Polynomials (Standard/Noll-like ordering but explicit)
    # Z1: Piston = 1
    # Z2: Tilt X = x (or rho cos theta)
    # Z3: Tilt Y = y
    # Z4: Defocus = 2r^2 - 1
    # Z5: Astigmatism 0/90 = x^2 - y^2
    # Z6: Astigmatism 45 = 2xy
    # Z7: Coma X = (3r^2 - 2)x
    # Z8: Coma Y = (3r^2 - 2)y
    # Z9: Spherical = 6r^4 - 6r^2 + 1
    
    r2 = x_flat**2 + y_flat**2
    
    Z = []
    Z.append(np.ones_like(x_flat))        # Piston
    Z.append(x_flat)                      # Tilt X
    Z.append(y_flat)                      # Tilt Y
    Z.append(2*r2 - 1)                    # Defocus
    Z.append(x_flat**2 - y_flat**2)       # Astig 0
    Z.append(2*x_flat*y_flat)             # Astig 45
    Z.append((3*r2 - 2)*x_flat)           # Coma X
    Z.append((3*r2 - 2)*y_flat)           # Coma Y
    Z.append(6*r2**2 - 6*r2 + 1)          # Spherical
    
    A = np.column_stack(Z)
    
    coeffs, _, _, _ = np.linalg.lstsq(A, phase_flat, rcond=None)
    
    names = [
        "Piston", "Tilt X", "Tilt Y", "Defocus", 
        "Astig 0", "Astig 45", "Coma X", "Coma Y", "Spherical"
    ]
    
    return {name: coeff for name, coeff in zip(names, coeffs)}

def format_zernike_title(title: str, coeffs: dict) -> str:
    """Format Zernike coefficients into the plot title."""
    lines = [title]
    
    # Group coeffs for compact display
    lines.append(f"Tilt: X={coeffs['Tilt X']:.3g}, Y={coeffs['Tilt Y']:.3g}")
    lines.append(f"Defocus: {coeffs['Defocus']:.3g}, Sph: {coeffs['Spherical']:.3g}")
    lines.append(f"Astig: 0={coeffs['Astig 0']:.3g}, 45={coeffs['Astig 45']:.3g}")
    lines.append(f"Coma: X={coeffs['Coma X']:.3g}, Y={coeffs['Coma Y']:.3g}")
    
    return "\n".join(lines)

def plot_rays_2d(
    x: NDArray, 
    y: NDArray, 
    L: NDArray, 
    M: NDArray, 
    title: str = "Ray Distribution"
):
    """Plot ray positions and direction vectors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot of positions
    sc = ax1.scatter(x, y, s=5, alpha=0.6)
    ax1.set_title(f"{title} - Positions")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Quiver plot of directions
    # Subsample if too many rays
    if len(x) > 500:
        idx = np.random.choice(len(x), 500, replace=False)
        x_sub, y_sub = x[idx], y[idx]
        L_sub, M_sub = L[idx], M[idx]
    else:
        x_sub, y_sub = x, y
        L_sub, M_sub = L, M
        
    # Normalize vectors for visualization (arrows show direction only)
    # Avoid division by zero
    mag = np.sqrt(L_sub**2 + M_sub**2)
    # Handle zero magnitude (rays strictly along Z)
    # If magnitude is 0, we can't draw a direction in 2D projection meaningfully
    # except maybe as a dot (which scatter does). 
    # Let's silence potential warnings and set direction to 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        L_norm = np.where(mag > 1e-10, L_sub / mag, 0)
        M_norm = np.where(mag > 1e-10, M_sub / mag, 0)
        
    # Use quiver with fixed scale
    # scale: Number of data units per arrow length unit
    # pivot='mid' centers arrow on the point
    Q = ax2.quiver(
        x_sub, y_sub, 
        L_norm, M_norm, 
        pivot='mid',
        color='blue',
        width=0.003,
        scale=20, # Adjust scale as needed (smaller scale = longer arrows)
        headwidth=4,
        headlength=5
    )
    
    # Add a key/legend? Not strictly necessary for normalized arrows unless length meant something.
    
    ax2.scatter(x_sub, y_sub, s=2, c='red', alpha=0.4) # Dots at origins
    ax2.set_title(f"{title} - Directions (Normalized)")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_phase(
    phase: NDArray, 
    title: str = "Phase",
    x: Optional[NDArray] = None,
    y: Optional[NDArray] = None,
    mask: Optional[NDArray] = None
):
    """Plot phase map with Zernike fitting."""
    
    # 1. Fit Zernikes
    # Use provided coordinates relative to center
    if x is not None and y is not None:
        # Normalize coords for fitting if not already
        r_max = np.max(np.sqrt(x**2 + y**2))
        x_norm = x / r_max if r_max > 0 else x
        y_norm = y / r_max if r_max > 0 else y
        coeffs = fit_low_order_zernike(phase, x_norm, y_norm, mask)
    else:
        coeffs = fit_low_order_zernike(phase, mask=mask)
        
    full_title = format_zernike_title(title, coeffs)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    if phase.ndim == 2:
        # Image plot
        im = ax.imshow(phase, cmap='jet', origin='lower')
        plt.colorbar(im, ax=ax, label='Phase (rad or waves)')
    else:
        # Scatter plot for irregular data (rays)
        sc = ax.scatter(x, y, c=phase, cmap='jet', s=10)
        plt.colorbar(sc, ax=ax, label='Phase (rad or waves)')
        ax.set_aspect('equal')
    
    ax.set_title(full_title)
    plt.tight_layout()
    plt.show()

def plot_comparison(
    ground_truth: NDArray,
    estimated: NDArray,
    title: str = "Comparison"
):
    """Plot comparison between two 2D arrays (e.g., Reconstructed vs Actual)."""
    
    diff = estimated - ground_truth
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(ground_truth, cmap='jet', origin='lower')
    axes[0].set_title("Ground Truth / Reference")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(estimated, cmap='jet', origin='lower')
    axes[1].set_title("Estimated / Reconstructed")
    plt.colorbar(im1, ax=axes[1])
    
    # Diff with symmetric colorbar centered at 0
    vmax = np.max(np.abs(diff))
    im2 = axes[2].imshow(diff, cmap='seismic', origin='lower', vmin=-vmax, vmax=vmax)
    axes[2].set_title("Difference")
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_opd_increment(
    x: NDArray, 
    y: NDArray, 
    opd_inc: NDArray, 
    sign_t: Optional[NDArray] = None,
    step_name: str = "OPD Increment"
):
    """Plot OPD increment and sign of t for raytracing steps."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot OPD Increment
    sc1 = axes[0].scatter(x, y, c=opd_inc, cmap='jet', s=10)
    axes[0].set_title(f"{step_name}")
    plt.colorbar(sc1, ax=axes[0])
    axes[0].axis('equal')
    
    # Plot Sign(t) if provided
    if sign_t is not None:
        sc2 = axes[1].scatter(x, y, c=sign_t, cmap='coolwarm', s=10)
        axes[1].set_title(f"Sign(t) - Direction Check")
        plt.colorbar(sc2, ax=axes[1], ticks=[-1, 0, 1])
        axes[1].axis('equal')
    else:
        axes[1].axis('off')
        
    plt.tight_layout()
    plt.show()
