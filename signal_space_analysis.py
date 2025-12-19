"""
Communications II - Signal Space Analysis Project
EECS 316 Fall 25-26
Cairo University - Faculty of Engineering

This module implements Gram-Schmidt orthogonalization and signal space analysis
for communication systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


def Basis_Cal(Signals, n):
    """
    Part 1.1: Gram-Schmidt Orthogonalization
    
    Calculates the Gram-Schmidt orthonormal basis functions (phi1, phi2, ..., phim)
    for two or more input signals.
    
    Parameters:
    -----------
    Signals : numpy.ndarray
        n x N matrix where each row is a signal (1 x N vector)
        n: number of signals
        N: length of each signal
    n : int
        Number of signals
    
    Returns:
    --------
    Phis : numpy.ndarray
        m x N matrix containing the orthonormal basis functions
        m: number of basis functions (m <= n)
    m : int
        Number of basis functions
    """
    # Get dimensions
    n_signals, N = Signals.shape
    
    # Initialize list to store basis functions
    basis_functions = []
    
    # First basis function: normalize the first signal
    s1 = Signals[0, :]
    norm_s1 = np.linalg.norm(s1)
    
    if norm_s1 < 1e-10:  # Check if signal is zero
        # Skip zero signals
        pass
    else:
        phi1 = s1 / norm_s1
        basis_functions.append(phi1)
    
    # Process remaining signals
    for i in range(1, n_signals):
        si = Signals[i, :]
        
        # Subtract projections onto all previous basis functions
        si_orthogonal = si.copy()
        for phi in basis_functions:
            # Projection coefficient
            proj_coeff = np.dot(si, phi)
            # Subtract projection
            si_orthogonal = si_orthogonal - proj_coeff * phi
        
        # Normalize to get new basis function
        norm_si_orth = np.linalg.norm(si_orthogonal)
        
        if norm_si_orth > 1e-10:  # Only add if not zero (linearly independent)
            phi_new = si_orthogonal / norm_si_orth
            basis_functions.append(phi_new)
    
    # Convert to numpy array
    if len(basis_functions) == 0:
        m = 0
        Phis = np.array([]).reshape(0, N)
    else:
        m = len(basis_functions)
        Phis = np.array(basis_functions)
    
    return Phis, m


def Signal_Rep(Phis, signal):
    """
    Part 1.2: Signal Space Representation
    
    Calculates the signal space representation (coefficients) for a given signal
    using orthonormal basis functions.
    
    Parameters:
    -----------
    Phis : numpy.ndarray
        m x N matrix containing orthonormal basis functions
    signal : numpy.ndarray
        1 x N vector representing the signal
    
    Returns:
    --------
    signal_vector : numpy.ndarray
        1 x m vector containing the signal space coefficients
    """
    m, N = Phis.shape
    
    # Calculate coefficients by projecting signal onto each basis function
    signal_vector = np.zeros(m)
    for i in range(m):
        signal_vector[i] = np.dot(signal, Phis[i, :])
    
    return signal_vector


def Decision_boundaries(Phis, Signals):
    """
    Part 1.3: Decision Boundaries
    
    Draws the decision boundaries for the given signals using orthonormal basis functions.
    Only works for m = 1 or m = 2 dimensions.
    
    Parameters:
    -----------
    Phis : numpy.ndarray
        m x N matrix containing orthonormal basis functions
    Signals : numpy.ndarray
        n x N matrix where each row is a signal
    """
    m, N = Phis.shape
    n, _ = Signals.shape
    
    if m > 2:
        print(f"Decision boundaries can only be drawn for m=1 or m=2. Current m={m}")
        return
    
    # Convert signals to signal space representation
    signal_vectors = []
    for i in range(n):
        vec = Signal_Rep(Phis, Signals[i, :])
        signal_vectors.append(vec)
    signal_vectors = np.array(signal_vectors)
    
    if m == 1:
        # 1D case: plot on a line
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Plot signal points
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        for i in range(n):
            ax.scatter(signal_vectors[i, 0], 0, s=100, c=[colors[i]], 
                      label=f'Signal {i+1}', marker='o', edgecolors='black', linewidths=1.5)
        
        # Draw decision boundaries (midpoints between adjacent signals)
        if n > 1:
            sorted_indices = np.argsort(signal_vectors[:, 0])
            sorted_vectors = signal_vectors[sorted_indices, 0]
            
            for i in range(len(sorted_vectors) - 1):
                midpoint = (sorted_vectors[i] + sorted_vectors[i+1]) / 2
                ax.axvline(x=midpoint, color='red', linestyle='--', linewidth=1.5, 
                          alpha=0.7, label='Decision Boundary' if i == 0 else '')
        
        ax.set_xlabel('φ₁ (First Basis Function)', fontsize=12, fontweight='bold')
        ax.set_title('Signal Space Representation (1D) with Decision Boundaries', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 0.5)
        plt.tight_layout()
        plt.show()
        
    elif m == 2:
        # 2D case: plot in 2D plane
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot signal points
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        for i in range(n):
            ax.scatter(signal_vectors[i, 0], signal_vectors[i, 1], s=150, 
                      c=[colors[i]], label=f'Signal {i+1}', marker='o', 
                      edgecolors='black', linewidths=2, zorder=5)
        
        # Draw decision boundaries (Voronoi diagram approximation)
        if n > 1:
            # Create a grid for decision boundaries
            x_min, x_max = signal_vectors[:, 0].min() - 1, signal_vectors[:, 0].max() + 1
            y_min, y_max = signal_vectors[:, 1].min() - 1, signal_vectors[:, 1].max() + 1
            
            # Draw lines between signal pairs (perpendicular bisectors)
            for i in range(n):
                for j in range(i+1, n):
                    # Midpoint
                    mid_x = (signal_vectors[i, 0] + signal_vectors[j, 0]) / 2
                    mid_y = (signal_vectors[i, 1] + signal_vectors[j, 1]) / 2
                    
                    # Direction vector
                    dx = signal_vectors[j, 0] - signal_vectors[i, 0]
                    dy = signal_vectors[j, 1] - signal_vectors[i, 1]
                    
                    # Perpendicular direction
                    perp_dx = -dy
                    perp_dy = dx
                    
                    # Normalize
                    norm = np.sqrt(perp_dx**2 + perp_dy**2)
                    if norm > 1e-10:
                        perp_dx /= norm
                        perp_dy /= norm
                        
                        # Draw line segment
                        scale = 5
                        ax.plot([mid_x - scale*perp_dx, mid_x + scale*perp_dx],
                               [mid_y - scale*perp_dy, mid_y + scale*perp_dy],
                               'r--', linewidth=1.5, alpha=0.6, zorder=1)
        
        ax.set_xlabel('φ₁ (First Basis Function)', fontsize=12, fontweight='bold')
        ax.set_ylabel('φ₂ (Second Basis Function)', fontsize=12, fontweight='bold')
        ax.set_title('Signal Space Representation (2D) with Decision Boundaries', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()


def Signal_Space_Analysis(Phis, Signals):
    """
    Part 1.4: Signal Space Analysis
    
    Calculates the Euclidean Distance and cross correlation for given signals.
    Prints all signal pairs that have the minimum distance between them and
    the corresponding (distance, cross correlation).
    
    Parameters:
    -----------
    Phis : numpy.ndarray
        m x N matrix containing orthonormal basis functions
    Signals : numpy.ndarray
        n x N matrix where each row is a signal
    """
    n, N = Signals.shape
    
    # Convert all signals to signal space representation
    signal_vectors = []
    for i in range(n):
        vec = Signal_Rep(Phis, Signals[i, :])
        signal_vectors.append(vec)
    signal_vectors = np.array(signal_vectors)
    
    # Calculate Euclidean distances and cross correlations
    distances = np.zeros((n, n))
    cross_correlations = np.zeros((n, n))
    
    min_distance = float('inf')
    min_pairs = []
    
    print("\n" + "="*70)
    print("SIGNAL SPACE ANALYSIS")
    print("="*70)
    print(f"\nNumber of signals: {n}")
    print(f"Number of basis functions: {Phis.shape[0]}")
    print("\n" + "-"*70)
    print("Euclidean Distances and Cross Correlations:")
    print("-"*70)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0
                cross_correlations[i, j] = 1.0  # Auto-correlation
            else:
                # Euclidean distance in signal space
                dist = np.linalg.norm(signal_vectors[i, :] - signal_vectors[j, :])
                distances[i, j] = dist
                
                # Cross correlation (normalized)
                sig_i = Signals[i, :]
                sig_j = Signals[j, :]
                cross_corr = np.dot(sig_i, sig_j) / (np.linalg.norm(sig_i) * np.linalg.norm(sig_j))
                cross_correlations[i, j] = cross_corr
                
                # Track minimum distance
                if dist < min_distance:
                    min_distance = dist
                    min_pairs = [(i, j)]
                elif abs(dist - min_distance) < 1e-10:
                    min_pairs.append((i, j))
    
    # Print distance matrix
    print("\nEuclidean Distance Matrix:")
    print("     ", end="")
    for j in range(n):
        print(f"  S{j+1}  ", end="")
    print()
    for i in range(n):
        print(f"S{i+1}  ", end="")
        for j in range(n):
            print(f"{distances[i, j]:7.4f}", end="")
        print()
    
    # Print cross correlation matrix
    print("\nCross Correlation Matrix:")
    print("     ", end="")
    for j in range(n):
        print(f"  S{j+1}  ", end="")
    print()
    for i in range(n):
        print(f"S{i+1}  ", end="")
        for j in range(n):
            print(f"{cross_correlations[i, j]:7.4f}", end="")
        print()
    
    # Print minimum distance pairs
    print("\n" + "-"*70)
    print("Minimum Distance Analysis:")
    print("-"*70)
    print(f"Minimum Euclidean Distance: {min_distance:.6f}")
    print(f"\nSignal pairs with minimum distance:")
    for pair in min_pairs:
        i, j = pair
        print(f"  Signal {i+1} <-> Signal {j+1}:")
        print(f"    Distance: {distances[i, j]:.6f}")
        print(f"    Cross Correlation: {cross_correlations[i, j]:.6f}")
    
    print("="*70 + "\n")
    
    return distances, cross_correlations


def AWGN_Signal_Space(Phis, Signals, No_over_2, plot_title="AWGN in Signal Space"):
    """
    Part 1.5: AWGN in Signal Space
    
    Adds AWGN noise to the signals with zero mean and No/2 variance,
    then plots the noisy signals on signal space.
    Labels the original signals which are not affected by noise.
    
    Parameters:
    -----------
    Phis : numpy.ndarray
        m x N matrix containing orthonormal basis functions
    Signals : numpy.ndarray
        n x N matrix where each row is a signal
    No_over_2 : float
        Variance of the noise (No/2)
    plot_title : str
        Title for the plot
    """
    m, N = Phis.shape
    n, _ = Signals.shape
    
    if m > 3:
        print(f"3D plotting not supported for m > 3. Current m={m}")
        return
    
    # Convert original signals to signal space
    original_vectors = []
    for i in range(n):
        vec = Signal_Rep(Phis, Signals[i, :])
        original_vectors.append(vec)
    original_vectors = np.array(original_vectors)
    
    # Add AWGN noise to signals
    noisy_signals = Signals.copy()
    for i in range(n):
        noise = np.random.normal(0, np.sqrt(No_over_2), N)
        noisy_signals[i, :] = Signals[i, :] + noise
    
    # Convert noisy signals to signal space
    noisy_vectors = []
    for i in range(n):
        vec = Signal_Rep(Phis, noisy_signals[i, :])
        noisy_vectors.append(vec)
    noisy_vectors = np.array(noisy_vectors)
    
    # Plotting
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    
    if m == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot original signals
        for i in range(n):
            ax.scatter(original_vectors[i, 0], 0, s=200, c=[colors[i]], 
                      label=f'Original Signal {i+1}', marker='o', 
                      edgecolors='black', linewidths=2, zorder=5)
        
        # Plot noisy signals
        for i in range(n):
            ax.scatter(noisy_vectors[i, 0], 0, s=100, c=[colors[i]], 
                      marker='x', linewidths=2, alpha=0.7, zorder=3,
                      label=f'Noisy Signal {i+1}' if i == 0 else '')
        
        ax.set_xlabel('φ₁ (First Basis Function)', fontsize=12, fontweight='bold')
        ax.set_title(f'{plot_title}\n(Noise Variance: No/2 = {No_over_2:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 0.5)
        
    elif m == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot original signals
        for i in range(n):
            ax.scatter(original_vectors[i, 0], original_vectors[i, 1], s=200, 
                      c=[colors[i]], label=f'Original Signal {i+1}', marker='o', 
                      edgecolors='black', linewidths=2, zorder=5)
        
        # Plot noisy signals
        for i in range(n):
            ax.scatter(noisy_vectors[i, 0], noisy_vectors[i, 1], s=100, 
                      c=[colors[i]], marker='x', linewidths=2, alpha=0.7, zorder=3,
                      label=f'Noisy Signal {i+1}' if i == 0 else '')
        
        ax.set_xlabel('φ₁ (First Basis Function)', fontsize=12, fontweight='bold')
        ax.set_ylabel('φ₂ (Second Basis Function)', fontsize=12, fontweight='bold')
        ax.set_title(f'{plot_title}\n(Noise Variance: No/2 = {No_over_2:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
    elif m == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original signals
        for i in range(n):
            ax.scatter(original_vectors[i, 0], original_vectors[i, 1], 
                      original_vectors[i, 2], s=200, c=[colors[i]], 
                      label=f'Original Signal {i+1}', marker='o', 
                      edgecolors='black', linewidths=2)
        
        # Plot noisy signals
        for i in range(n):
            ax.scatter(noisy_vectors[i, 0], noisy_vectors[i, 1], 
                      noisy_vectors[i, 2], s=100, c=[colors[i]], 
                      marker='x', linewidths=2, alpha=0.7,
                      label=f'Noisy Signal {i+1}' if i == 0 else '')
        
        ax.set_xlabel('φ₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('φ₂', fontsize=12, fontweight='bold')
        ax.set_zlabel('φ₃', fontsize=12, fontweight='bold')
        ax.set_title(f'{plot_title}\n(Noise Variance: No/2 = {No_over_2:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return noisy_signals, noisy_vectors


def create_signal_from_piecewise(t, time_ranges, values):
    """
    Helper function to create signals from piecewise definitions.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time vector
    time_ranges : list of tuples
        List of (t_start, t_end) tuples defining time ranges
    values : list of floats
        List of values corresponding to each time range
    
    Returns:
    --------
    signal : numpy.ndarray
        Signal vector
    """
    signal = np.zeros_like(t)
    for (t_start, t_end), value in zip(time_ranges, values):
        mask = (t >= t_start) & (t < t_end)
        signal[mask] = value
    return signal


def plot_signals(t, Signals, title="Signals", labels=None):
    """
    Helper function to plot signals in time domain.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time vector
    Signals : numpy.ndarray
        n x N matrix where each row is a signal
    title : str
        Plot title
    labels : list of str
        Labels for each signal
    """
    n, _ = Signals.shape
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i in range(n):
        label = labels[i] if labels else f'Signal {i+1}'
        ax.plot(t, Signals[i, :], linewidth=2, color=colors[i], label=label)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_basis_functions(t, Phis, title="Basis Functions"):
    """
    Helper function to plot basis functions.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time vector
    Phis : numpy.ndarray
        m x N matrix containing basis functions
    title : str
        Plot title
    """
    m, N = Phis.shape
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, m))
    for i in range(m):
        ax.plot(t, Phis[i, :], linewidth=2, color=colors[i], 
               label=f'φ{i+1}(t)', linestyle='-')
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

