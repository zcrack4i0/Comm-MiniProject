import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')
# ============================================================================
# PART 1.1: GRAM-SCHMIDT ORTHOGONALIZATION
# ============================================================================
def Basis_Cal(Signals, n):
    """
    Part 1.1: Gram-Schmidt Orthogonalization
    
    Calculates the Gram-Schmidt orthonormal basis functions (phi1, phi2, ..., phim)
    for two or more input signals.
    """
    n_signals, N = Signals.shape
    
    basis_functions = []
    
    # First basis function: normalize the first signal
    s1 = Signals[0, :]
    norm_s1 = np.linalg.norm(s1)
    
    if norm_s1 < 1e-10:  
        # Skip zero signals
        pass
    else:
        phi1 = s1 / norm_s1
        basis_functions.append(phi1)
    
    for i in range(1, n_signals):
        si = Signals[i, :]
        
        si_orthogonal = si.copy()
        for phi in basis_functions:
            proj_coeff = np.dot(si, phi)
            si_orthogonal = si_orthogonal - proj_coeff * phi
        
        # Normalize to get new basis function
        norm_si_orth = np.linalg.norm(si_orthogonal)
        
        if norm_si_orth > 1e-10:  # Only add if not zero 
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


# ============================================================================
# PART 1.2: SIGNAL SPACE REPRESENTATION
# ============================================================================
def Signal_Rep(Phis, signal):
    """
    Part 1.2: Signal Space Representation
    
    Calculates the signal space representation (coefficients) for a given signal
    using orthonormal basis functions.
    """
    m, N = Phis.shape
    
    # Calculate coefficients by projecting signal onto each basis function
    signal_vector = np.zeros(m)
    for i in range(m):
        signal_vector[i] = np.dot(signal, Phis[i, :])
    
    return signal_vector


# ============================================================================
# PART 1.3: DECISION BOUNDARIES
# ============================================================================
def Decision_boundaries(Phis, Signals):
    """
    Part 1.3: Decision Boundaries
    
    Draws the decision boundaries for the given signals using orthonormal basis functions.
    for m = 1 or m = 2 dimensions.
    """
    m, N = Phis.shape
    n, _ = Signals.shape
    
    if m > 2:
        handle_logs(f"Decision boundaries can only be drawn for m=1 or m=2. Current m={m}")
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
        
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        for i in range(n):
            ax.scatter(signal_vectors[i, 0], signal_vectors[i, 1], s=150, 
                      c=[colors[i]], label=f'Signal {i+1}', marker='o', 
                      edgecolors='black', linewidths=2, zorder=5)
        
        # Draw decision boundaries (perpendicular bisectors between signal pairs)
        if n > 1:
            for i in range(n):
                for j in range(i+1, n):
                    mid_x = (signal_vectors[i, 0] + signal_vectors[j, 0]) / 2
                    mid_y = (signal_vectors[i, 1] + signal_vectors[j, 1]) / 2
                    
                    dx = signal_vectors[j, 0] - signal_vectors[i, 0]
                    dy = signal_vectors[j, 1] - signal_vectors[i, 1]
                    
                    perp_dx = -dy
                    perp_dy = dx
                    
                    norm = np.sqrt(perp_dx**2 + perp_dy**2)
                    if norm > 1e-10:
                        perp_dx /= norm
                        perp_dy /= norm
                        
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


# ============================================================================
# PART 1.4: SIGNAL SPACE ANALYSIS
# ============================================================================
def Signal_Space_Analysis(Phis, Signals):
    """
    Part 1.4: Signal Space Analysis
    
    Calculates the Euclidean Distance and cross correlation for given signals.
    handle_logss all signal pairs that have the minimum distance between them and
    the corresponding (distance, cross correlation).
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
    
    handle_logs("\n" + "="*70)
    handle_logs("SIGNAL SPACE ANALYSIS")
    handle_logs("="*70)
    handle_logs(f"\nNumber of signals: {n}")
    handle_logs(f"Number of basis functions: {Phis.shape[0]}")
    handle_logs("\n" + "-"*70)
    handle_logs("Euclidean Distances and Cross Correlations:")
    handle_logs("-"*70)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0
                cross_correlations[i, j] = 1.0  
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
    
    # handle_logs distance matrix
    handle_logs("\nEuclidean Distance Matrix:")
    handle_logs("     ", end="")
    for j in range(n):
        handle_logs(f"  S{j+1}  ", end="")
    handle_logs()
    for i in range(n):
        handle_logs(f"S{i+1}  ", end="")
        for j in range(n):
            handle_logs(f"{distances[i, j]:7.4f}", end="")
        handle_logs()
    
    # handle_logs cross correlation matrix
    handle_logs("\nCross Correlation Matrix:")
    handle_logs("     ", end="")
    for j in range(n):
        handle_logs(f"  S{j+1}  ", end="")
    handle_logs()
    for i in range(n):
        handle_logs(f"S{i+1}  ", end="")
        for j in range(n):
            handle_logs(f"{cross_correlations[i, j]:7.4f}", end="")
        handle_logs()
    
    # handle_logs minimum distance pairs
    handle_logs("\n" + "-"*70)
    handle_logs("Minimum Distance Analysis:")
    handle_logs("-"*70)
    handle_logs(f"Minimum Euclidean Distance: {min_distance:.6f}")
    handle_logs(f"\nSignal pairs with minimum distance:")
    for pair in min_pairs:
        i, j = pair
        handle_logs(f"  Signal {i+1} <-> Signal {j+1}:")
        handle_logs(f"    Distance: {distances[i, j]:.6f}")
        handle_logs(f"    Cross Correlation: {cross_correlations[i, j]:.6f}")
    
    handle_logs("="*70 + "\n")
    
    return distances, cross_correlations


# ============================================================================
# PART 1.5: AWGN IN SIGNAL SPACE
# ============================================================================
def AWGN_Signal_Space(Phis, Signals, No_over_2, plot_title="AWGN in Signal Space"):
    """
    Part 1.5: AWGN in Signal Space
    
    Adds AWGN noise to the signals with zero mean and No/2 variance,
    then plots the noisy signals on signal space.
    Labels the original signals which are not affected by noise.
    """
    m, N = Phis.shape
    n, _ = Signals.shape
    
    if m > 3:
        handle_logs(f"3D plotting not supported for m > 3. Current m={m}")
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
    Function to create signals from piecewise definitions.
    """
    signal = np.zeros_like(t)
    for (t_start, t_end), value in zip(time_ranges, values):
        mask = (t >= t_start) & (t < t_end)
        signal[mask] = value
    return signal


def plot_signals(t, Signals, title="Signals", labels=None):
    """     
    Function to plot signals in time domain.
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
    Function to plot basis functions.
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



# Set random seed for reproducibility
np.random.seed(42)

ts = 0.04  # 40 ms


def handle_logs(*args, end='\n', sep=' '):
    message = sep.join(str(arg) for arg in args) if args else ''
    print(message, end=end)
    try:
        with open('logs.txt', 'a') as f:
            f.write(message + (end if end != '\n' else '\n'))
    except:
        pass  # Ignore file writing errors
    

def solve_problem_1():
    """
    Problem 1:
    S1 = { -3      0 < t < 0.75
           0.7     0.75 ≤ t < 1
    S2 = { 7.5     0 < t < 0.75
          -1.75    0.75 ≤ t < 1
    """
    handle_logs("\n" + "="*80)
    handle_logs("PROBLEM 1")
    handle_logs("="*80)
    
    # Time vector
    t = np.arange(0, 1, ts)
    N = len(t)
    
    # Create signals
    S1 = create_signal_from_piecewise(t, [(0, 0.75), (0.75, 1)], [-3, 0.7])
    S2 = create_signal_from_piecewise(t, [(0, 0.75), (0.75, 1)], [7.5, -1.75])
    
    Signals = np.array([S1, S2])
    n = 2
    
    # Plot original signals
    plot_signals(t, Signals, "Problem 1: Original Signals", 
                labels=['S1', 'S2'])
    
    # Part 1.1: Calculate basis functions using Gram-Schmidt Orthogonalization
    Phis, m = Basis_Cal(Signals, n)  # ← Part 1.1: Basis_Cal function
    handle_logs(f"\nNumber of basis functions: m = {m}")
    handle_logs(f"Basis functions shape: {Phis.shape}")
    
    # Plot basis functions
    plot_basis_functions(t, Phis, "Problem 1: Orthonormal Basis Functions")
    
    # Part 1.2: Signal space representation
    handle_logs("\nSignal Space Coefficients:")
    for i in range(n):
        vec = Signal_Rep(Phis, Signals[i, :])  # ← Part 1.2: Signal_Rep function
        handle_logs(f"Signal {i+1} (S{i+1}): {vec}")
    
    # Part 1.3: Decision boundaries
    if m <= 2:
        Decision_boundaries(Phis, Signals)  # ← Part 1.3: Decision_boundaries function
    
    # Part 1.4: Signal space analysis
    distances, cross_correlations = Signal_Space_Analysis(Phis, Signals)  # ← Part 1.4: Signal_Space_Analysis function
    
    
    return Phis, Signals, t


def solve_problem_2():
    """
    Problem 2:
    S1 = { 1      0 < t < 1
           0      else
    S2 = { 1      0 < t < 0.75
          -1      0.75 ≤ t < 1
    """
    handle_logs("\n" + "="*80)
    handle_logs("PROBLEM 2")
    handle_logs("="*80)
    
    # Time vector
    t = np.arange(0, 1, ts)
    N = len(t)
    
    # Create signals
    S1 = create_signal_from_piecewise(t, [(0, 1)], [1])
    S2 = create_signal_from_piecewise(t, [(0, 0.75), (0.75, 1)], [1, -1])
    
    Signals = np.array([S1, S2])
    n = 2
    
    # Plot original signals
    plot_signals(t, Signals, "Problem 2: Original Signals", 
                labels=['S1', 'S2'])
    
    # Part 1.1: Calculate basis functions using Gram-Schmidt Orthogonalization
    Phis, m = Basis_Cal(Signals, n)  # ← Part 1.1: Basis_Cal function
    handle_logs(f"\nNumber of basis functions: m = {m}")
    handle_logs(f"Basis functions shape: {Phis.shape}")
    
    # Plot basis functions
    plot_basis_functions(t, Phis, "Problem 2: Orthonormal Basis Functions")
    
    # Part 1.2: Signal space representation
    handle_logs("\nSignal Space Coefficients:")
    for i in range(n):
        vec = Signal_Rep(Phis, Signals[i, :])  # ← Part 1.2: Signal_Rep function
        handle_logs(f"Signal {i+1} (S{i+1}): {vec}")
    
    # Part 1.3: Decision boundaries
    if m <= 2:
        Decision_boundaries(Phis, Signals)  # ← Part 1.3: Decision_boundaries function
    
    # Part 1.4: Signal space analysis
    distances, cross_correlations = Signal_Space_Analysis(Phis, Signals)  # ← Part 1.4: Signal_Space_Analysis function
    
    
    return Phis, Signals, t


def solve_problem_3():
    """
    Problem 3:
    S1 = { 1      0 < t < 1
           0      else
    S2 = -S1
    S3 = { 2      0 < t < 0.75
           0.5    0.75 ≤ t < 1
    S4 = -S3
    """
    handle_logs("\n" + "="*80)
    handle_logs("PROBLEM 3")
    handle_logs("="*80)
    
    # Time vector
    t = np.arange(0, 1, ts)
    N = len(t)
    
    # Create signals
    S1 = create_signal_from_piecewise(t, [(0, 1)], [1])
    S2 = -S1
    S3 = create_signal_from_piecewise(t, [(0, 0.75), (0.75, 1)], [2, 0.5])
    S4 = -S3
    
    Signals = np.array([S1, S2, S3, S4])
    n = 4
    
    # Plot original signals
    plot_signals(t, Signals, "Problem 3: Original Signals", 
                labels=['S1', 'S2 = -S1', 'S3', 'S4 = -S3'])
    
    # Part 1.1: Calculate basis functions using Gram-Schmidt Orthogonalization
    Phis, m = Basis_Cal(Signals, n)  # ← Part 1.1: Basis_Cal function
    handle_logs(f"\nNumber of basis functions: m = {m}")
    handle_logs(f"Basis functions shape: {Phis.shape}")
    
    # Plot basis functions
    plot_basis_functions(t, Phis, "Problem 3: Orthonormal Basis Functions")
    
    # Part 1.2: Signal space representation
    handle_logs("\nSignal Space Coefficients:")
    for i in range(n):
        vec = Signal_Rep(Phis, Signals[i, :])  # ← Part 1.2: Signal_Rep function
        handle_logs(f"Signal {i+1} (S{i+1}): {vec}")
    
    # Part 1.3: Decision boundaries
    if m <= 2:
        Decision_boundaries(Phis, Signals)  # ← Part 1.3: Decision_boundaries function
    
    # Part 1.4: Signal space analysis
    distances, cross_correlations = Signal_Space_Analysis(Phis, Signals)  # ← Part 1.4: Signal_Space_Analysis function
    
    
    return Phis, Signals, t


def noise_exercise():
    """
    Noise Exercise:
    S1 = { 1      0 < t < 1
           0      else
    S2 = -2*S1
    S3 = { 1.5    0 < t < 0.75
           0.7    0.75 ≤ t < 1
    S4 = -3*S3
    
    Find bases functions, draw signals with decision boundaries,
    simulate channel with noise for E1/No = 10, 5, 0, -5, -10 dB
    Generate 50 noisy samples and plot them on signal space.
    """
    handle_logs("\n" + "="*80)
    handle_logs("NOISE EXERCISE")
    handle_logs("="*80)
    
    # Time vector
    t = np.arange(0, 1, ts)
    N = len(t)
    
    # Create signals
    S1 = create_signal_from_piecewise(t, [(0, 1)], [1])
    S2 = -2 * S1
    S3 = create_signal_from_piecewise(t, [(0, 0.75), (0.75, 1)], [1.5, 0.7])
    S4 = -3 * S3
    
    Signals = np.array([S1, S2, S3, S4])
    n = 4
    
    # Plot original signals
    plot_signals(t, Signals, "Noise Exercise: Original Signals", 
                labels=['S1', 'S2 = -2*S1', 'S3', 'S4 = -3*S3'])
    
    # Part 1.1: Calculate basis functions using Gram-Schmidt Orthogonalization
    Phis, m = Basis_Cal(Signals, n)  # ← Part 1.1: Basis_Cal function
    handle_logs(f"\nNumber of basis functions: m = {m}")
    handle_logs(f"Basis functions shape: {Phis.shape}")
    
    # Plot basis functions
    plot_basis_functions(t, Phis, "Noise Exercise: Orthonormal Basis Functions")
    
    # Part 1.2: Signal space representation
    handle_logs("\nSignal Space Coefficients:")
    signal_vectors = []
    for i in range(n):
        vec = Signal_Rep(Phis, Signals[i, :])  # ← Part 1.2: Signal_Rep function
        signal_vectors.append(vec)
        handle_logs(f"Signal {i+1} (S{i+1}): {vec}")
    signal_vectors = np.array(signal_vectors)
    
    # Part 1.3: Decision boundaries
    if m <= 2:
        Decision_boundaries(Phis, Signals)  # ← Part 1.3: Decision_boundaries function
    
    # Part 1.4: Signal space analysis
    distances, cross_correlations = Signal_Space_Analysis(Phis, Signals)  # ← Part 1.4: Signal_Space_Analysis function
    
    # Calculate E1 (energy of first signal)
    E1 = np.sum(S1**2) * ts  # Energy = integral of signal squared
    
    # E1/No values in dB
    E1_No_dB_values = [10, 5, 0, -5, -10]
    
    # Convert to linear scale and calculate No/2
    # E1/No (linear) = 10^(E1/No_dB / 10)
    # No = E1 / (E1/No_linear)
    # No/2 = E1 / (2 * E1/No_linear)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    num_samples = 50
    
    for E1_No_dB in E1_No_dB_values:
        # Convert dB to linear
        E1_No_linear = 10**(E1_No_dB / 10)
        No = E1 / E1_No_linear
        No_over_2 = No / 2
        
        handle_logs(f"\nE1/No = {E1_No_dB} dB")
        handle_logs(f"  E1 = {E1:.6f}")
        handle_logs(f"  E1/No (linear) = {E1_No_linear:.6f}")
        handle_logs(f"  No = {No:.6f}")
        handle_logs(f"  No/2 = {No_over_2:.6f}")
        
        # Generate 50 noisy samples for each signal
        all_noisy_vectors = []
        for i in range(n):
            signal_noisy_vectors = []
            for sample in range(num_samples):
                # Add AWGN noise
                noise = np.random.normal(0, np.sqrt(No_over_2), N)
                noisy_signal = Signals[i, :] + noise
                # Part 1.2: Convert to signal space
                noisy_vec = Signal_Rep(Phis, noisy_signal)  # ← Part 1.2: Signal_Rep function (for noisy signals)
                signal_noisy_vectors.append(noisy_vec)
            all_noisy_vectors.append(np.array(signal_noisy_vectors))
        
        # Plot
        if m == 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot original signals (circles)
            for i in range(n):
                ax.scatter(signal_vectors[i, 0], 0, s=300, c=[colors[i]], 
                          label=f'Original S{i+1}', marker='o', 
                          edgecolors='black', linewidths=3, zorder=10)
            
            # Plot noisy samples (crosses)
            for i in range(n):
                ax.scatter(all_noisy_vectors[i][:, 0], 
                          np.zeros(num_samples), s=50, c=[colors[i]], 
                          marker='x', linewidths=1.5, alpha=0.6, zorder=5,
                          label=f'Noisy S{i+1} (50 samples)' if i == 0 else '')
            
            ax.set_xlabel('φ₁ (First Basis Function)', fontsize=12, fontweight='bold')
            ax.set_title(f'Noise Exercise: E1/No = {E1_No_dB} dB\n' +
                        f'(No/2 = {No_over_2:.6f}, 50 noisy samples per signal)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, 0.5)
            
        elif m == 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot original signals (circles)
            for i in range(n):
                ax.scatter(signal_vectors[i, 0], signal_vectors[i, 1], 
                          s=300, c=[colors[i]], label=f'Original S{i+1}', 
                          marker='o', edgecolors='black', linewidths=3, zorder=10)
            
            # Plot noisy samples (crosses)
            for i in range(n):
                ax.scatter(all_noisy_vectors[i][:, 0], 
                          all_noisy_vectors[i][:, 1], s=50, c=[colors[i]], 
                          marker='x', linewidths=1.5, alpha=0.6, zorder=5,
                          label=f'Noisy S{i+1} (50 samples)' if i == 0 else '')
            
            ax.set_xlabel('φ₁ (First Basis Function)', fontsize=12, fontweight='bold')
            ax.set_ylabel('φ₂ (Second Basis Function)', fontsize=12, fontweight='bold')
            ax.set_title(f'Noise Exercise: E1/No = {E1_No_dB} dB\n' +
                        f'(No/2 = {No_over_2:.6f}, 50 noisy samples per signal)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()
        
        # Count errors (noisy samples that cross decision boundaries)
        # For simplicity, we'll count samples that are closer to a different signal
        error_count = 0
        for i in range(n):
            for noisy_vec in all_noisy_vectors[i]:
                # Find closest original signal
                distances_to_originals = [np.linalg.norm(noisy_vec - signal_vectors[j, :]) 
                                         for j in range(n)]
                closest_idx = np.argmin(distances_to_originals)
                if closest_idx != i:
                    error_count += 1
        
        total_samples = n * num_samples
        error_rate = error_count / total_samples * 100
        
        handle_logs(f"  Error Analysis:")
        handle_logs(f"    Total samples: {total_samples}")
        handle_logs(f"    Errors (misclassified): {error_count}")
        handle_logs(f"    Error rate: {error_rate:.2f}%")
        
        # Add comment about noise effect
        if E1_No_dB >= 10:
            comment = "High SNR: Noise has minimal effect. Very few errors occur. Signals are clearly distinguishable."
        elif E1_No_dB >= 5:
            comment = "Moderate-high SNR: Some noise spread visible. Low error rate. Signals mostly distinguishable."
        elif E1_No_dB >= 0:
            comment = "Moderate SNR: Noticeable noise spread. Moderate error rate. Some signal overlap."
        elif E1_No_dB >= -5:
            comment = "Low SNR: Significant noise spread. High error rate. Considerable signal overlap."
        else:
            comment = "Very low SNR: Severe noise spread. Very high error rate. Signals are barely distinguishable."
        
        handle_logs(f"  Comment: {comment}")
    
    return Phis, Signals, t


if __name__ == "__main__":
    
    # Solve Problem 1
    Phis1, Signals1, t1 = solve_problem_1()
    
    # Solve Problem 2
    Phis2, Signals2, t2 = solve_problem_2()
    
    # Solve Problem 3
    Phis3, Signals3, t3 = solve_problem_3()
    
    # Noise Exercise
    Phis_noise, Signals_noise, t_noise = noise_exercise()
    
    handle_logs("\n" + "="*80)
    handle_logs("ALL PROBLEMS COMPLETED SUCCESSFULLY!")
    handle_logs("="*80)




