"""
Communications II - Signal Space Analysis Project
EECS 316 Fall 25-26
Cairo University - Faculty of Engineering

Main script to solve all project problems and generate required outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from signal_space_analysis import (
    Basis_Cal, Signal_Rep, Decision_boundaries, Signal_Space_Analysis,
    AWGN_Signal_Space, create_signal_from_piecewise, plot_signals, plot_basis_functions
)

# Set random seed for reproducibility
np.random.seed(42)

ts = 0.04  # 40 ms


def handle_logs(message):
    print(f"\n{message}")
    with open('logs.txt', 'a') as f:
        f.write(f"\n{message}")
    

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




