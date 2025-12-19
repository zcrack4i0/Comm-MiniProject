# Communications II - Signal Space Analysis Project

**EECS 316 Fall 25-26**  
**Cairo University - Faculty of Engineering**

This project implements Gram-Schmidt orthogonalization and signal space analysis for communication systems using Python.

## Project Overview

The project consists of two main parts:

### Part 1: Core Functions
1. **Gram-Schmidt Orthogonalization** (`Basis_Cal`): Calculates orthonormal basis functions
2. **Signal Space Representation** (`Signal_Rep`): Converts signals to signal space coefficients
3. **Decision Boundaries** (`Decision_boundaries`): Draws decision boundaries for m=1,2
4. **Signal Space Analysis** (`Signal_Space_Analysis`): Calculates Euclidean distances and cross correlations
5. **AWGN in Signal Space** (`AWGN_Signal_Space`): Adds noise and visualizes in signal space

### Part 2: Problem Solving
- **Problem 1**: Two piecewise constant signals
- **Problem 2**: Rectangular pulse and modified rectangular pulse
- **Problem 3**: Four signals with relationships (S2=-S1, S4=-S3)
- **Noise Exercise**: Analysis with different E1/No values (10, 5, 0, -5, -10 dB)

## Files Structure

```
.
├── signal_space_analysis.py    # Core functions implementation
├── main_project.py             # Main script to run all problems (interactive plots)
├── generate_report_plots.py     # Script to generate and save all plots for report
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── figures/                     # Directory for saved plots (created automatically)
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode (with plots displayed)
Run the main script to see all plots interactively:
```bash
python main_project.py
```

### Generate Report Plots
To generate and save all plots for the report:
```bash
python generate_report_plots.py
```
This will create a `figures/` directory with all plots saved as PNG files.

## Key Functions

### `Basis_Cal(Signals, n)`
Calculates Gram-Schmidt orthonormal basis functions.

**Parameters:**
- `Signals`: n×N numpy array (n signals, each of length N)
- `n`: Number of signals

**Returns:**
- `Phis`: m×N numpy array (m basis functions)
- `m`: Number of basis functions

### `Signal_Rep(Phis, signal)`
Converts a signal to signal space representation.

**Parameters:**
- `Phis`: m×N numpy array (basis functions)
- `signal`: 1×N numpy array (signal vector)

**Returns:**
- `signal_vector`: 1×m numpy array (signal space coefficients)

### `Decision_boundaries(Phis, Signals)`
Draws decision boundaries for m=1 or m=2.

**Parameters:**
- `Phis`: m×N numpy array (basis functions)
- `Signals`: n×N numpy array (signals)

### `Signal_Space_Analysis(Phis, Signals)`
Analyzes signal space: calculates distances and correlations.

**Parameters:**
- `Phis`: m×N numpy array (basis functions)
- `Signals`: n×N numpy array (signals)

**Returns:**
- `distances`: n×n distance matrix
- `cross_correlations`: n×n correlation matrix

### `AWGN_Signal_Space(Phis, Signals, No_over_2, plot_title)`
Adds AWGN noise and visualizes in signal space.

**Parameters:**
- `Phis`: m×N numpy array (basis functions)
- `Signals`: n×N numpy array (signals)
- `No_over_2`: Noise variance (No/2)
- `plot_title`: Title for the plot

## Sampling

The project uses a sampling time of **ts = 40 ms = 0.04 s** as specified in the requirements.

## Output

The script generates:
- Time-domain signal plots
- Basis function plots
- Signal space representations
- Decision boundary plots
- AWGN noise analysis plots
- Console output with numerical results (distances, correlations, etc.)

## Notes

- All plots are properly labeled with legends and axis labels
- Original signals are marked with circles (○)
- Noisy signals are marked with crosses (×)
- The code is fully commented and organized
- Random seed is set for reproducibility

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- matplotlib >= 3.5.0

## Author

This project was implemented for the Communications II course mini-project.

