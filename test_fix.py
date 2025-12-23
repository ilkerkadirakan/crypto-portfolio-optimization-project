"""
Quick test script to verify the optimization fix.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.optim_models import solve_portfolio

# Create test data with realistic values
n_assets = 3
mu = np.array([0.001, 0.002, 0.0015])
# Create a proper positive definite covariance matrix
sigma = np.array([
    [0.04, 0.01, 0.005],
    [0.01, 0.06, 0.008],
    [0.005, 0.008, 0.05]
])
skew = np.array([0.1, -0.2, 0.05])
kurt = np.array([3.0, 4.0, 3.5])
cvar = np.array([0.02, 0.03, 0.025])

print("Testing optimization models with sample data...")
print(f"Mean returns: {mu}")
print(f"Covariance matrix:\n{sigma}")

models = ["MV", "MVSK", "MCVaRSK"]

for model in models:
    try:
        print(f"\n{'='*60}")
        print(f"Testing {model} model...")

        # Use custom params without restrictive max_weight
        custom_params = {
            "portfolio": {
                "allow_short": False,
                "min_weight": 0.0,
                "max_weight": 1.0
            },
            "models": {
                "solver": {"primary": "ECOS"},
                "objectives": {
                    "mv": {"lambda_mean": 1.0, "lambda_variance": 1.0},
                    "mvsk": {"lambda_mean": 1.0, "lambda_variance": 1.0, "lambda_skew": 1.0, "lambda_kurtosis": 1.0},
                    "mcvarsk": {"lambda_mean": 1.0, "lambda_skew": 1.0, "lambda_kurtosis": 1.0, "lambda_cvar": 1.0}
                }
            }
        }

        if model == "MV":
            weights = solve_portfolio(model, mu, sigma, params=custom_params)
        elif model == "MVSK":
            weights = solve_portfolio(model, mu, sigma, skew=skew, kurt=kurt, params=custom_params)
        else:  # MCVaRSK
            weights = solve_portfolio(model, mu, sigma, skew=skew, kurt=kurt, cvar_series=cvar, params=custom_params)

        print(f"[OK] {model} optimization succeeded!")
        print(f"  Weights: {weights}")
        print(f"  Sum: {weights.sum():.6f}")

    except Exception as exc:
        print(f"[FAIL] {model} optimization failed!")
        print(f"  Error: {exc}")

print("\n" + "="*60)
print("Testing with problematic inputs (near-singular matrix)...")

# Create a near-singular covariance matrix (2x2 for simplicity)
mu_2 = np.array([0.001, 0.002])
sigma_singular = np.array([[0.01, 0.0099], [0.0099, 0.01]])
print(f"Near-singular covariance matrix:\n{sigma_singular}")

try:
    weights = solve_portfolio("MV", mu_2, sigma_singular)
    print(f"[OK] Handled near-singular matrix successfully!")
    print(f"  Weights: {weights}")
except Exception as exc:
    print(f"[FAIL] Failed to handle near-singular matrix")
    print(f"  Error: {exc}")

print("\n" + "="*60)
print("Testing with NaN inputs...")

mu_nan = np.array([0.001, np.nan, 0.001])
try:
    weights = solve_portfolio("MV", mu_nan, sigma)
    print(f"[FAIL] Should have caught NaN input!")
except ValueError as exc:
    print(f"[OK] Correctly caught NaN input!")
    print(f"  Error message: {exc}")

print("\n" + "="*60)
print("All tests completed!")
