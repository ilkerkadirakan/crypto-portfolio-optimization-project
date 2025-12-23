"""
Quick test of main pipeline with fixed optimization.
Tests only 2-asset combinations that were failing before.
"""

from pathlib import Path
from main import run_pipeline

# Test with specific failing combinations
test_combos = [
    ("NEARBTC", "XLMBTC"),
    ("SOLBTC", "STXBTC"),
    ("SOLBTC", "TRXBTC"),
]

print("="*80)
print("QUICK TEST: Testing previously failing 2-asset combinations")
print("="*80)
print(f"Testing {len(test_combos)} combinations")
print("Combos:", test_combos)
print("="*80)

# Create a test config with limited scope
config_path = Path("configs/params.yaml")

try:
    # Import backtest_engine directly for targeted testing
    from src import backtest_engine

    results = backtest_engine.run_backtest(
        freq="1D",
        version="baseline",
        model_list=["MV", "MVSK", "MCVaRSK"],
        combo_iterable=test_combos,
    )

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    if results.empty:
        print("[FAIL] No results generated!")
    else:
        print(f"[OK] Generated {len(results)} result rows")
        print(f"\nUnique combinations tested: {results['combo'].nunique()}")
        print(f"Models: {results['model'].unique().tolist()}")

        # Check for any failures
        failed_combos = []
        for combo in test_combos:
            combo_label = "_".join(combo)
            combo_results = results[results['combo'] == combo_label]
            if combo_results.empty:
                failed_combos.append(combo_label)
                print(f"\n[WARN] {combo_label}: No results (may have failed)")
            else:
                models_tested = combo_results['model'].unique()
                print(f"\n[OK] {combo_label}: {len(models_tested)} models succeeded")
                for model in ["MV", "MVSK", "MCVARSK"]:
                    model_data = combo_results[combo_results['model'] == model]
                    if not model_data.empty:
                        mean_ret = model_data['net_return'].mean()
                        print(f"     {model}: mean return = {mean_ret:.6f}")
                    else:
                        print(f"     {model}: [FAILED]")

        if failed_combos:
            print(f"\n[SUMMARY] {len(failed_combos)} combinations failed: {failed_combos}")
        else:
            print(f"\n[SUMMARY] All {len(test_combos)} combinations succeeded!")

except Exception as exc:
    print(f"\n[ERROR] Test failed with exception:")
    print(f"  {type(exc).__name__}: {exc}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
