"""
Run all experiments and generate a comprehensive report.
"""
import sys
sys.path.insert(0, '..')

import time
from datetime import datetime


def run_all_experiments():
    """Run all experiments sequentially."""
    print("=" * 70)
    print("COMPREHENSIVE ANTONYM ANALYSIS EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    total_start = time.time()

    # Experiment 1: Polysemy
    print("\n\n" + "▓" * 70)
    print("EXPERIMENT 1: POLYSEMY HANDLING")
    print("▓" * 70)
    try:
        from polysemy import run_polysemy_experiment
        t0 = time.time()
        results['polysemy'] = run_polysemy_experiment()
        print(f"\n[Experiment 1 completed in {time.time()-t0:.1f}s]")
    except Exception as e:
        print(f"\n[Experiment 1 failed: {e}]")

    # Experiment 2: Intensity
    print("\n\n" + "▓" * 70)
    print("EXPERIMENT 2: INTENSITY ANALYSIS")
    print("▓" * 70)
    try:
        from intensity import run_intensity_experiment
        t0 = time.time()
        results['intensity'] = run_intensity_experiment()
        print(f"\n[Experiment 2 completed in {time.time()-t0:.1f}s]")
    except Exception as e:
        print(f"\n[Experiment 2 failed: {e}]")

    # Experiment 3: Generation
    print("\n\n" + "▓" * 70)
    print("EXPERIMENT 3: ANTONYM GENERATION")
    print("▓" * 70)
    try:
        from generation import run_generation_experiment
        t0 = time.time()
        results['generation'] = run_generation_experiment()
        print(f"\n[Experiment 3 completed in {time.time()-t0:.1f}s]")
    except Exception as e:
        print(f"\n[Experiment 3 failed: {e}]")

    # Experiment 4: Dimension Analysis
    print("\n\n" + "▓" * 70)
    print("EXPERIMENT 4: DIMENSION ANALYSIS")
    print("▓" * 70)
    try:
        from dimension_analysis import run_dimension_experiment
        t0 = time.time()
        results['dimensions'] = run_dimension_experiment()
        print(f"\n[Experiment 4 completed in {time.time()-t0:.1f}s]")
    except Exception as e:
        print(f"\n[Experiment 4 failed: {e}]")

    # Experiment 5: Model Comparison
    print("\n\n" + "▓" * 70)
    print("EXPERIMENT 5: MODEL COMPARISON")
    print("▓" * 70)
    try:
        from model_comparison import run_model_comparison
        t0 = time.time()
        results['models'] = run_model_comparison()
        print(f"\n[Experiment 5 completed in {time.time()-t0:.1f}s]")
    except Exception as e:
        print(f"\n[Experiment 5 failed: {e}]")

    # Summary
    total_time = time.time() - total_start
    print("\n\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_all_experiments()
