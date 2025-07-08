#!/usr/bin/env python3
"""
Example script to run PCA analysis on RanPAC Features_h

This script provides examples of how to run the PCA analysis on saved Features_h
from RanPAC experiments.

Requirements:
    pip install matplotlib seaborn scikit-learn pandas

Usage examples:
    # Basic usage (with automatic caching)
    python run_pca_analysis.py logs/ncm/cifar224/0/10/my_experiment

    # With custom components
    python run_pca_analysis.py logs/adapter/cifar224/0/10/my_experiment --n_components 100

    # Force recompute (ignore cache)
    python run_pca_analysis.py --force_recompute logs/ncm/cifar224/0/10/my_experiment

    # Only generate plots (fast)
    python run_pca_analysis.py --plot_only logs/ncm/cifar224/0/10/my_experiment

    # Multiple experiments with shared cache
    python run_pca_analysis.py --shared_cache --output_base ./results logs/exp1 logs/exp2
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path

def find_experiment_directories(base_path: str) -> list:
    """Find all experiment directories under a base path."""
    if os.path.isdir(base_path):
        # Check if this is already an experiment directory
        features_dir = os.path.join(base_path, "features_h")
        if os.path.exists(features_dir):
            return [base_path]
        
        # Look for experiment directories
        exp_dirs = []
        for root, dirs, files in os.walk(base_path):
            if "features_h" in dirs:
                exp_dirs.append(root)
        return exp_dirs
    
    return []

def run_pca_analysis(exp_path: str, n_components: int = 50, output_dir: str = None,
                    force_recompute: bool = False, no_cache: bool = False, 
                    plot_only: bool = False, cache_file: str = None):
    """Run PCA analysis for a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running PCA analysis for: {exp_path}")
    print(f"{'='*60}")
    
    # Build command
    cmd = ["python", "pca_analysis.py", "--exp_path", exp_path, "--n_components", str(n_components)]
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    if force_recompute:
        cmd.append("--force_recompute")
    if no_cache:
        cmd.append("--no_cache")
    if plot_only:
        cmd.append("--plot_only")
    if cache_file:
        cmd.extend(["--cache_file", cache_file])
    
    # Run the analysis
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("PCA analysis completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running PCA analysis: {e}")
        if e.stderr:
            print("Error details:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: pca_analysis.py not found. Make sure you're in the RanPAC directory.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run PCA analysis on RanPAC experiments')
    parser.add_argument('experiment_paths', nargs='+', 
                       help='Path(s) to experiment directories or base directories to search')
    parser.add_argument('--n_components', type=int, default=50,
                       help='Number of PCA components to compute (default: 50)')
    parser.add_argument('--output_base', type=str, default=None,
                       help='Base output directory (default: use experiment directory)')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search for experiment directories')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recomputation of PCA even if cache exists')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable PCA result caching')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plots from existing cache (no PCA computation)')
    parser.add_argument('--shared_cache', action='store_true',
                       help='Use a shared cache file for all experiments')
    
    args = parser.parse_args()
    
    # Check if pca_analysis.py exists
    if not os.path.exists('pca_analysis.py'):
        print("Error: pca_analysis.py not found in current directory.")
        print("Please run this script from the RanPAC directory.")
        sys.exit(1)
    
    # Collect all experiment directories
    all_exp_dirs = []
    for path in args.experiment_paths:
        if args.recursive:
            exp_dirs = find_experiment_directories(path)
            all_exp_dirs.extend(exp_dirs)
        else:
            if os.path.isdir(path):
                # Check if it has features_h subdirectory
                features_dir = os.path.join(path, "features_h")
                if os.path.exists(features_dir):
                    all_exp_dirs.append(path)
                else:
                    print(f"Warning: No features_h directory found in {path}")
            else:
                print(f"Warning: Directory not found: {path}")
    
    if not all_exp_dirs:
        print("No experiment directories with Features_h found!")
        sys.exit(1)
    
    print(f"Found {len(all_exp_dirs)} experiment(s) to analyze:")
    for exp_dir in all_exp_dirs:
        print(f"  - {exp_dir}")
    
    # Run analysis for each experiment
    successful = 0
    failed = 0
    
    for exp_dir in all_exp_dirs:
        output_dir = None
        cache_file = None
        
        if args.output_base:
            # Create a unique output directory name
            rel_path = os.path.relpath(exp_dir, os.path.commonpath(all_exp_dirs))
            output_dir = os.path.join(args.output_base, rel_path.replace(os.sep, '_'))
        
        if args.shared_cache and args.output_base:
            # Use a shared cache file for all experiments
            cache_file = os.path.join(args.output_base, 'shared_pca_cache.pkl')
        
        success = run_pca_analysis(
            exp_dir, 
            args.n_components, 
            output_dir,
            force_recompute=args.force_recompute,
            no_cache=args.no_cache,
            plot_only=args.plot_only,
            cache_file=cache_file
        )
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"PCA Analysis Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    # Example usage patterns
    if len(sys.argv) == 1:
        print(__doc__)
        print("\nExample usage patterns:")
        print("  # Analyze a specific experiment (with caching)")
        print("  python run_pca_analysis.py logs/ncm/cifar224/0/10/my_experiment")
        print()
        print("  # Force recompute PCA (ignore cache)")
        print("  python run_pca_analysis.py --force_recompute logs/ncm/cifar224/0/10/my_experiment")
        print()
        print("  # Only generate plots from existing cache")
        print("  python run_pca_analysis.py --plot_only logs/ncm/cifar224/0/10/my_experiment")
        print()
        print("  # Analyze all experiments in a directory")
        print("  python run_pca_analysis.py --recursive logs/ncm/cifar224/")
        print()
        print("  # Analyze multiple experiments with shared cache")
        print("  python run_pca_analysis.py --shared_cache --output_base ./results logs/ncm/cifar224/0/10/exp1 logs/adapter/cifar224/0/10/exp2")
        print()
        print("Use --help for full options.")
        sys.exit(0)
    
    main() 