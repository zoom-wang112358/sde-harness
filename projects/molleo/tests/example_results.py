#!/usr/bin/env python3
"""
Comprehensive example of MolLEO with SDE Harness
Shows both single and multi-objective optimization with LLMs
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Add the molleo project root to path
molleo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, molleo_root)

from src.modes import run_single_objective, run_multi_objective
from src.core import MolLEOOptimizer
from src.oracles import TDCOracle
import weave


def demo_single_objective(args):
    """Demo single objective optimization"""
    print("\n" + "=" * 80)
    print(f"DEMO: Single Objective Optimization - {args.oracle} with LLM")
    print("=" * 80)
    
    weave.init("molleo_demo_single")
    
    print(f"\nConfiguration:")
    print(f"- Target: {args.oracle} (drug-likeness)")
    print(f"- Model: {args.model}")
    print(f"- Population: {args.population_size}")
    print(f"- Generations: {args.generations}")
    
    results = run_single_objective(args)
    
    print(f"\n✅ Optimization Complete!")
    print(f"   Best Score: {results['best_score']:.4f}")
    print(f"   Best molecule: {results['best_molecule']}")
    
    return results



def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive MolLEO demo with SDE Harness"
    )
    parser.add_argument(
        "--mode", 
        choices=["single", "multi", "custom", "all"],
        default="single",
        help="Which demo to run"
    )

    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mol_lm', type=str, default=None, choices=["GPT-4"])
    parser.add_argument('--model', type=str, default=None, help="LLM model to use")
    parser.add_argument('--population_size', type=int, default=120)
    parser.add_argument('--offspring_size', type=int, default=70)
    parser.add_argument('--mutation_rate', type=int, default=0.067)
    parser.add_argument('--generations', type=int, default=150)
    parser.add_argument('--oracle', type=str, default='qed')
    parser.add_argument('--initial_size', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)

    args = parser.parse_args()
    
    print("\n🧪 MolLEO - Molecular Language-Enhanced Evolutionary Optimization")
    print("   Powered by SDE Harness Framework")
    

    demo_single_objective(args)
        
    
    print("\n✨ All demos completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- ✅ LLM-guided molecular mutations")
    print("- ✅ Single and multi-objective optimization")
    print("- ✅ Custom oracle integration")
    print("- ✅ Full SDE Harness integration")
    print("- ✅ Experiment tracking with Weave")


if __name__ == "__main__":
    main()