#!/usr/bin/env python
"""
Results Analysis Script for Model Optimization Evaluation

This script analyzes and compares results from all evaluation types:
- Standard (baseline)
- L-Mul
- FP8
- Combined L-Mul + FP8

Provides clear performance comparisons and summaries across all optimization techniques.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results
    python analyze_results.py --compare standard lmul
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def find_result_directories(results_dir="results"):
    """Find all result directories and categorize them by type."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found.")
        return {}
    
    evaluation_types = {
        "standard": [],
        "l_mul": [],
        "fp8": [],
        "lmul_fp8": []
    }
    
    for item in results_path.iterdir():
        if item.is_dir():
            if "_standard_" in item.name:
                evaluation_types["standard"].append(item)
            elif "_lmul_fp8_" in item.name:
                evaluation_types["lmul_fp8"].append(item)
            elif "_l_mul_" in item.name:
                evaluation_types["l_mul"].append(item)
            elif "_fp8_" in item.name:
                evaluation_types["fp8"].append(item)
    
    # Sort all lists
    for eval_type in evaluation_types:
        evaluation_types[eval_type].sort()
    
    return evaluation_types

def load_task_results(result_dir):
    """Load all task results from a result directory."""
    results = {}
    for json_file in result_dir.glob("*.json"):
        if not json_file.name.endswith("_error.json"):
            with open(json_file, 'r') as f:
                results[json_file.stem] = json.load(f)
    return results

def extract_performance_metrics(task_results):
    """Extract performance metrics from task results."""
    all_metrics = []
    
    for task_name, task_data in task_results.items():
        # Handle new structure where metrics are at task level
        if isinstance(task_data, dict) and "performance_metrics" in task_data:
            metrics = task_data["performance_metrics"].copy()
            metrics["task"] = task_name
            all_metrics.append(metrics)
    
    return all_metrics

def aggregate_metrics(metrics_list):
    """Aggregate metrics across all tasks."""
    if not metrics_list:
        return {}
    
    df = pd.DataFrame(metrics_list)
    
    # Calculate aggregated metrics
    aggregated = {
        "total_duration_seconds": df["duration_seconds"].sum(),
        "avg_tokens_per_second": df["tokens_per_second"].mean(),
        "total_tokens": df["total_new_tokens"].sum(),
        "avg_cpu_usage": df["cpu_percent_usage"].mean(),
        "peak_ram_gb": df["ram_usage_gb"].max(),
        "peak_vram_gb": df["vram_peak_gb"].max() if "vram_peak_gb" in df else 0,
        "total_emissions_kg_co2": df["emissions_kg_co2"].sum(),
        "total_energy_kwh": df["energy_consumed_kwh"].sum(),
        "num_tasks": len(df),
    }
    
    return aggregated

def compare_two_results(baseline_metrics, optimized_metrics, baseline_name="baseline", optimized_name="optimized"):
    """Compare two sets of metrics."""
    if not baseline_metrics or not optimized_metrics:
        return {}
    
    comparison = {}
    
    # Calculate speedup/efficiency ratios
    if baseline_metrics["avg_tokens_per_second"] > 0:
        comparison["tokens_per_second_ratio"] = optimized_metrics["avg_tokens_per_second"] / baseline_metrics["avg_tokens_per_second"]
        comparison["speedup_percentage"] = (comparison["tokens_per_second_ratio"] - 1) * 100
    
    if baseline_metrics["total_energy_kwh"] > 0:
        comparison["energy_efficiency_ratio"] = baseline_metrics["total_energy_kwh"] / optimized_metrics["total_energy_kwh"]
        comparison["energy_savings_percentage"] = (1 - optimized_metrics["total_energy_kwh"] / baseline_metrics["total_energy_kwh"]) * 100
    
    if baseline_metrics["total_emissions_kg_co2"] > 0:
        comparison["emissions_reduction_ratio"] = baseline_metrics["total_emissions_kg_co2"] / optimized_metrics["total_emissions_kg_co2"]
        comparison["emissions_reduction_percentage"] = (1 - optimized_metrics["total_emissions_kg_co2"] / baseline_metrics["total_emissions_kg_co2"]) * 100
    
    comparison["duration_difference_seconds"] = optimized_metrics["total_duration_seconds"] - baseline_metrics["total_duration_seconds"]
    comparison["vram_difference_gb"] = optimized_metrics["peak_vram_gb"] - baseline_metrics["peak_vram_gb"]
    
    return comparison

def print_detailed_comparison(eval_types: Dict, compare_types: List[str] = None):
    """Print detailed comparison between evaluation types."""
    available_types = [et for et in eval_types if eval_types[et]]
    
    if not available_types:
        print("No evaluation results found.")
        return
    
    # Default comparison: all available types
    if compare_types is None:
        compare_types = available_types
    else:
        # Filter to only available types
        compare_types = [ct for ct in compare_types if ct in available_types]
    
    if len(compare_types) < 2:
        print("Need at least 2 evaluation types for comparison.")
        return
    
    print("=" * 100)
    print("Model Optimization Evaluation Results")
    print("=" * 100)
    
    # Load latest results for each type
    latest_results = {}
    for eval_type in compare_types:
        if eval_types[eval_type]:
            latest_dir = eval_types[eval_type][-1]  # Get most recent
            results = load_task_results(latest_dir)
            metrics_list = extract_performance_metrics(results)
            aggregated = aggregate_metrics(metrics_list)
            latest_results[eval_type] = {
                "dir": latest_dir,
                "results": results,
                "metrics": aggregated
            }
    
    # Print individual summaries
    print("\nIndividual Evaluation Summaries:")
    print("-" * 100)
    
    for eval_type, data in latest_results.items():
        print(f"\n{eval_type.upper()} Evaluation:")
        print(f"  Directory: {data['dir'].name}")
        print(f"  Tasks completed: {data['metrics'].get('num_tasks', 0)}")
        print(f"  Avg tokens/sec: {data['metrics'].get('avg_tokens_per_second', 0):.2f}")
        print(f"  Total energy: {data['metrics'].get('total_energy_kwh', 0):.6f} kWh")
        print(f"  Peak VRAM: {data['metrics'].get('peak_vram_gb', 0):.2f} GB")
        print(f"  CO2 emissions: {data['metrics'].get('total_emissions_kg_co2', 0):.6f} kg")
    
    # Pairwise comparisons with standard as baseline
    if "standard" in latest_results and len(compare_types) > 1:
        print("\n\nComparisons vs Standard Baseline:")
        print("=" * 100)
        
        baseline = latest_results["standard"]["metrics"]
        
        for eval_type in compare_types:
            if eval_type == "standard":
                continue
            
            print(f"\n{eval_type.upper()} vs STANDARD:")
            print("-" * 50)
            
            optimized = latest_results[eval_type]["metrics"]
            comparison = compare_two_results(baseline, optimized, "standard", eval_type)
            
            if "speedup_percentage" in comparison:
                speedup = comparison["speedup_percentage"]
                print(f"  Throughput: {'+' if speedup > 0 else ''}{speedup:.1f}% "
                      f"({comparison['tokens_per_second_ratio']:.2f}x)")
            
            if "energy_savings_percentage" in comparison:
                savings = comparison["energy_savings_percentage"]
                print(f"  Energy savings: {'+' if savings > 0 else ''}{savings:.1f}% "
                      f"({comparison['energy_efficiency_ratio']:.2f}x more efficient)")
            
            if "emissions_reduction_percentage" in comparison:
                reduction = comparison["emissions_reduction_percentage"]
                print(f"  CO2 reduction: {'+' if reduction > 0 else ''}{reduction:.1f}% "
                      f"({comparison['emissions_reduction_ratio']:.2f}x less)")
            
            vram_diff = comparison.get("vram_difference_gb", 0)
            print(f"  VRAM difference: {'+' if vram_diff > 0 else ''}{vram_diff:.2f} GB")
    
    # Summary table
    if len(compare_types) > 2:
        print("\n\nSummary Table:")
        print("=" * 100)
        print(f"{'Optimization':<15} {'Tokens/sec':<12} {'vs Standard':<12} {'Energy (kWh)':<15} {'vs Standard':<12} {'VRAM (GB)':<10}")
        print("-" * 100)
        
        standard_metrics = latest_results.get("standard", {}).get("metrics", {})
        
        for eval_type in compare_types:
            metrics = latest_results[eval_type]["metrics"]
            tokens_sec = metrics.get("avg_tokens_per_second", 0)
            energy = metrics.get("total_energy_kwh", 0)
            vram = metrics.get("peak_vram_gb", 0)
            
            if eval_type == "standard":
                tokens_vs = "-"
                energy_vs = "-"
            else:
                if standard_metrics and standard_metrics.get("avg_tokens_per_second", 0) > 0:
                    tokens_ratio = tokens_sec / standard_metrics["avg_tokens_per_second"]
                    tokens_vs = f"{tokens_ratio:.2f}x"
                else:
                    tokens_vs = "N/A"
                
                if standard_metrics and standard_metrics.get("total_energy_kwh", 0) > 0:
                    energy_ratio = standard_metrics["total_energy_kwh"] / energy
                    energy_vs = f"{energy_ratio:.2f}x"
                else:
                    energy_vs = "N/A"
            
            print(f"{eval_type:<15} {tokens_sec:<12.2f} {tokens_vs:<12} {energy:<15.6f} {energy_vs:<12} {vram:<10.2f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze model optimization evaluation results")
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Directory containing evaluation results")
    parser.add_argument("--compare", type=str, nargs="+", 
                       choices=["standard", "l_mul", "fp8", "lmul_fp8"],
                       help="Specific evaluation types to compare")
    
    args = parser.parse_args()
    
    eval_types = find_result_directories(args.results_dir)
    print_detailed_comparison(eval_types, args.compare)

if __name__ == "__main__":
    main() 