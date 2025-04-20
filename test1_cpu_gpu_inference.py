#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Huggingface Model Inference Performance Benchmark.

This script benchmarks the performance difference between CPU and GPU inference
for locally stored Huggingface transformer models. It supports both NVIDIA GPUs
(via CUDA) and Apple Silicon GPUs (via Metal Performance Shaders).

Features:
    - CPU vs GPU performance comparison
    - Perplexity calculation
    - Multiple runs for statistical significance
    - Results compilation in pandas DataFrame

Example usage:
    python test1_cpu_gpu.py --model_path /path/to/local/model --prompt "Your test prompt" --runs 3 --perplexity

Author: Cascade AI Assistant
Created: April 2025
"""

import argparse
import time
import math
import statistics
from typing import Dict, Union, List, Tuple, Optional

import torch
import platform
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_perplexity(model, tokenizer, text: str, device: str) -> float:
    """Calculate the perplexity score for the given text.
    
    Perplexity measures how well a language model predicts a text sample.
    Lower perplexity indicates better prediction (the model is less "perplexed").
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        text: Text to calculate perplexity for
        device: Device to run calculation on ('cpu', 'cuda', or 'mps')
        
    Returns:
        Perplexity score (lower is better)
    """
    # Encode the text
    encodings = tokenizer(text, return_tensors="pt").to(device)
    
    # Get the input IDs and create target IDs (shifted by 1)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    
    # Calculate loss with no gradient accumulation
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    
    # Calculate perplexity as e^(average negative log-likelihood)
    ppl = torch.exp(neg_log_likelihood).item()
    
    return ppl


def run_inference(model_path: str, prompt: str, device: str, calculate_ppl: bool = False) -> Dict[str, Union[float, str, float]]:
    """Run inference on the given model and prompt using the specified device.
    
    This function handles the complete inference pipeline including:
    1. Loading the model and tokenizer from the specified path
    2. Tokenizing the input prompt
    3. Generating text using the model
    4. Measuring both load time and inference time separately
    5. Optionally calculating perplexity
    
    Args:
        model_path: Path to the local Huggingface model directory
        prompt: Text prompt to use for inference
        device: Device to run inference on ('cpu', 'cuda', or 'mps')
        calculate_ppl: Whether to calculate perplexity score
        
    Returns:
        Dictionary containing:
            - load_time: Time taken to load the model (seconds)
            - inference_time: Time taken for text generation (seconds)
            - result: Generated text output
            - perplexity: Perplexity score (if calculate_ppl is True)
    
    Note:
        The model is loaded with float16 precision to optimize for both performance
        and memory usage, which works well on most modern GPUs.
    """
    # Start measuring model load time
    print(f"Loading model and tokenizer on {device}...")
    start_load = time.time()
    
    # Load tokenizer first (typically faster than model loading)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # Use local files only to avoid network latency affecting benchmarks
        local_files_only=True
    )
    
    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # Use float16 precision for better performance and memory efficiency
        torch_dtype=torch.float16,
        # Map model to the specified device (cpu/cuda/mps)
        device_map=device,
        # Use local files only to avoid network latency affecting benchmarks
        local_files_only=True
    )
    
    # Calculate and report model load time
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Calculate perplexity if requested
    perplexity = None
    if calculate_ppl:
        print("Calculating perplexity...")
        perplexity = calculate_perplexity(model, tokenizer, prompt, device)
        print(f"Perplexity score: {perplexity:.2f} (lower is better)")
    
    # Tokenize input prompt and move tensors to the target device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Start measuring inference time (generation only, not including tokenization)
    start_inference = time.time()
    
    # Disable gradient calculation for inference to reduce memory usage and improve speed
    with torch.no_grad():
        # Generate text using the model
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,        # Limit output sequence length
            do_sample=True,        # Use sampling for more diverse outputs
            temperature=0.7,       # Control randomness (lower = more deterministic)
            # Additional parameters could be added here based on requirements:
            # top_p=0.9,           # Use nucleus sampling (helps with quality)
            # num_beams=1,         # Simple greedy sampling for faster inference
            # repetition_penalty=1.1  # Discourage repetition in generation
        )
    
    # Calculate time spent on text generation
    inference_time = time.time() - start_inference
    
    # Decode the output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Prepare return dictionary
    results = {
        "load_time": load_time,
        "inference_time": inference_time,
        "result": result
    }
    
    # Add perplexity if calculated
    if perplexity is not None:
        results["perplexity"] = perplexity
    
    return results


def run_multiple_inferences(model_path: str, prompt: str, device: str, num_runs: int, calculate_ppl: bool) -> List[Dict]:
    """Run inference multiple times and collect results.
    
    Args:
        model_path: Path to the local model directory
        prompt: Text prompt to use for inference
        device: Device to run on ('cpu', 'cuda', or 'mps')
        num_runs: Number of runs to perform
        calculate_ppl: Whether to calculate perplexity
        
    Returns:
        List of result dictionaries from each run
    """
    results = []
    
    # First run loads the model, subsequent runs reuse it
    first_result = run_inference(model_path, prompt, device, calculate_ppl)
    results.append(first_result)
    
    # For subsequent runs, we'll need to reload the model each time for fair comparison
    # This simulates real-world usage where the model is loaded fresh each time
    for i in range(1, num_runs):
        print(f"\nRun {i+1}/{num_runs} on {device}")
        result = run_inference(model_path, prompt, device, calculate_ppl)
        results.append(result)
    
    return results


def compile_statistics(cpu_results: List[Dict], gpu_results: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Compile statistics from multiple runs into a pandas DataFrame.
    
    Args:
        cpu_results: List of result dictionaries from CPU runs
        gpu_results: List of result dictionaries from GPU runs (if available)
        
    Returns:
        DataFrame with compiled statistics
    """
    # Extract metrics
    stats = {
        'CPU Load Time (s)': [r['load_time'] for r in cpu_results],
        'CPU Inference Time (s)': [r['inference_time'] for r in cpu_results],
    }
    
    # Add perplexity if available
    if 'perplexity' in cpu_results[0]:
        stats['CPU Perplexity'] = [r['perplexity'] for r in cpu_results]
    
    # Add GPU stats if available
    if gpu_results:
        stats['GPU Load Time (s)'] = [r['load_time'] for r in gpu_results]
        stats['GPU Inference Time (s)'] = [r['inference_time'] for r in gpu_results]
        
        # Calculate speedup for each run
        stats['Inference Speedup'] = [c/g if g > 0 else 0 
                                    for c, g in zip([r['inference_time'] for r in cpu_results],
                                                   [r['inference_time'] for r in gpu_results])]
        
        # Add GPU perplexity if available
        if 'perplexity' in gpu_results[0]:
            stats['GPU Perplexity'] = [r['perplexity'] for r in gpu_results]
    
    # Create DataFrame
    df = pd.DataFrame(stats)
    
    # Add summary statistics
    summary = pd.DataFrame({
        'metric': list(stats.keys()),
        'min': [min(values) for values in stats.values()],
        'max': [max(values) for values in stats.values()],
        'mean': [statistics.mean(values) for values in stats.values()],
        'median': [statistics.median(values) for values in stats.values()],
        'std': [statistics.stdev(values) if len(values) > 1 else 0 for values in stats.values()]
    })
    
    return df, summary


def main() -> None:
    """Main function to execute the benchmark.
    
    This function handles command-line argument parsing, device detection,
    and orchestrates running the benchmark on both CPU and available GPU.
    It then calculates and displays performance metrics for comparison.
    
    Returns:
        None
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compare inference time on CPU vs GPU for a local Huggingface model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help='Path to the local model directory containing model files'
    )
    
    parser.add_argument(
        '--prompt', 
        type=str, 
        default="The quick brown fox jumps over the lazy dog. Then, the fox", 
        help='Text prompt to run inference on'
    )
    
    parser.add_argument(
        '--runs', 
        type=int, 
        default=1, 
        help='Number of runs to perform for statistical significance'
    )
    
    parser.add_argument(
        '--perplexity', 
        action='store_true', 
        help='Calculate perplexity score (language model quality metric)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Detect available GPU acceleration options based on hardware
    # Order of preference: MPS (Apple Silicon) -> CUDA (NVIDIA) -> CPU only
    
    # Check if running on Apple Silicon with MPS support
    # Both conditions are necessary as MPS backend might be available on non-M-series Macs
    if torch.backends.mps.is_available() and platform.processor() == 'arm':
        # Apple Silicon detected (M1/M2/M3 series) with Metal Performance Shaders
        gpu_type = "mps"
        gpu_available = True
        print("Apple Silicon GPU (MPS) is available - Using Metal Performance Shaders")
    
    # Check for NVIDIA GPU with CUDA support as fallback
    elif torch.cuda.is_available():
        # NVIDIA GPU with CUDA support detected
        gpu_type = "cuda"
        gpu_available = True
        print(f"NVIDIA GPU (CUDA) is available - {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # No GPU acceleration available, will run CPU-only benchmark
    else:
        gpu_available = False
        print("Warning: No GPU acceleration is available. Will only run on CPU.")
        print(f"CPU: {platform.processor()}")
    
    # First run inference on CPU for baseline measurements
    # This is always performed regardless of GPU availability
    print("\n==== Running on CPU ====")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Number of runs: {args.runs}")
    
    # Execute inference on CPU multiple times if requested
    cpu_results_list = run_multiple_inferences(
        args.model_path, args.prompt, "cpu", args.runs, args.perplexity
    )
    
    # Run on GPU only if hardware acceleration is available
    if gpu_available:
        print(f"\n==== Running on GPU ({gpu_type}) ====")
        print(f"Prompt: \"{args.prompt}\"")
        print(f"Number of runs: {args.runs}")
        
        # Execute inference on detected GPU multiple times if requested
        gpu_results_list = run_multiple_inferences(
            args.model_path, args.prompt, gpu_type, args.runs, args.perplexity
        )
        
        # Compile statistics from all runs
        results_df, summary_df = compile_statistics(cpu_results_list, gpu_results_list)
        
        # Compare and display performance metrics between CPU and GPU
        print("\n==== Results Comparison ====")
        
        # For multiple runs, use the mean values
        cpu_load_mean = statistics.mean([r['load_time'] for r in cpu_results_list])
        gpu_load_mean = statistics.mean([r['load_time'] for r in gpu_results_list])
        load_speedup = cpu_load_mean / gpu_load_mean if gpu_load_mean > 0 else 0
        
        cpu_inference_mean = statistics.mean([r['inference_time'] for r in cpu_results_list])
        gpu_inference_mean = statistics.mean([r['inference_time'] for r in gpu_results_list])
        inference_speedup = cpu_inference_mean / gpu_inference_mean if gpu_inference_mean > 0 else 0
        
        # Display average performance metrics
        print(f"Average CPU load time: {cpu_load_mean:.2f} seconds")
        print(f"Average GPU load time: {gpu_load_mean:.2f} seconds")
        print(f"Average load time speedup: {load_speedup:.2f}x")
        
        print(f"Average CPU inference time: {cpu_inference_mean:.2f} seconds")
        print(f"Average GPU inference time: {gpu_inference_mean:.2f} seconds")
        print(f"Average inference speedup factor: {inference_speedup:.2f}x")
        
        # Display perplexity if calculated
        if args.perplexity:
            cpu_ppl_mean = statistics.mean([r['perplexity'] for r in cpu_results_list])
            gpu_ppl_mean = statistics.mean([r['perplexity'] for r in gpu_results_list])
            print(f"\nAverage CPU perplexity: {cpu_ppl_mean:.2f}")
            print(f"Average GPU perplexity: {gpu_ppl_mean:.2f}")
        
        # Display DataFrame with all runs
        print("\n==== Detailed Results (All Runs) ====")
        print(results_df.to_string(index=True))
        
        # Display summary statistics
        print("\n==== Summary Statistics ====")
        print(summary_df.to_string(index=False))
        
        # Check if outputs match to verify correct functioning
        # Just compare the first run for simplicity
        outputs_match = cpu_results_list[0]['result'] == gpu_results_list[0]['result']
        print(f"\nOutputs match (first run): {outputs_match}")
        
        # If outputs differ significantly, provide additional information
        if not outputs_match:
            print("Note: Different outputs are expected when using sampling (do_sample=True)")
            print("For deterministic comparison, modify the generation parameters")
    else:
        # Compile statistics from CPU runs only
        results_df, summary_df = compile_statistics(cpu_results_list)
        
        # For CPU-only systems, just display the CPU performance metrics
        print("\n==== Results (CPU Only) ====")
        
        # For multiple runs, use the mean values
        cpu_load_mean = statistics.mean([r['load_time'] for r in cpu_results_list])
        cpu_inference_mean = statistics.mean([r['inference_time'] for r in cpu_results_list])
        
        print(f"Average CPU load time: {cpu_load_mean:.2f} seconds")
        print(f"Average CPU inference time: {cpu_inference_mean:.2f} seconds")
        
        # Display perplexity if calculated
        if args.perplexity:
            cpu_ppl_mean = statistics.mean([r['perplexity'] for r in cpu_results_list])
            print(f"\nAverage CPU perplexity: {cpu_ppl_mean:.2f}")
        
        # Display DataFrame with all runs
        print("\n==== Detailed Results (All Runs) ====")
        print(results_df.to_string(index=True))
        
        # Display summary statistics
        print("\n==== Summary Statistics ====")
        print(summary_df.to_string(index=False))
        
        print("\nNote: For GPU acceleration, run this script on a system with:")
        print("  - An Apple Silicon Mac (M1/M2/M3) with PyTorch MPS support")
        print("  - A system with an NVIDIA GPU and CUDA support")


if __name__ == "__main__":
    # OPTION 1: Normal command-line execution
    # Uncomment this block for normal usage with command-line arguments
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        # Re-raise the exception for debugging if needed
        raise
    
    # OPTION 2: Direct execution in VSCode without command line
    # Uncomment and modify this block to run directly in VSCode
    # Delete or comment out the above block (OPTION 1) when using this
    """
    import sys
    # Override sys.argv with the parameters you want to use
    sys.argv = [
        'test1_cpu_gpu.py',                  # Script name (can be anything)
        '--model_path', '/path/to/your/model',  # CHANGE THIS to your model path
        '--prompt', 'Your test prompt here',     # Optional: custom prompt
        '--runs', '3',                          # Optional: number of runs
        '--perplexity'                          # Optional: calculate perplexity
    ]
    main()
    """
    
    # OPTION 3: Interactive testing in VSCode Python Interactive Window
    # Use the cell below with #%% to run interactively
    # You can select this cell and press Shift+Enter or right-click and select "Run Cell"
    
#%%
# Uncomment this cell for interactive testing in VSCode
"""
# Interactive test of specific functions
# This won't run automatically - you need to select it and run it manually

# Test parameters
model_path = "/path/to/your/model"  # CHANGE THIS to your model path
prompt = "Your test prompt here"
device = "cpu"  # or "mps" for Apple Silicon / "cuda" for NVIDIA
calculate_ppl = True

# Test a single inference
result = run_inference(model_path, prompt, device, calculate_ppl)
print(result)
"""