#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct Preference Optimization (DPO) Fine-Tuning Performance Benchmark

This script compares the performance of fine-tuning a small language model using
TRL's Direct Preference Optimization (DPO) method on CPU versus GPU (MPS) on an
Apple Silicon MacBook Pro (M2 Pro). It measures and compares:

1. Training efficiency (time per epoch, memory usage)
2. Model accuracy on validation set

The script is compatible with MacBook Pro with M2 Pro chip and automatically detects
and uses Metal Performance Shaders (MPS) for GPU acceleration when available.

Example usage:
    python test2_cpu_gpu_dpo.py \
        --model_path path/to/base/model \
        --dataset anthropic/hh-rlhf \
        --subset harmless-base \
        --epochs 3 \
        --output_dir ./dpo_outputs

Author: Cascade AI Assistant
Created: April 2025
"""

import os
import time
import argparse
import platform
import statistics
import psutil
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EvalPrediction,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOTrainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from accelerate import Accelerator


def format_size(size_bytes):
    """Format bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Return resident set size in bytes


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics for the model.
    
    Args:
        eval_pred: Evaluation predictions from trainer
        
    Returns:
        Dictionary of metric names and values
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def setup_dpo_trainer(model_path: str, dataset_name: str, dataset_subset: str, 
                      device: str, output_dir: str, num_train_epochs: int, batch_size: int,
                      use_8bit: bool = False) -> Tuple[DPOTrainer, Any, Any]:
    """Set up the DPO trainer for fine-tuning.
    
    Args:
        model_path: Path to the base model
        dataset_name: Name of the dataset on Hugging Face Hub
        dataset_subset: Subset of the dataset
        device: Device to run on ('cpu' or 'mps')
        output_dir: Directory to save model and outputs
        num_train_epochs: Number of training epochs
        batch_size: Batch size for training
        use_8bit: Whether to use 8-bit quantization (for larger models)
        
    Returns:
        Tuple of (trainer, train_dataset, eval_dataset)
    """
    # Start memory usage tracking
    start_memory = get_memory_usage()
    print(f"Initial memory usage: {format_size(start_memory)}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization if needed
    print(f"Loading model on {device} device")
    model_kwargs = {
        "device_map": device,
        "torch_dtype": torch.float16,
    }
    
    if use_8bit:
        print("Using 8-bit quantization for model loading")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model_kwargs["quantization_config"] = quantization_config

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # Set up LoRA configuration for efficient fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Load and prepare dataset
    print(f"Loading dataset: {dataset_name}/{dataset_subset}")
    dataset = load_dataset(dataset_name, dataset_subset)
    
    # Prepare data splits
    # Note: Adjust split ratios based on dataset size
    # For this example we'll use a small portion for speed
    if "train" in dataset and "test" in dataset:
        train_dataset = dataset["train"].select(range(min(1000, len(dataset["train"]))))
        eval_dataset = dataset["test"].select(range(min(100, len(dataset["test"]))))
    else:
        # If no predefined splits, create our own
        dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = dataset["train"].select(range(min(1000, len(dataset["train"]))))
        eval_dataset = dataset["test"].select(range(min(100, len(dataset["test"]))))
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True if device != "cpu" else False,  # Use fp16 on GPU
        bf16=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        report_to="none",  # Disable wandb/tensorboard for benchmarking
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # DPO hyperparameter
        max_length=512,
        max_prompt_length=128,
    )
    
    # Report memory usage after setup
    end_memory = get_memory_usage()
    print(f"Memory usage after setup: {format_size(end_memory)}")
    print(f"Memory increase during setup: {format_size(end_memory - start_memory)}")
    
    return trainer, train_dataset, eval_dataset


def run_dpo_fine_tuning(model_path: str, dataset_name: str, dataset_subset: str, 
                        device: str, output_dir: str, num_train_epochs: int, batch_size: int,
                        use_8bit: bool = False) -> Dict[str, Any]:
    """Run DPO fine-tuning and collect performance metrics.
    
    Args:
        model_path: Path to the base model
        dataset_name: Name of the dataset on Hugging Face Hub
        dataset_subset: Subset of the dataset
        device: Device to run on ('cpu' or 'mps')
        output_dir: Directory to save model and outputs
        num_train_epochs: Number of training epochs
        batch_size: Batch size for training
        use_8bit: Whether to use 8-bit quantization
        
    Returns:
        Dictionary with performance metrics
    """
    # Setup output directory with device name
    device_name = "cpu" if device == "cpu" else "gpu"
    output_dir = os.path.join(output_dir, f"{device_name}_run_{int(time.time())}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics collection
    metrics = {
        "device": device,
        "time_per_epoch": [],
        "memory_usage": [],
        "validation_metrics": [],
    }
    
    try:
        # Setup trainer
        print(f"\n{'='*50}\nSetting up DPO trainer on {device}\n{'='*50}")
        start_time = time.time()
        trainer, train_dataset, eval_dataset = setup_dpo_trainer(
            model_path, dataset_name, dataset_subset, device, 
            output_dir, num_train_epochs, batch_size, use_8bit
        )
        setup_time = time.time() - start_time
        print(f"Trainer setup completed in {setup_time:.2f} seconds")
        
        # Record initial memory
        initial_memory = get_memory_usage()
        metrics["initial_memory"] = initial_memory
        metrics["setup_time"] = setup_time
        
        # Train the model and collect metrics per epoch
        print(f"\n{'='*50}\nStarting DPO training on {device} for {num_train_epochs} epochs\n{'='*50}")
        
        for epoch in range(num_train_epochs):
            print(f"\nEpoch {epoch+1}/{num_train_epochs}")
            
            # Train for one epoch
            epoch_start = time.time()
            train_results = trainer.train()
            epoch_time = time.time() - epoch_start
            
            # Record metrics
            current_memory = get_memory_usage()
            metrics["time_per_epoch"].append(epoch_time)
            metrics["memory_usage"].append(current_memory)
            
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            print(f"Memory usage: {format_size(current_memory)}")
            
            # Evaluate after each epoch
            print("Running evaluation...")
            eval_start = time.time()
            eval_results = trainer.evaluate()
            eval_time = time.time() - eval_start
            
            metrics["validation_metrics"].append({
                "epoch": epoch + 1,
                "eval_time": eval_time,
                **eval_results
            })
            
            print(f"Evaluation completed in {eval_time:.2f} seconds")
            print(f"Evaluation results: {eval_results}")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        final_eval = trainer.evaluate()
        metrics["final_metrics"] = final_eval
        
        # Save model if desired
        if output_dir:
            print(f"\nSaving model to {output_dir}")
            trainer.save_model(output_dir)
        
        # Calculate overall metrics
        metrics["total_train_time"] = sum(metrics["time_per_epoch"])
        metrics["avg_time_per_epoch"] = statistics.mean(metrics["time_per_epoch"])
        metrics["peak_memory_usage"] = max(metrics["memory_usage"])
        metrics["memory_increase"] = metrics["peak_memory_usage"] - metrics["initial_memory"]
        
        return metrics
        
    except Exception as e:
        print(f"Error during fine-tuning on {device}: {str(e)}")
        # Return partial metrics if available
        if len(metrics["time_per_epoch"]) > 0:
            metrics["total_train_time"] = sum(metrics["time_per_epoch"])
            metrics["avg_time_per_epoch"] = statistics.mean(metrics["time_per_epoch"])
            metrics["peak_memory_usage"] = max(metrics["memory_usage"]) if metrics["memory_usage"] else 0
            metrics["error"] = str(e)
        return metrics


def compare_performance(cpu_metrics: Dict[str, Any], gpu_metrics: Dict[str, Any]) -> pd.DataFrame:
    """Compare and visualize performance metrics between CPU and GPU runs.
    
    Args:
        cpu_metrics: Metrics collected from CPU run
        gpu_metrics: Metrics collected from GPU run
        
    Returns:
        DataFrame with comparison metrics
    """
    # Create efficiency comparison DataFrame
    efficiency_data = {
        "Metric": [
            "Total Training Time (s)", 
            "Avg Time per Epoch (s)",
            "Setup Time (s)",
            "Peak Memory Usage", 
            "Memory Increase"
        ],
        "CPU": [
            cpu_metrics.get("total_train_time", 0), 
            cpu_metrics.get("avg_time_per_epoch", 0),
            cpu_metrics.get("setup_time", 0),
            format_size(cpu_metrics.get("peak_memory_usage", 0)),
            format_size(cpu_metrics.get("memory_increase", 0))
        ],
        "GPU (MPS)": [
            gpu_metrics.get("total_train_time", 0), 
            gpu_metrics.get("avg_time_per_epoch", 0),
            gpu_metrics.get("setup_time", 0),
            format_size(gpu_metrics.get("peak_memory_usage", 0)),
            format_size(gpu_metrics.get("memory_increase", 0))
        ],
        "Speedup (CPU/GPU)": [
            cpu_metrics.get("total_train_time", 0) / gpu_metrics.get("total_train_time", 1) 
                if gpu_metrics.get("total_train_time", 0) > 0 else 0,
            cpu_metrics.get("avg_time_per_epoch", 0) / gpu_metrics.get("avg_time_per_epoch", 1)
                if gpu_metrics.get("avg_time_per_epoch", 0) > 0 else 0,
            cpu_metrics.get("setup_time", 0) / gpu_metrics.get("setup_time", 1)
                if gpu_metrics.get("setup_time", 0) > 0 else 0,
            "N/A",  # No speedup metric for memory
            "N/A"
        ]
    }
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Create accuracy comparison if we have final metrics
    if "final_metrics" in cpu_metrics and "final_metrics" in gpu_metrics:
        # Get all available metrics from both runs
        all_metric_keys = set()
        if cpu_metrics["final_metrics"]:
            all_metric_keys.update(cpu_metrics["final_metrics"].keys())
        if gpu_metrics["final_metrics"]:
            all_metric_keys.update(gpu_metrics["final_metrics"].keys())
        
        # Create entries for each metric
        accuracy_data = {
            "Metric": [],
            "CPU": [],
            "GPU (MPS)": [],
            "Difference (GPU-CPU)": []
        }
        
        for key in all_metric_keys:
            # Skip non-numeric or irrelevant metrics
            if key.startswith("eval_"):
                metric_name = key.replace("eval_", "")
                cpu_value = cpu_metrics["final_metrics"].get(key, 0)
                gpu_value = gpu_metrics["final_metrics"].get(key, 0)
                
                accuracy_data["Metric"].append(metric_name)
                accuracy_data["CPU"].append(cpu_value)
                accuracy_data["GPU (MPS)"].append(gpu_value)
                accuracy_data["Difference (GPU-CPU)"].append(gpu_value - cpu_value)
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        # Return both DataFrames
        return efficiency_df, accuracy_df
    
    # If no accuracy metrics, just return efficiency
    return efficiency_df, None


def main():
    """Main function to run the CPU vs GPU comparison benchmark."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compare DPO fine-tuning performance between CPU and GPU on Apple Silicon',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="facebook/opt-350m",  # Small model that should work on both CPU and GPU
        help='Path to the base model (local directory or HF model ID)'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default="anthropic/hh-rlhf",
        help='Dataset name on Hugging Face Hub'
    )
    
    parser.add_argument(
        '--subset', 
        type=str, 
        default="harmless-base",
        help='Dataset subset name'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=4,
        help='Batch size for training and evaluation'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="./dpo_outputs",
        help='Directory to save model and outputs'
    )
    
    parser.add_argument(
        '--cpu_only', 
        action='store_true',
        help='Run only on CPU even if GPU is available'
    )
    
    parser.add_argument(
        '--gpu_only', 
        action='store_true',
        help='Run only on GPU (skips CPU benchmark)'
    )
    
    parser.add_argument(
        '--use_8bit', 
        action='store_true',
        help='Use 8-bit quantization for loading larger models'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for GPU availability (Apple Silicon MPS)
    gpu_available = torch.backends.mps.is_available() and platform.processor() == 'arm'
    
    if gpu_available:
        print("Apple Silicon GPU (MPS) is available - Using Metal Performance Shaders")
    else:
        print("Warning: Apple Silicon GPU acceleration not available. Will run on CPU only.")
        if args.gpu_only:
            print("Error: --gpu_only specified but GPU is not available")
            return
    
    # Run benchmarks based on user selection
    cpu_metrics = None
    gpu_metrics = None
    
    # Run CPU benchmark if requested
    if not args.gpu_only:
        print("\n" + "="*50)
        print(" STARTING CPU BENCHMARK ")
        print("="*50)
        
        cpu_metrics = run_dpo_fine_tuning(
            model_path=args.model_path,
            dataset_name=args.dataset,
            dataset_subset=args.subset,
            device="cpu",
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            batch_size=args.batch_size,
            use_8bit=args.use_8bit
        )
    
    # Run GPU benchmark if available and requested
    if gpu_available and not args.cpu_only:
        print("\n" + "="*50)
        print(" STARTING GPU (MPS) BENCHMARK ")
        print("="*50)
        
        gpu_metrics = run_dpo_fine_tuning(
            model_path=args.model_path,
            dataset_name=args.dataset,
            dataset_subset=args.subset,
            device="mps",
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            batch_size=args.batch_size,
            use_8bit=args.use_8bit
        )
    
    # Compare results if both benchmarks were run
    if cpu_metrics and gpu_metrics:
        print("\n" + "="*50)
        print(" PERFORMANCE COMPARISON RESULTS ")
        print("="*50)
        
        efficiency_df, accuracy_df = compare_performance(cpu_metrics, gpu_metrics)
        
        print("\nEFFICIENCY METRICS:")
        print(efficiency_df.to_string(index=False))
        
        if accuracy_df is not None:
            print("\nACCURACY METRICS:")
            print(accuracy_df.to_string(index=False))
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        efficiency_df.to_csv(f"{args.output_dir}/efficiency_comparison_{timestamp}.csv", index=False)
        if accuracy_df is not None:
            accuracy_df.to_csv(f"{args.output_dir}/accuracy_comparison_{timestamp}.csv", index=False)
        
        print(f"\nResults saved to {args.output_dir}")
    
    elif cpu_metrics:
        print("\nCPU BENCHMARK RESULTS:")
        for key, value in cpu_metrics.items():
            if key not in ["time_per_epoch", "memory_usage", "validation_metrics", "final_metrics"]:
                print(f"{key}: {value}")
    
    elif gpu_metrics:
        print("\nGPU BENCHMARK RESULTS:")
        for key, value in gpu_metrics.items():
            if key not in ["time_per_epoch", "memory_usage", "validation_metrics", "final_metrics"]:
                print(f"{key}: {value}")


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
        import traceback
        traceback.print_exc()
    
    # OPTION 2: Direct execution in VSCode without command line
    # Uncomment and modify this block to run directly in VSCode
    # Delete or comment out the above block (OPTION 1) when using this
    """
    import sys
    # Override sys.argv with the parameters you want to use
    sys.argv = [
        'test2_cpu_gpu_dpo.py',                    # Script name (can be anything)
        '--model_path', 'facebook/opt-350m',        # Path to local model or HF model name
        '--dataset', 'anthropic/hh-rlhf',          # HF dataset name
        '--subset', 'harmless-base',               # Dataset subset
        '--epochs', '3',                           # Number of epochs
        '--batch_size', '4',                       # Batch size
        '--output_dir', './dpo_outputs',           # Output directory
        # '--cpu_only',                            # Uncomment to run only on CPU
        # '--gpu_only',                            # Uncomment to run only on GPU
        # '--use_8bit'                             # Uncomment to use 8-bit quantization
    ]
    main()
    """
