# Hugging Face Performance Benchmarking Tools

This directory contains tools for benchmarking and evaluating Hugging Face transformer models on MacBook Pro with M2 Pro chip, with a focus on performance comparison between CPU and GPU (MPS) execution. The tools are specifically optimized for Apple Silicon GPUs using Metal Performance Shaders.

Two main benchmarking tools are provided:

1. **Inference Benchmarking** (`test1_cpu_gpu_inference.py`): Compare CPU vs GPU performance for model inference
2. **DPO Fine-tuning Benchmarking** (`test2_cpu_gpu_dpo.py`): Compare CPU vs GPU performance for Direct Preference Optimization (DPO) fine-tuning

## Available Scripts

### `test1_cpu_gpu_inference.py`

A comprehensive benchmarking tool that compares the execution time of running a Hugging Face model inference on CPU versus GPU.

#### Features

- Supports both NVIDIA GPUs (via CUDA) and Apple Silicon GPUs (via Metal Performance Shaders)
- Automatically detects available hardware acceleration
- Measures and compares both model loading time and inference time
- Works with any locally downloaded Hugging Face model
- Compatible with MacBook Pro M2 Pro chip using MPS acceleration
- Calculates perplexity scores to evaluate language model quality
- Supports multiple runs for statistical significance
- Compiles results into pandas DataFrames with summary statistics

#### Requirements

- Python 3.6+
- PyTorch 1.12+ (for MPS support on Apple Silicon)
- Transformers library
- pandas and numpy for statistics compilation
- A locally downloaded Hugging Face model

#### Usage

##### Command Line Execution

```bash
# Basic usage
python test1_cpu_gpu.py --model_path /path/to/your/model --prompt "Your custom prompt"

# With perplexity calculation
python test1_cpu_gpu.py --model_path /path/to/your/model --perplexity

# Multiple runs for statistical significance
python test1_cpu_gpu.py --model_path /path/to/your/model --runs 5

# Complete example with all features
python test1_cpu_gpu.py --model_path /path/to/your/model --prompt "Your test prompt" --runs 3 --perplexity
```

##### Direct Execution in VSCode

The script includes built-in options for running directly in VSCode without using the command line:

1. **Option 1: Hardcoded Arguments**
   - Open the script in VSCode
   - Go to the bottom of the file
   - Comment out the "OPTION 1" block
   - Uncomment the "OPTION 2" block
   - Update the model path and other parameters
   - Run the script normally (F5 or Run button)

2. **Option 3: Interactive Testing**
   - Open the script in VSCode
   - Scroll to the bottom where you'll find a code cell marked with `#%%`
   - Uncomment the code inside the triple quotes
   - Update the model path and other parameters
   - Select the cell and press Shift+Enter (or right-click and select "Run Cell")
   - This allows testing individual functions without running the entire script

#### Arguments

- `--model_path` (required): Path to the local model directory containing model files
- `--prompt` (optional): Text prompt to run inference on (default: "The quick brown fox jumps over the lazy dog. Then, the fox")
- `--runs` (optional): Number of runs to perform for statistical significance (default: 1)
- `--perplexity` (optional): Calculate perplexity score as a language model quality metric

#### Example Output

```
Apple Silicon GPU (MPS) is available - Using Metal Performance Shaders

==== Running on CPU ====
Prompt: "The quick brown fox jumps over the lazy dog. Then, the fox"
Number of runs: 3
Loading model and tokenizer on cpu...
Model loaded in 5.23 seconds
Calculating perplexity...
Perplexity score: 12.45 (lower is better)
...

Run 2/3 on cpu
...

Run 3/3 on cpu
...

==== Running on GPU (mps) ====
Prompt: "The quick brown fox jumps over the lazy dog. Then, the fox"
Number of runs: 3
Loading model and tokenizer on mps...
Model loaded in 4.12 seconds
Calculating perplexity...
Perplexity score: 12.43 (lower is better)
...

==== Results Comparison ====
Average CPU load time: 5.31 seconds
Average GPU load time: 4.18 seconds
Average load time speedup: 1.27x
Average CPU inference time: 8.52 seconds
Average GPU inference time: 1.25 seconds
Average inference speedup factor: 6.82x

Average CPU perplexity: 12.45
Average GPU perplexity: 12.43

==== Detailed Results (All Runs) ====
   CPU Load Time (s)  CPU Inference Time (s)  CPU Perplexity  GPU Load Time (s)  GPU Inference Time (s)  GPU Perplexity  Inference Speedup
0             5.23                   8.45           12.45              4.12                   1.23          12.43               6.87
1             5.35                   8.56           12.46              4.20                   1.26          12.44               6.79
2             5.36                   8.54           12.45              4.22                   1.25          12.43               6.83

==== Summary Statistics ====
                 metric    min    max   mean  median    std
 CPU Load Time (s)        5.23   5.36   5.31    5.35   0.07
 CPU Inference Time (s)   8.45   8.56   8.52    8.54   0.06
 CPU Perplexity          12.45  12.46  12.45   12.45   0.01
 GPU Load Time (s)        4.12   4.22   4.18    4.20   0.05
 GPU Inference Time (s)   1.23   1.26   1.25    1.25   0.02
 GPU Perplexity          12.43  12.44  12.43   12.43   0.01
 Inference Speedup        6.79   6.87   6.82    6.83   0.04

Outputs match (first run): False
Note: Different outputs are expected when using sampling (do_sample=True)
```

### `test2_cpu_gpu_dpo.py`

An advanced benchmarking tool that compares the performance of fine-tuning a small language model using TRL's Direct Preference Optimization (DPO) method on CPU versus GPU.

#### Features

- Compares training efficiency between CPU and MPS GPU acceleration
- Measures both training time and model quality metrics
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Supports 8-bit quantization for larger models
- Tracks memory usage throughout training
- Compiles comprehensive statistics in pandas DataFrames
- Exports results to CSV files for further analysis
- Compatible with MacBook Pro with M2 Pro chip using MPS acceleration
- VSCode integration for direct execution without command line

#### Requirements

- Python 3.6+
- PyTorch 1.12+ (for MPS support on Apple Silicon)
- Transformers library
- TRL library for DPO fine-tuning
- PEFT library for parameter-efficient fine-tuning
- Datasets library
- pandas and numpy for statistics compilation
- psutil for memory tracking

#### Usage

##### Command Line Execution

```bash
# Basic usage with a small model
python test2_cpu_gpu_dpo.py --model_path facebook/opt-350m

# Custom dataset example
python test2_cpu_gpu_dpo.py --model_path facebook/opt-350m --dataset anthropic/hh-rlhf --subset harmless-base

# Configure training parameters
python test2_cpu_gpu_dpo.py --model_path facebook/opt-350m --epochs 5 --batch_size 8

# Run only CPU or GPU benchmark
python test2_cpu_gpu_dpo.py --model_path facebook/opt-350m --cpu_only  # CPU only
python test2_cpu_gpu_dpo.py --model_path facebook/opt-350m --gpu_only  # GPU only

# Use 8-bit quantization for larger models
python test2_cpu_gpu_dpo.py --model_path facebook/opt-1.3b --use_8bit
```

##### Direct Execution in VSCode

The script includes built-in options for running directly in VSCode without using the command line:

1. **Option 1: Hardcoded Arguments**
   - Open the script in VSCode
   - Go to the bottom of the file
   - Comment out the "OPTION 1" block
   - Uncomment the "OPTION 2" block
   - Update the model path and other parameters
   - Run the script normally (F5 or Run button)

#### Arguments

- `--model_path` (optional): Path to the base model (local directory or HF model ID) (default: "facebook/opt-350m")
- `--dataset` (optional): Dataset name on Hugging Face Hub (default: "anthropic/hh-rlhf")
- `--subset` (optional): Dataset subset name (default: "harmless-base")
- `--epochs` (optional): Number of training epochs (default: 3)
- `--batch_size` (optional): Batch size for training and evaluation (default: 4)
- `--output_dir` (optional): Directory to save model and outputs (default: "./dpo_outputs")
- `--cpu_only` (flag): Run only on CPU even if GPU is available
- `--gpu_only` (flag): Run only on GPU (skips CPU benchmark)
- `--use_8bit` (flag): Use 8-bit quantization for loading larger models

#### Example Output

```
Apple Silicon GPU (MPS) is available - Using Metal Performance Shaders

==================================================
 STARTING CPU BENCHMARK 
==================================================
Setting up DPO trainer on cpu
Initial memory usage: 123.45 MB
Loading tokenizer from facebook/opt-350m
Loading model on cpu device
...

==================================================
 STARTING GPU (MPS) BENCHMARK 
==================================================
Setting up DPO trainer on mps
Initial memory usage: 125.67 MB
Loading tokenizer from facebook/opt-350m
Loading model on mps device
...

==================================================
 PERFORMANCE COMPARISON RESULTS 
==================================================

EFFICIENCY METRICS:
                Metric         CPU   GPU (MPS)  Speedup (CPU/GPU)
Total Training Time (s)     1256.78     329.45              3.81
Avg Time per Epoch (s)       418.93     109.82              3.82
Setup Time (s)                89.45      76.23              1.17
Peak Memory Usage             4.56 GB    5.23 GB            N/A
Memory Increase               2.34 GB    3.12 GB            N/A

ACCURACY METRICS:
    Metric        CPU   GPU (MPS)  Difference (GPU-CPU)
     loss      0.3456      0.3423              -0.0033
accuracy      0.8723      0.8745               0.0022
      f1      0.8534      0.8567               0.0033
precision     0.8612      0.8645               0.0033
  recall      0.8456      0.8489               0.0033

Results saved to ./dpo_outputs
```

## Best Practices

- For the most accurate benchmarks, close other applications before running tests
- Use the `--runs` parameter with inference benchmarks to perform multiple runs automatically (e.g., `--runs 5`)
- For deterministic comparisons, modify the generation parameters to disable sampling
- The first run may be slower due to compilation/optimization overhead
- When running in VSCode, use the interactive cell option for quick testing of specific functions
- For production use or batch processing, use the command-line interface
- For fine-tuning benchmarks, start with a small model like OPT-350M before trying larger models
- When benchmarking larger models with the DPO script, use the `--use_8bit` flag to reduce memory usage

## Troubleshooting

### General Issues
- If you encounter "MPS not available" errors on Apple Silicon, ensure you have PyTorch 1.12+ installed
- For CUDA errors on NVIDIA systems, verify that you have the correct CUDA toolkit installed
- If running in VSCode and getting argument errors, check that you've properly configured the arguments in Option 2 or are using the interactive cell correctly

### Inference Benchmark Issues
- If you get "out of memory" errors during inference, try reducing the model size or batch size
- For import errors with the inference benchmark: `pip install torch pandas numpy transformers`

### DPO Fine-tuning Issues
- For fine-tuning out of memory errors, try:
  - Using the `--use_8bit` quantization flag
  - Reducing the batch size with `--batch_size`
  - Using a smaller model (e.g., OPT-350M)
- If you encounter issues with TRL or PEFT, install the required dependencies:
  ```bash
  pip install trl peft accelerate datasets psutil scipy scikit-learn
  ```
- For Mac Silicon specific fine-tuning issues:
  - Ensure you are running PyTorch 2.0+ for best MPS support
  - Some operations may fall back to CPU execution - this is normal with the current state of MPS support
