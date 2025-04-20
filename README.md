# Hugging Face Inference Benchmarking Tools

This directory contains tools for benchmarking and evaluating Hugging Face transformer models, with a focus on performance comparison between CPU and GPU execution.

## Available Scripts

### `test1_cpu_gpu.py`

A comprehensive benchmarking tool that compares the execution time of running a Hugging Face model on CPU versus GPU.

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

## Best Practices

- For the most accurate benchmarks, close other applications before running tests
- Use the `--runs` parameter to perform multiple runs automatically (e.g., `--runs 5`)
- For deterministic comparisons, modify the generation parameters to disable sampling
- The first run may be slower due to compilation/optimization overhead
- When running in VSCode, use the interactive cell option for quick testing of specific functions
- For production use or batch processing, use the command-line interface

## Troubleshooting

- If you encounter "MPS not available" errors on Apple Silicon, ensure you have PyTorch 1.12+ installed
- For CUDA errors on NVIDIA systems, verify that you have the correct CUDA toolkit installed
- If you get "out of memory" errors, try reducing the model size or batch size
- If running in VSCode and getting argument errors, check that you've properly configured the arguments in Option 2 or are using the interactive cell correctly
- For import errors, ensure you have all dependencies installed: `pip install torch pandas numpy transformers`
