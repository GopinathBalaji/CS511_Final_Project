# Benchmarking Speculative Decoding in vLLM

This project aims to benchmark the performance of speculative decoding in vLLM, an optimized inference and serving engine for large language models (LLMs). The goal is to measure the speedup and efficiency gained by using speculative decoding under various configurations.

## Introduction

Speculative decoding is a technique to accelerate text generation from language models by allowing the model to “speculate” multiple tokens ahead. This reduces latency and improves throughput without significantly impacting the quality of the generated text. vLLM provides support for speculative decoding, making it easier to leverage this technique for faster inference.


## Installation

To reproduce the experiments, follow these steps to set up the environment:

1. **Create a Conda Environment**:

    ```bash
    conda create -n vllm python=3.9
    conda activate vllm
    ```

2. **Install vLLM and Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    Note: Ensure you have CUDA and GPU drivers installed if you plan to run the experiments on a GPU.

3. **Verify Installation**:
    - Run `python -c "import vllm"` to ensure vLLM is installed correctly.
    - Check your CUDA installation with `nvidia-smi`.

## Project Structure

The project is organized into the following files:
- **main.py**: The main script that orchestrates the experiment.
- **experiment.py**: Contains the core functions and logic for running the experiments.
- **config.yaml**: A YAML configuration file specifying the experiment parameters.
- **README.md**: Documentation and instructions for reproducing the experiments.
- **requirements.txt**: A list of Python packages required for the project.

## Configuration

Edit the `config.yaml` file to adjust the experiment parameters:

```yaml
# config.yaml

# Experiment parameters
num_requests_list:
  - 32
  - 128
  - 512
batch_size_list:
  - 1
  - 4
  - 8
  - 16
  - 32
speculative_models:
  - null        # Represents no speculative decoding
  - '[ngram]'   # N-gram speculative model
  - facebook/opt-125m
  - facebook/opt-350m

# Model settings
target_model_name: facebook/opt-1.3b
num_speculative_tokens_list:
  - 1
  - 3
  - 5
  - 10
gpu_memory_utilization: 0.9

# N-gram model settings
ngram_prompt_lookup_max: 4
ngram_prompt_lookup_min: 1

# Prompt settings
base_prompt: "The future of AI is"

# Output settings
output_csv: 'vllm_benchmark_results.csv'
```

- **num_requests_list**: List of total prompts to process.
- **batch_size_list**: List of batch sizes to test.
- **speculative_models**: List of speculative models to use (null represents no speculative decoding).
- **num_speculative_tokens_list**: Different values for the number of speculative tokens.
- **gpu_memory_utilization**: Fraction of GPU memory to utilize (0.0 to 1.0).
- **ngram_prompt_lookup_max** and **ngram_prompt_lookup_min**: Parameters for the n-gram speculative model.
- **base_prompt**: The base text for prompts.
- **output_csv**: Filename for saving the results.

## Running the Experiment

1. **Run the Main Script**:

    ```bash
    python main.py
    ```

2. **Monitor GPU Memory Usage**:
    Use `nvidia-smi` to monitor GPU memory usage during the experiment:

    ```bash
    watch -n 1 nvidia-smi
    ```

    Note: Adjust batch sizes or models in `config.yaml` if you encounter memory issues.

3. **View the Results**:
    - The experiment results will be printed to the console.
    - Results are also saved to the CSV file specified in `config.yaml` (default is `vllm_benchmark_results.csv`) in the `raw_data/` directory.
    - Plots will be displayed if matplotlib is installed.

4. **Analyzing the Results**:
    - Run `python analyze_results.py <path_to_csv>` 
    - The resulting graphs will be stored under `visualizations/`.
    
## References

- [Accelerating Language Model Decoding via Speculative Sampling](#)
- [vLLM GitHub Repository](#)
- [Accelerating Text Generation with Speculative Decoding (Research Paper)](#)
