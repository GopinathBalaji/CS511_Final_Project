# experiment.py

import time
from vllm import LLM
from transformers import AutoTokenizer
import numpy as np
import torch 
from tqdm import tqdm

def generate_prompts(num_prompts, base_prompt="The future of AI is"):
    """Generate a list of prompts for testing."""
    return [f"{base_prompt} #{i}" for i in range(num_prompts)]

def measure_metrics(llm, prompts, batch_size, tokenizer):
    """Measure latency and calculate throughput metrics."""
    num_requests = len(prompts)
    total_input_tokens = 0
    total_output_tokens = 0
    total_sequences = num_requests
    latencies = []
    
    # total_accept_rate = 0.0
    # num_batches_with_accept_rate = 0

    # Process prompts in batches
    for i in range(0, num_requests, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        start_time = time.time()
        outputs = llm.generate(batch_prompts)
        end_time = time.time()

        latency = end_time - start_time
        latencies.append(latency)

        # Calculate tokens using the tokenizer
        for output in outputs:
            input_tokens = len(tokenizer.encode(output.prompt, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output.outputs[0].text, add_special_tokens=False))
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # Get accept rate if speculative decoding is used
        # if hasattr(llm.engine, 'speculative_decoding_stats'):
            # accept_rate = llm.engine.speculative_decoding_stats.accept_rate
            # total_accept_rate += accept_rate
            # num_batches_with_accept_rate += 1

    # Calculate average latency
    average_latency = np.mean(latencies)

    # Calculate throughputs
    total_time = sum(latencies)
    output_token_throughput = total_output_tokens / total_time if total_time > 0 else 0
    total_token_throughput = (total_input_tokens + total_output_tokens) / total_time if total_time > 0 else 0
    sequence_throughput = total_sequences / total_time if total_time > 0 else 0

    # Calculate average accept rate
    # average_accept_rate = (total_accept_rate / num_batches_with_accept_rate) if num_batches_with_accept_rate > 0 else None

    return {
        'latency': average_latency,
        'output_token_throughput': output_token_throughput,
        'total_token_throughput': total_token_throughput,
        'sequence_throughput': sequence_throughput,
        # 'accept_rate': average_accept_rate  # Include accept rate in the metrics
    }

def run_experiment(config):
    """Run the benchmark experiment."""
    results = []

    # Extract configuration parameters
    num_requests_list = config['num_requests_list']
    batch_size_list = config['batch_size_list']
    speculative_models = config['speculative_models']
    target_model_name = config['target_model_name']
    num_speculative_tokens_list = config['num_speculative_tokens_list']
    base_prompt = config.get('base_prompt', "The future of AI is")
    gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
    ngram_prompt_lookup_max = config.get('ngram_prompt_lookup_max', 4)
    ngram_prompt_lookup_min = config.get('ngram_prompt_lookup_min', 1)

    # Initialize the tokenizer for the target model
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    total_experiments = len(num_requests_list) * len(batch_size_list) * len(speculative_models) * len(num_speculative_tokens_list)
    experiment_counter = 0

    # Generate prompts once to keep them consistent across runs
    for num_requests in num_requests_list:
        prompts = generate_prompts(num_requests, base_prompt=base_prompt)
        for batch_size in batch_size_list:
            for speculative_model in speculative_models:
                for num_speculative_tokens in num_speculative_tokens_list:
                    llm = None 
                    experiment_counter += 1
                    percentage_complete = (experiment_counter / total_experiments) * 100
                    print(f"\nRunning experiment {experiment_counter}/{total_experiments} ({percentage_complete:.2f}% complete):")
                    print(f"NUM_REQUESTS={num_requests}, BATCH_SIZE={batch_size}, Speculative Model={speculative_model}, Num Speculative Tokens={num_speculative_tokens}")
                    try:
                        if speculative_model is None:
                            # Without speculative decoding
                            if num_speculative_tokens != num_speculative_tokens_list[0]:
                                continue
                            llm = LLM(
                                model=target_model_name,
                                gpu_memory_utilization=gpu_memory_utilization
                            )
                            spec_model_name = "None"
                            num_spec_tokens = 0  # Set to 0 for no speculative decoding
                        elif speculative_model == '[ngram]':
                            # N-gram speculative model
                            llm = LLM(
                                model=target_model_name,
                                speculative_model='[ngram]',
                                num_speculative_tokens=num_speculative_tokens,
                                ngram_prompt_lookup_max=ngram_prompt_lookup_max,
                                ngram_prompt_lookup_min=ngram_prompt_lookup_min,
                                gpu_memory_utilization=gpu_memory_utilization
                            )
                            spec_model_name = '[ngram]'
                            num_spec_tokens = num_speculative_tokens
                        else:
                            # With speculative decoding using a model
                            llm = LLM(
                                model=target_model_name,
                                speculative_model=speculative_model,
                                num_speculative_tokens=num_speculative_tokens,
                                gpu_memory_utilization=gpu_memory_utilization
                            )
                            spec_model_name = speculative_model
                            num_spec_tokens = num_speculative_tokens

                        print(f"Running: NUM_REQUESTS={num_requests}, BATCH_SIZE={batch_size}, Speculative Model={spec_model_name}, Num Speculative Tokens={num_spec_tokens}")
                        metrics = measure_metrics(llm, prompts, batch_size, tokenizer)
                        metrics.update({
                            'num_requests': num_requests,
                            'batch_size': batch_size,
                            'speculative_model': spec_model_name,
                            'num_speculative_tokens': num_spec_tokens,
                            # 'accept_rate': metrics.get('accept_rate')
                        })
                        results.append(metrics)
                        print(f"Completed: NUM_REQUESTS={num_requests}, BATCH_SIZE={batch_size}, Speculative Model={spec_model_name}, Num Speculative Tokens={num_spec_tokens}")
                    except Exception as e:
                        # Log the error and continue with the next iteration
                        print(f"Error encountered with NUM_REQUESTS={num_requests}, BATCH_SIZE={batch_size}, Speculative Model={spec_model_name}, Num Speculative Tokens={num_spec_tokens}")
                        print(f"Error message: {e}")
                        continue  # Proceed to the next iteration
                    finally:
                        if llm is not None:
                            del llm
                        torch.cuda.empty_cache()
    return results
