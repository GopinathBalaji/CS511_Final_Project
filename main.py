# main.py

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from experiment import run_experiment

def main():
    # Load configuration from config.yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Convert 'null' strings to None in speculative_models
    speculative_models = config['speculative_models']
    config['speculative_models'] = [None if model is None or str(model).lower() == 'null' else model for model in speculative_models]

    # Run the experiment
    results = run_experiment(config)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    columns_order = [
        'speculative_model',
        'num_speculative_tokens',
        'num_requests',
        'batch_size',
        'latency',
        'output_token_throughput',
        'total_token_throughput',
        'sequence_throughput'
    ]
    df = df[columns_order]

    # Display the results
    print("\nExperiment Results:")
    print(df)

    # Save results to CSV
    output_csv = "raw_data/" + config.get('output_csv', 'vllm_benchmark_results.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Optional: Plot the results
    # plot_results(df, config)

# def plot_results(df, config):
#     num_requests_list = config['num_requests_list']
#     batch_size_list = config['batch_size_list']
#     num_speculative_tokens_list = config['num_speculative_tokens_list']

#     for num_requests in num_requests_list:
#         for batch_size in batch_size_list:
#             for num_speculative_tokens in num_speculative_tokens_list:
#                 df_subset = df[
#                     (df['num_requests'] == num_requests) &
#                     (df['batch_size'] == batch_size) &
#                     (df['num_speculative_tokens'] == num_speculative_tokens)
#                 ]
#                 plt.figure(figsize=(10, 6))
#                 plt.bar(df_subset['speculative_model'], df_subset['output_token_throughput'])
#                 plt.xlabel('Speculative Model')
#                 plt.ylabel('Output Token Throughput')
#                 plt.title(f'Output Token Throughput for NUM_REQUESTS={num_requests}, BATCH_SIZE={batch_size}, Num Speculative Tokens={num_speculative_tokens}')
#                 plt.xticks(rotation=45)
#                 plt.tight_layout()
#                 plt.savefig(f'output_token_throughput_NUM_REQUESTS={num_requests}_BATCH_SIZE={batch_size}_NumSpecTokens={num_speculative_tokens}.png')

if __name__ == '__main__':
    main()
