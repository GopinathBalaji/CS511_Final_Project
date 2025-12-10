# This script reads the benchmark results from raw_data/ and generates visualizations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_data(data_file):
    if data_file.startswith("raw_data/"):
        data_file = data_file.split("raw_data/")[1]
    data = pd.read_csv(f"raw_data/{data_file}")
    data = data.fillna("None")
    gpu = data_file.split(".")[0].split("vllm_benchmark_results_")[1]
    target_dir = f"visualizations/{gpu}"
    gpu = gpu.split("_")[0].upper()
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # Create line plot for token throughput
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data,
        x="num_speculative_tokens",
        y="total_token_throughput",
        hue="speculative_model",
    )
    plt.title(f"Token Throughput vs Number of Speculative Tokens ({gpu})")
    plt.xlabel("Number of Speculative Tokens")
    plt.ylabel("Total Token Throughput")
    plt.savefig(
        f"{target_dir}/token_throughput_vs_num_tokens.png",
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data,
        x="speculative_model",
        y="total_token_throughput",
        hue="num_speculative_tokens",
    )
    plt.title(f"Total Token Throughput by Speculative Model ({gpu})")
    plt.xlabel("Speculative Model")
    plt.ylabel("Total Token Throughput")
    plt.savefig(
        f"{target_dir}/total_token_throughput_by_speculative_model.png",
    )

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=data, x="speculative_model", y="total_token_throughput", split=True
    )
    plt.title(f"Total Token Throughput Distribution by Speculative Model ({gpu})")
    plt.xlabel("Speculative Model")
    plt.ylabel("Total Token Throughput")
    plt.savefig(
        f"{target_dir}/total_token_throughput_distribution_by_speculative_model.png",
    )

    # Pivot data for heatmap
    plt.figure(figsize=(10, 6))
    pivot_data = data.pivot_table(
        values="total_token_throughput",
        index="speculative_model",
        columns="num_speculative_tokens",
    )
    sns.heatmap(pivot_data, annot=True, cmap="YlOrRd")
    plt.title(
        f"Total Token Throughput by Speculative Model and Number of Speculative Tokens ({gpu})"
    )
    plt.xlabel("Number of Speculative Tokens")
    plt.ylabel("Speculative Model")
    plt.savefig(
        f"{target_dir}/total_token_throughput_heatmap.png",
    )
    
    
    # Throughput vs num requests
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data,
        x="num_requests",
        y="total_token_throughput",
        hue="speculative_model",
    )
    plt.title(f"Token Throughput vs Number of Requests ({gpu})")
    plt.xlabel("Number of Requests")
    plt.ylabel("Total Token Throughput")
    plt.savefig(
        f"{target_dir}/token_throughput_vs_num_requests.png",
    )
    
    # Throughput vs batch size
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data[data["batch_size"] < 10],
        x="batch_size",
        y="total_token_throughput",
        hue="speculative_model",
    )
    plt.title(f"Token Throughput vs Batch Size ({gpu})")
    plt.xlabel("Batch Size")
    plt.ylabel("Total Token Throughput")
    plt.savefig(
        f"{target_dir}/token_throughput_vs_batch_size.png",
    )
    
    
    # Speculative tokens vs latency
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data,
        x="num_speculative_tokens",
        y="latency",
        hue="speculative_model",
    )
    plt.title(f"Total Latency vs Number of Speculative Tokens ({gpu})")
    plt.xlabel("Number of Speculative Tokens")
    plt.ylabel("Latency")
    plt.savefig(
        f"{target_dir}/total_latency_vs_num_tokens.png",
    )
    
    # Model vs latency
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data,
        x="speculative_model",
        y="latency",
        hue="num_speculative_tokens",
    )
    plt.title(f"Total Latency vs Speculative Model ({gpu})")
    plt.xlabel("Speculative Model")
    plt.ylabel("Latency")
    plt.savefig(
        f"{target_dir}/total_latency_vs_speculative_model.png",
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_data.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    analyze_data(data_file)


if __name__ == "__main__":
    main()
