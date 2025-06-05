import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_before_after(before_file: str, after_files: list, model_names: list, output_path: str, dataset: str):
    def load_ndcg_data(filepath, suffix_to_remove=""):
        ndcg_scores = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    run_name = parts[0].replace(suffix_to_remove, "")
                    try:
                        score = float(parts[1])
                        ndcg_scores[run_name] = score
                    except ValueError:
                        continue
        return ndcg_scores

    # Load before scores (common for all models)
    before = load_ndcg_data(before_file, ".run")
    
    # Load after scores for each model
    after_scores = {}
    for model_name, after_file in zip(model_names, after_files):
        after_scores[model_name] = load_ndcg_data(after_file, "/cv-5fold-run-test.run")
    
    # Find common runs across before and all after files, sort by before NDCG
    common_runs = sorted(
        set(before.keys()) & set.intersection(*(set(scores) for scores in after_scores.values())),
        key=lambda x: before[x], reverse=False
    )
    
    # Prepare data for the plot
    before_vals = [before[k] for k in common_runs]
    after_vals = {model: [scores[k] for k in common_runs] for model, scores in after_scores.items()}
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bar_width = 0.15  
    bar_spacing = 0.05  
    group_spacing = 0.5  
    x = np.arange(len(common_runs)) * (bar_width * (len(model_names) + 1) + group_spacing)
    
    # Plot 'Before' bars
    plt.bar(x, before_vals, width=bar_width, label='Before', color='tab:blue')
    
    # Plot 'After' bars for each model with consistent width and spacing
    colors = ['tab:red', 'tab:green', 'tab:orange', 'tab: SXpurple', 'tab:brown', 'tab:pink']
    for idx, model in enumerate(model_names):
        offset = (idx + 1) * (bar_width + bar_spacing)
        plt.bar(x + offset, after_vals[model], width=bar_width, label=f'After ({model})', color=colors[idx % len(colors)])
    
    # Customize plot
    plt.xlabel("Run")
    plt.ylabel("NDCG@20")
    plt.title(f"NDCG@20 Before and After Reranking Across Models -- {dataset}")
    plt.xticks(x + bar_width * len(model_names) / 2, common_runs, rotation=90)
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting before and after summaries of NDCG scores in a bar graph for multiple models")
    parser.add_argument("--before", required=True, type=str, help="Path to NDCG scores before reranking")
    parser.add_argument("--after", required=True, type=str, nargs='+', help="Paths to NDCG scores after reranking for each model")
    parser.add_argument("--models", required=True, type=str, nargs='+', help="Names of the models corresponding to after files")
    parser.add_argument("--dataset", default="", required=False, type=str, help="the dataset to put in the plot title")
    parser.add_argument("--output", required=True, type=str, help="Path to save the bar graph figure")
    args = parser.parse_args()
    
    # Ensure the number of after files matches the number of model names
    if len(args.after) != len(args.models):
        raise ValueError("The number of after files must match the number of model names")
    
    plot_before_after(before_file=args.before, after_files=args.after, model_names=args.models, output_path=args.output, dataset=args.dataset)