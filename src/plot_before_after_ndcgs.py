import matplotlib.pyplot as plt
import argparse


def plot_before_after(before_file: str, after_file: str, output_path: str):
# before_file = "before_summary.txt"
# after_file = "after_summary.txt"

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

    before = load_ndcg_data(before_file, ".run")
    after = load_ndcg_data(after_file, "/cv-5fold-run-test.run")

    # Keep only runs present in both
    # Keep only runs present in both, and sort by "before" NDCG
    common_runs = sorted(set(before) & set(after), key=lambda x: before[x], reverse=False)

    before_vals = [before[k] for k in common_runs]
    after_vals = [after[k] for k in common_runs]


    #####

    plt.figure(figsize=(8, 6))
    bar_width = 0.1
    x = range(len(common_runs))

    plt.bar(x, before_vals, width=bar_width, label='Before', color='tab:blue')
    plt.bar([i + bar_width+0.05 for i in x], after_vals, width=bar_width, label='After', color='tab:red')

    plt.xlabel("Run")
    plt.ylabel("NDCG@20")
    plt.title("NDCG@20 Before and After Reranking")
    plt.xticks([i + bar_width / 2 for i in x], common_runs, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(output_path)




if __name__=="__main__":
    parser = argparse.ArgumentParser("plotting before and after summmaries of ndcgs of the run files")
    parser.add_argument("--before", required=True, type=str, help="ndcgs before the reranked")
    parser.add_argument("--after", required=True, type=str, help="ndcgs after the reranked")
    parser.add_argument("--output", required=True, type=str, help="path to save the figure")
    args = parser.parse_args()
    plot_before_after(before_file=args.before, after_file=args.after, output_path=args.output)