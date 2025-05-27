import os
import argparse
from collections import defaultdict

def read_run(file_path):
    run = defaultdict(dict)
    with open(file_path) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            run[qid][docid] = float(score)
    return run

def get_all_qid_doc_pairs(run):
    return {(qid, docid) for qid in run for docid in run[qid]}

def write_filtered_run(run, allowed_qid_doc_pairs, output_path):
    with open(output_path, 'w') as out:
        for qid in sorted(run.keys()):
            for docid in sorted(run[qid].keys()):
                if (qid, docid) in allowed_qid_doc_pairs:
                    score = run[qid][docid]
                    out.write(f"{qid} Q0 {docid} 0 {score} filtered\n")  # rank=0, will be ignored by ranklips

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True, help="Path to the base system .run file")
    parser.add_argument("--feature-runs", nargs='+', required=True, help="List of feature .run files to filter")
    parser.add_argument("--output-dir", required=True, help="Directory to write filtered feature runs")

    args = parser.parse_args()

    base_run = read_run(args.base_run)
    allowed_pairs = get_all_qid_doc_pairs(base_run)

    os.makedirs(args.output_dir, exist_ok=True)
    for feat_path in args.feature_runs:
        run = read_run(feat_path)
        out_path = os.path.join(args.output_dir, os.path.basename(feat_path))
        write_filtered_run(run, allowed_pairs, out_path)
        print(f"Wrote filtered feature file to {out_path}")
