import os
import subprocess
import tempfile
from pathlib import Path

# ===== Configuration ===== #
QRELS_PATH = "/home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt"
BASE_DIR = "/home/nf1104/work/Summer 25/LTR_Rubric/ranklips-results/flant5/dl19"
ORIG_RUNS_DIR = "/home/nf1104/work/trec-dl-2019/runs"

MAX_QUERIES = None  # Set to None to process all queries
MAX_DOCS_PER_QUERY = None  # Set to None to process all docs per query

SUMMARY_BEFORE = "ndcg_summary_before_flant5.txt"
SUMMARY_AFTER = "ndcg_summary_after_flant5.txt"
LOG_FILE = "ndcg_evaluation.log"

# ===== Utilities ===== #
def log_message(message, log_path=LOG_FILE):
    print(message)
    with open(log_path, 'a') as logf:
        logf.write(f"{message}\n")

def clear_files(file_paths):
    for path in file_paths:
        open(path, 'w').close()

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# ===== Run File Cleaning ===== #
def clean_run(input_path, output_path, log_path, max_docs_per_query=None):
    if not os.path.isfile(input_path) or not os.access(input_path, os.R_OK):
        log_message(f"Error: Cannot read {input_path}", log_path)
        return False

    if os.path.getsize(input_path) == 0:
        log_message(f"Error: File {input_path} is empty", log_path)
        return False

    with open(log_path, 'a') as lf, open(input_path, 'r') as infile:
        lf.write(f"First 5 lines of {input_path}:\n")
        for i, line in enumerate(infile):
            if i >= 5: break
            lf.write(line)
        lf.write("-" * 24 + "\n")

    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            doc_count = 0
            prev_qid = None

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    log_message(f"Skipping empty line {line_num} in {input_path}", log_path)
                    continue

                fields = line.split()
                if len(fields) >= 6:
                    qid = fields[0]
                    if qid != prev_qid:
                        doc_count = 0
                    # Only apply doc limit if max_docs_per_query is not None
                    if max_docs_per_query is None or doc_count < max_docs_per_query:
                        outfile.write(line + '\n')
                        doc_count += 1
                    prev_qid = qid
                elif len(fields) >= 3:
                    qid = fields[0]
                    docid = fields[2]
                    rank = fields[3] if len(fields) > 3 and fields[3].isdigit() else "1000"
                    score = fields[4] if len(fields) > 4 and is_number(fields[4]) else "0.0"
                    tag = fields[5] if len(fields) > 5 else "AUTO"
                    reconstructed = f"{qid} Q0 {docid} {rank} {score} {tag}"
                    if qid != prev_qid:
                        doc_count = 0
                    # Only apply doc limit if max_docs_per_query is not None
                    if max_docs_per_query is None or doc_count < max_docs_per_query:
                        outfile.write(reconstructed + '\n')
                        doc_count += 1
                    prev_qid = qid
                else:
                    log_message(f"Malformed line {line_num} in {input_path}: {line}", log_path)
    except Exception as e:
        log_message(f"Error processing {input_path}: {e}", log_path)
        return False

    if os.path.getsize(output_path) == 0:
        log_message(f"Error: Output file {output_path} is empty", log_path)
        return False

    return True

# ===== Evaluation ===== #
def limit_queries(input_path, max_queries=None):
    # If max_queries is None, return the original file path
    if max_queries is None:
        return input_path
    
    output = tempfile.NamedTemporaryFile(mode='w', delete=False)
    seen_qids = set()
    count = 0

    with open(input_path, 'r') as infile, open(output.name, 'w') as outfile:
        for line in infile:
            fields = line.strip().split()
            if not fields:
                continue
            qid = fields[0]
            if qid not in seen_qids:
                count += 1
                seen_qids.add(qid)
                if count > max_queries:
                    continue
            outfile.write(line)

    return output.name

def extract_ndcg_from_output(output_str):
    for line in output_str.splitlines():
        if 'ndcg_cut_20' in line:
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return None

def run_trec_eval(qrels_path, run_path, log_path):
    result = subprocess.run(['trec_eval', '-m', 'ndcg_cut.20', qrels_path, run_path],
                            capture_output=True, text=True)
    with open(log_path, 'a') as logf:
        if result.stderr:
            logf.write(result.stderr)
    return result

def evaluate_run(run_file, summary_file, file_label, clean=True, qrels_path=QRELS_PATH, log_path=LOG_FILE, output_file=None, max_queries=None, max_docs_per_query=None):
    log_message(f"Evaluating: {run_file}", log_path)
    cleaned_path = run_file

    if clean:
        temp_clean = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_clean.close()
        if not clean_run(run_file, temp_clean.name, log_path, max_docs_per_query):
            os.unlink(temp_clean.name)
            return False
        cleaned_path = temp_clean.name

    if os.path.getsize(cleaned_path) == 0:
        log_message(f"Error: {cleaned_path} is empty", log_path)
        if clean:
            os.unlink(cleaned_path)
        return False

    limited_path = limit_queries(cleaned_path, max_queries)

    result = run_trec_eval(qrels_path, limited_path, log_path)

    if result.returncode != 0:
        log_message(f"Error: trec_eval failed for {run_file}", log_path)
        return False

    ndcg = extract_ndcg_from_output(result.stdout)
    if ndcg is None:
        log_message(f"Error: Could not parse nDCG from trec_eval output for {run_file}", log_path)
        return False

    with open(summary_file, 'a') as sf:
        sf.write(f"{file_label} {ndcg}\n")

    if output_file:
        with open(output_file, 'w') as outf:
            outf.write(result.stdout)

    # Clean up temporary files
    # Only unlink limited_path if it's different from cleaned_path (i.e., when max_queries was not None)
    if limited_path != cleaned_path:
        os.unlink(limited_path)
    if clean:
        os.unlink(cleaned_path)

    return True

# ===== Main ===== #
def evaluate_runs_in_directory(directory, summary_file, clean_runs, file_pattern="*.run", output_name=None, max_queries=None, max_docs_per_query=None):
    path = Path(directory)
    if not path.exists():
        log_message(f"Directory not found: {directory}")
        return
    for run_file in path.rglob(file_pattern):
        label = str(run_file.relative_to(directory)) if clean_runs is False else run_file.name
        output_path = run_file.parent / output_name if output_name else None
        evaluate_run(str(run_file), summary_file, label, clean=clean_runs, 
                    output_file=str(output_path) if output_path else None,
                    max_queries=max_queries, max_docs_per_query=max_docs_per_query)

def main():
    clear_files([SUMMARY_BEFORE, SUMMARY_AFTER, LOG_FILE])
    log_message("=== Evaluating BEFORE reranking ===")
    evaluate_runs_in_directory(ORIG_RUNS_DIR, SUMMARY_BEFORE, clean_runs=True, 
                              max_queries=MAX_QUERIES, max_docs_per_query=MAX_DOCS_PER_QUERY)

    log_message("=== Evaluating AFTER reranking ===")
    evaluate_runs_in_directory(BASE_DIR, SUMMARY_AFTER, clean_runs=False, 
                              file_pattern="cv-5fold-run-test.run", output_name="ndcg_scores.txt",
                              max_queries=MAX_QUERIES, max_docs_per_query=MAX_DOCS_PER_QUERY)

if __name__ == "__main__":
    main()