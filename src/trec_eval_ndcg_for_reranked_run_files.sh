#!/bin/bash

QRELS_PATH="/home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt"
BASE_DIR="/home/nf1104/work/Summer 25/LTR_Rubric/ranklips-results/llama3.3-70b/dl19"
ORIG_RUNS_DIR="/home/nf1104/work/trec-dl-2019/runs"

SUMMARY_BEFORE="ndcg_summary_before.txt"
SUMMARY_AFTER="ndcg_summary_after.txt"
LOG_FILE="ndcg_evaluation.log"

# Initialize output files
> "$SUMMARY_BEFORE"
> "$SUMMARY_AFTER"
> "$LOG_FILE"

# Function to clean run files and log malformed lines
clean_run_file() {
    local input_file="$1"
    local output_file="$2"
    local log_file="$3"

    # Check if input file exists and is readable
    if [ ! -f "$input_file" ] || [ ! -r "$input_file" ]; then
        echo "Error: Input file $input_file does not exist or is not readable" | tee -a "$log_file"
        return 1
    fi

    # Check if input file is empty
    if [ ! -s "$input_file" ]; then
        echo "Error: Input file $input_file is empty" | tee -a "$log_file"
        return 1
    fi

    # Log first few lines of input file for debugging
    echo "First 5 lines of $input_file:" >> "$log_file"
    head -n 5 "$input_file" >> "$log_file"
    echo "------------------------" >> "$log_file"

    # Clean the file
    awk -v log_file="$log_file" -v input_file="$input_file" '
    BEGIN { line_num = 0 }
    {
        line_num++;
        # Skip empty lines
        if ($0 ~ /^[ \t]*$/) {
            print "Skipping empty line " line_num " in " input_file >> log_file;
            next;
        }
        # Trim whitespace
        gsub(/^[ \t]+|[ \t]+$/, "", $0);
        nf = split($0, fields, /[ \t]+/);

        if (nf >= 6) {
            # Well-formed TREC line
            print $0;
        } else if (nf >= 3) {
            # Partially malformed, attempt to fix
            qid = fields[1];
            docid = fields[3];
            rank = (nf >= 4 && fields[4] ~ /^[0-9]+$/) ? fields[4] : "1000";
            score = (nf >= 5 && fields[5] ~ /^[0-9]+(\.[0-9]+)?$/) ? fields[5] : "0.0";
            tag = (nf >= 6) ? fields[6] : "AUTO";
            print qid " Q0 " docid " " rank " " score " " tag;
        } else {
            # Log malformed line
            print "Malformed line " line_num " in " input_file ": " $0 >> log_file;
        }
    }' "$input_file" > "$output_file"

    # Check if output file is empty
    if [ ! -s "$output_file" ]; then
        echo "Error: Cleaned output file $output_file is empty" | tee -a "$log_file"
        return 1
    fi
}

# Function to evaluate nDCG@20
evaluate_ndcg() {
    local run_file="$1"
    local qrels_file="$2"
    local output_file="$3"
    local summary_file="$4"
    local log_file="$5"
    local file_label="$6"
    local clean_needed="$7"

    echo "Evaluating: $run_file" | tee -a "$log_file"
    local eval_file="$run_file"

    # Clean the run file if needed
    if [ "$clean_needed" = "true" ]; then
        local cleaned_run
        cleaned_run=$(mktemp)
        clean_run_file "$run_file" "$cleaned_run" "$log_file" || {
            echo "Error: Failed to clean $run_file" | tee -a "$log_file"
            rm -f "$cleaned_run"
            return 1
        }
        eval_file="$cleaned_run"
    fi

    # Check if eval_file is empty
    if [ ! -s "$eval_file" ]; then
        echo "Error: Run file $run_file (or cleaned version) is empty" | tee -a "$log_file"
        [ "$clean_needed" = "true" ] && rm -f "$cleaned_run"
        return 1
    fi

    # Run trec_eval
    local ndcg
    ndcg=$(trec_eval -m ndcg_cut.20 "$qrels_file" "$eval_file" 2>>"$log_file" | grep "ndcg_cut_20" | awk '{print $3}')

    if [ -z "$ndcg" ]; then
        echo "Error: Failed to compute nDCG@20 for $run_file" | tee -a "$log_file"
    else
        echo "$file_label $ndcg" >> "$summary_file"
        # Write detailed trec_eval output to output_file
        trec_eval -m ndcg_cut.20 "$qrels_file" "$eval_file" > "$output_file" 2>>"$log_file"
    fi

    [ "$clean_needed" = "true" ] && rm -f "$cleaned_run"
}

echo "=== Evaluating BEFORE reranking ===" | tee -a "$LOG_FILE"
find "$ORIG_RUNS_DIR" -type f -name "*.run" | while read -r run_file; do
    file_label=$(basename "$run_file")
    evaluate_ndcg "$run_file" "$QRELS_PATH" "${run_file%.run}_ndcg_before.txt" "$SUMMARY_BEFORE" "$LOG_FILE" "$file_label" "true"
done

echo "=== Evaluating AFTER reranking ===" | tee -a "$LOG_FILE"
find "$BASE_DIR" -type f -name "cv-5fold-run-test.run" | while read -r run_file; do
    file_label="${run_file#"$BASE_DIR"/}"
    evaluate_ndcg "$run_file" "$QRELS_PATH" "$(dirname "$run_file")/ndcg_scores.txt" "$SUMMARY_AFTER" "$LOG_FILE" "$file_label" "false"
done