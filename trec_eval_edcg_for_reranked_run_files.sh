#!/bin/bash

QRELS_PATH="/home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt"
BASE_DIR="/home/nf1104/work/Summer 25/LTR_Rubric/ranklips-results/flant5/dl19"
ORIG_RUNS_DIR="/home/nf1104/work/data/dl/data/dl2019/runs_dl2019"

SUMMARY_BEFORE="ndcg_summary_before.txt"
SUMMARY_AFTER="ndcg_summary_after.txt"
> "$SUMMARY_BEFORE"
> "$SUMMARY_AFTER"

echo "=== Evaluating BEFORE reranking ==="
for RUN_FILE in "$ORIG_RUNS_DIR"/*.run; do
    OUT_FILE="${RUN_FILE%.run}_ndcg_before.txt"
    echo "Evaluating: $RUN_FILE"
    CLEANED_RUN=$(mktemp)
    awk 'NF == 6' "$RUN_FILE" > "$CLEANED_RUN"

    NDCG=$(trec_eval -m ndcg_cut.20 "$QRELS_PATH" "$CLEANED_RUN" 2>/dev/null | grep "ndcg_cut_20" | awk '{print $3}')
    echo "$(basename "$RUN_FILE") $NDCG" >> "$SUMMARY_BEFORE"

    rm "$CLEANED_RUN"

    # NDCG=$(grep "ndcg_cut_20" "$OUT_FILE" | awk '{print $3}')
    # echo "$(basename "$RUN_FILE") $NDCG" >> "$SUMMARY_BEFORE"
done

echo "=== Evaluating AFTER reranking ==="
find "$BASE_DIR" -type f -name "cv-5fold-run-test.run" | while read -r RUN_FILE
do
    DIR=$(dirname "$RUN_FILE")
    OUT_FILE="$DIR/ndcg_scores.txt"
    echo "Evaluating: $RUN_FILE"
    trec_eval -m ndcg_cut.20 "$QRELS_PATH" "$RUN_FILE" > "$OUT_FILE"

    NDCG=$(grep "ndcg_cut_20" "$OUT_FILE" | awk '{print $3}')
    echo "${RUN_FILE#"$BASE_DIR"/} $NDCG" >> "$SUMMARY_AFTER"
done
