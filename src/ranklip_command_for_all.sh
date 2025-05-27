#!/bin/bash

BASE_FILTERED_DIR="/home/nf1104/work/Summer 25/LTR_Rubric/train/llama3.3-70b/dl19/filtered_dl19"
TRAIN_QREL='/home/nf1104/work/data/dl/data/dl2019/2019binary-qrel.txt'
ORG_RUN_FILES='/home/nf1104/work/trec-dl-2019/runs'
OUT_BASE_DIR='ranklips-results/llama3.3-70b/dl19'




# Shared settings
OUT_PREFIX_CV="cv-5fold"
EXPERIMENT_NAME="5-fold rank-lips experiment"
FEAT_PARAM="--feature-variant FeatScore"
# OPT_PARAM="--z-score --default-any-feature-value 0.0 --convergence-threshold 0.0001 --mini-batch-size 100  --folds 5 --restarts 10 --save-heldout-queries-in-model"
OPT_PARAM="--z-score --default-any-feature-value -99 --convergence-threshold 0.0001 --mini-batch-size 100  --folds 5 --restarts 10 --save-heldout-queries-in-model"



for SYSTEM_DIR in "${BASE_FILTERED_DIR}"/*/; do
  SYSTEM_NAME=$(basename "$SYSTEM_DIR")
  TRAIN_FEATURE_DIR="$SYSTEM_DIR"
  OUT_DIR="${OUT_BASE_DIR}/${SYSTEM_NAME}"

  echo " ==== Running rank-lips for system: ${SYSTEM_NAME} ===="

  mkdir -p "${OUT_DIR}"

  #adding each org system run file as a feature temporarily
  RUN_FILE="${ORG_RUN_FILES}/${SYSTEM_NAME}.run"
  TMP_RUN_NAME="OrigScore.run"
  cp "$RUN_FILE" "${TRAIN_FEATURE_DIR}/${TMP_RUN_NAME}"
  #####

  ./rank-lips train --train-cv \
    -d "${TRAIN_FEATURE_DIR}" \
    -q "${TRAIN_QREL}" \
    --trec-eval-run \
    -e "${EXPERIMENT_NAME}" \
    -O "${OUT_DIR}" \
    -o "${OUT_PREFIX_CV}" \
    ${OPT_PARAM} ${FEAT_PARAM}


  echo "Predicted Ranking for ${SYSTEM_NAME}:"
  cat "${OUT_DIR}/${OUT_PREFIX_CV}-run-test.run"

  echo "Train/Test MAP scores for ${SYSTEM_NAME}:" | tee "${OUT_DIR}/MAP_scores.txt"
  ./rank-lips train --train-cv \
    -d "${TRAIN_FEATURE_DIR}" \
    -q "${TRAIN_QREL}" \
    -e "${EXPERIMENT_NAME}" \
    -O "${OUT_DIR}" \
    --trec-eval-run \
    -o "${OUT_PREFIX_CV}" \
    ${OPT_PARAM} ${FEAT_PARAM} |& grep -e "Model test test metric" -e "Model train train metric" | tee -a "${OUT_DIR}/MAP_scores.txt"

    ## remove that temp feature
    rm "${TRAIN_FEATURE_DIR}/${TMP_RUN_NAME}"
    ###
done
