#!/bin/bash

BASE_FILTERED_DIR="/home/nf1104/work/Summer 25/LTR_Rubric/train/flant5/dl19/filtered_dl19"
TRAIN_QREL='/home/nf1104/work/data/dl/data/dl2019/2019binary-qrel.txt'

# Shared settings
OUT_BASE_DIR='ranklips-results/flant5/dl19'
OUT_PREFIX_CV="cv-5fold"
EXPERIMENT_NAME="5-fold rank-lips experiment"
FEAT_PARAM="--feature-variant FeatScore"
OPT_PARAM="--z-score --default-any-feature-value 0.0 --convergence-threshold 0.0001 --mini-batch-size 100  --folds 5 --restarts 10 --save-heldout-queries-in-model"

for SYSTEM_DIR in "${BASE_FILTERED_DIR}"/*/; do
  SYSTEM_NAME=$(basename "$SYSTEM_DIR")
  TRAIN_FEATURE_DIR="$SYSTEM_DIR"
  OUT_DIR="${OUT_BASE_DIR}/${SYSTEM_NAME}"

  echo " ==== Running rank-lips for system: ${SYSTEM_NAME} ===="

  mkdir -p "${OUT_DIR}"

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
done
