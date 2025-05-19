TRAIN_FEATURE_DIR='/home/nf1104/work/Summer 25/LTR_Rubric/train/llama3.3-70b/dl23'
# TRAIN_QREL='/home/nf1104/work/data/dl/data/dl2019/2019binary-qrel.txt'
# TRAIN_QREL='/home/nf1104/work/data/dl/data/dl2020/2020binary-qrel.txt'
TRAIN_QREL='/home/nf1104/work/data/dl/data/dl2023/converted/2023binary_qrels.txt'
FRIENDLY_NAME='llama3.3-70b-dl23'
OUT_DIR='ranklips-results/llama3.3-70bb/dl23'

OUT_PREFIX_CV="cv-5fold"
EXPERIMENT_NAME="5-fold rank-lips experiment"


FEAT_PARAM="--feature-variant FeatScore"
OPT_PARAM="--z-score --default-any-feature-value 0.0 --convergence-threshold 0.0001 --mini-batch-size 100  --folds 5 --restarts 20 --save-heldout-queries-in-model"

echo " ---- 5-FOLD CROSS-VALIDATION ----- "

./rank-lips train --train-cv \
  -d "${TRAIN_FEATURE_DIR}" \
  -q "${TRAIN_QREL}" \
  --trec-eval-run \
  -e "${EXPERIMENT_NAME}" \
  -O "${OUT_DIR}" \
  -o "${OUT_PREFIX_CV}" \
  ${OPT_PARAM} ${FEAT_PARAM}

echo "Predicted Ranking:"
cat "${OUT_DIR}/${OUT_PREFIX_CV}-run-test.run"

echo "Train/Test MAP scores:" | tee "${OUT_DIR}/MAP_scores.txt"
./rank-lips train --train-cv \
  -d "${TRAIN_FEATURE_DIR}" \
  -q "${TRAIN_QREL}" \
  -e "${EXPERIMENT_NAME}" \
  -O "${OUT_DIR}" \
  --trec-eval-run \
  -o "${OUT_PREFIX_CV}" \
  ${OPT_PARAM} ${FEAT_PARAM} |& grep -e "Model test test metric" -e "Model train train metric" | tee -a "${OUT_DIR}/MAP_scores.txt"

# Optional: Evaluate using trec_eval if installed
# trec_eval -c -m map ${TRAIN_QREL} ${OUT_DIR}/${OUT_PREFIX_CV}-run-test.run
