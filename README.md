


## Workflow for Reranking and Evaluation
This workflow outlines how to rerank documents using LLM-derived criteria judgments and evaluate the effectiveness of reranking using TREC-style metrics. It assumes that you already have LLM-generated .jsonl.gz judgment files and baseline run files.

### Step 1: Build Feature Vectors

Use build_feature_vectors.py to generate:

1. RankLib-compatible feature vectors (.ranklib file)

2. Criterion-specific run files for reranking

These are extracted from .jsonl.gz files containing LLM-generated judgments.



#### 🔧 Example (`build_feature.sh`)

```bash
input='/home/nf1104/work/data/rubric_format_inputs/flant5large/llmjudge_test_4prompts_qrel_flant5large.jsonl.gz'
output_dir='./feature_vectors'
qrel='/home/nf1104/work/data/dl/data/dl2023/converted/new_qrels.txt'
o_path='train/flant5/dl23'
f_v_output='./feature_vectors/flant5_dl23.ranklib'

# Repeat this for each criterion
for criterion in Exactness Coverage Topicality "Contextual Fit"; do
  python3 build_feature_vectors.py \
    --judgements "$input" \
    --qrel "$qrel" \
    --output "$f_v_output" \
    --criteria-run-dir "$output_dir" \
    --mode multi_criteria \
    --criteria-run-dir "$o_path" \
    --criterion "$criterion" \
    --no-one-hot
done
```
### Step 2: Filter Feature Run Files
Use ```batch_filter.py``` to filter the criterion-specific run files so that only query-document pairs that exist in the base system runs are retained.


Input:
Base runs: ```/home/nf1104/work/data/runs/runs_trecdl2019/*.run```
Feature runs: ```/home/nf1104/work/Summer 25/LTR_Rubric/train/llama3.3-70b/dl19/*.run``` (generated in step 1)

Output:

Filtered run files are saved as:
```train/llama3.3-70b/dl19/filtered_dl19/<system_name>/```

### Step 3: Rerank with Rank-LiPS
Use ranklip-command_for_all.sh to perform 5-fold cross-validation with the Rank-LiPS reranker for each system.
```bash ranklip-command_for_all.sh```
Input:

Filtered features: ```train/llama3.3-70b/dl19/filtered_dl19/<system>/```

QREL: ```/home/nf1104/work/data/dl/data/dl2019/2019binary-qrel.txt```

Output:

Reranked test runs and metrics in: ```ranklips-results/llama3.3-70b/dl19/<system>/```

Each system folder includes:

```cv-5fold-run-test.run```: Final reranked results

```MAP_scores.txt```: MAP scores on train/test sets



### Step 4: Evaluate with TREC Metrics
Use ndcg_eval_script.py to compute standard IR evaluation metrics (e.g., NDCG@20) on the reranked outputs.

This script performs the following:

- Cleans malformed or incomplete .run files

- Computes metrics with trec_eval

- Logs performance before and after reranking

#### Output:
— A .txt file of NDCG@20 scores before reranking

- A .txt file of NDCG@20 scores before reranking

- Per-run evaluation results in: ndcg_scores.txt of each runfile-specific folders in ```/ranklips-results```

---
## More About: Feature Vector Builder

This script (build_feature_vectors.py) processes a JSONL.gz file containing query-document pairs with self-ratings and generates feature vectors in RankLib format for learning-to-rank tasks. It supports flexible feature extraction modes and includes detailed debugging logs to trace ratings and feature computations.
Purpose
Extracts features from self-ratings in a JSONL.gz file (e.g., from TREC Deep Learning 2019 data) and saves them in RankLib format (<label> qid:<query_id> 1:<feature_1> ... # <doc_id>). Features are derived from self-ratings for prompt classes like nuggets, questions, and direct (e.g., FagB, HELM).
Features

# Modes:
nuggets: Features from NuggetSelfRatedPrompt.
questions: Features from QuestionSelfRatedUnanswerablePromptWithChoices.
one_hot: All prompt classes (including HELM, SUN, Faggioli, etc.), with optional one-hot encodings.
Default (no mode): All prompt classes.


One-Hot Control: Use --no-one-hot to skip one-hot encodings in one_hot or default modes.
Debugging: Logs raw self-ratings, processed ratings, feature values, and descriptions for each query-document pair.
Input: JSONL.gz file with QueryWithFullParagraphList objects, qrel file with relevance labels.
Output: RankLib-formatted feature file.

# Requirements

Custom module: exam_pp.data_model (provides QueryWithFullParagraphList, GradeFilter, parseQueryWithFullParagraphs)

Installation

Clone the repository or copy build_feature_vectors.py.
To include exam_pp.data_model and other dependecncies, use  `nix develop`.




# Usage
Run the script with command-line arguments:
python3 build_feature_vectors.py \
  --judgements <path_to_jsonl.gz> \
  --qrel <path_to_qrel.txt> \
  --output <output_ranklib_file> \
  [--mode {nuggets,questions,one_hot}] \
  [--no-one-hot] \
  [--max-query <int>] \
  [--max-passage <int>]

Example
python3 build_feature_vectors.py \
  --judgements /home/nf1104/work/data/pencils-down/dl19/Thomas-Sun_few-Sun-HELM-FagB_few-FagB-questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2019-qrels-runs-with-text.jsonl.gz \
  --qrel /home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt \
  --output /feature_vectors/featuresdl19.ranklib \
  --mode questions \
  --no-one-hot \
  --max-query 1 \
  --max-passage 1

Arguments

--judgements: Path to JSONL.gz file with query-document data and self-ratings.
--qrel: Path to qrel file (format: qid 0 did rel).
--output: Path to output RankLib feature file.
--mode: Feature extraction mode (nuggets, questions, one_hot, or empty for default).
--no-one-hot: Disable one-hot encodings (affects one_hot or default modes).
--max-query: Limit the number of queries processed (optional, for debugging).
--max-passage: Limit the number of passages per query (optional, for debugging).

# Output

RankLib File: Features in the format <label> qid:<query_id> 1:<feature_1> ... # <doc_id>.
For --mode questions --no-one-hot: 25 features (10 integer ratings sorted by mean rating, 10 by rating value, 5 counts).


Debug Logs: Written to console, including:
Raw and processed self-ratings.
Ratings for each feature type.
Feature values and descriptions.
Example:DEBUG: Raw self_ratings: [('q1', 4)]
DEBUG: QuestionSelfRatedUnanswerablePromptWithChoices_int_rating ratings: [4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DEBUG: Added counts: [1, 1, 1, 1, 1]
DEBUG: Final feature vector for qid:19335, did:1017759 (25 features):
DEBUG:   1: 4 (QuestionSelfRatedUnanswerablePromptWithChoices_int_mean_rating_0)
...




---
