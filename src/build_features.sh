# python3 build_feature_vectors.py \
# --judgements /home/nf1104/work/data/pencils-down/dl19/Thomas-Sun_few-Sun-HELM-FagB_few-FagB-questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2019-qrels-runs-with-text.jsonl.gz \
# --qrel /home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt \
# --output /feature_vectors/featuresdl19.ranklib \
# --mode questions \
# --no-one-hot \
# --max-query 1 \
# --max-passage 1

input='/home/nf1104/work/data/rubric_format_inputs/flant5large/llmjudge_test_4prompts_qrel_flant5large.jsonl.gz'
# input='/home/nf1104/work/data/rubric_format_inputs/llama3-8b/dl23/test_decomposed_relavance_qrel.jsonl.gz'
# input='/home/nf1104/work/data/rubric_format_inputs/llama3-8b/4_prompts_dl2020.jsonl.gz'
# input='/home/nf1104/work/data/rubric_format_inputs/llama3.3-70b/dl23/test_decomposed_relavance_qrel_llama70b.jsonl.gz'

# o_path='train/llama3.3-70b/dl23'
# f_v_output='./feature_vectors/llama3.3-70b_dl23.ranklib'

o_path='train/flant5/dl23'
f_v_output='./feature_vectors/flant5_dl23.ranklib'

# qrel='/home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt'
# qrel='/home/nf1104/work/data/dl/data/dl2020/2020qrels-pass.txt'
qrel='/home/nf1104/work/data/dl/data/dl2023/converted/new_qrels.txt'

python3 build_feature_vectors.py \
--judgements $input \
--qrel $qrel \
--output $f_v_output \
--criteria-run-dir ./feature_vectors/ \
--mode multi_criteria \
--criteria-run-dir $o_path \
--criterion Exactness \
--no-one-hot \

python3 build_feature_vectors.py \
--judgements $input \
--qrel $qrel \
--output $f_v_output \
--criteria-run-dir ./feature_vectors/ \
--mode multi_criteria \
--criteria-run-dir $o_path \
--criterion Coverage \
--no-one-hot \


python3 build_feature_vectors.py \
--judgements $input \
--qrel $qrel \
--output $f_v_output \
--criteria-run-dir ./feature_vectors/ \
--mode multi_criteria \
--criteria-run-dir $o_path \
--criterion Topicality \
--no-one-hot \

python3 build_feature_vectors.py \
--judgements $input \
--qrel $qrel \
--output $f_v_output \
--criteria-run-dir ./feature_vectors/ \
--mode multi_criteria \
--criteria-run-dir $o_path \
--criterion 'Contextual Fit' \
--no-one-hot \

