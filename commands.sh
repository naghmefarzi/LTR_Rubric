# python3 build_feature_vectors.py \
# --judgements /home/nf1104/work/data/pencils-down/dl19/Thomas-Sun_few-Sun-HELM-FagB_few-FagB-questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2019-qrels-runs-with-text.jsonl.gz \
# --qrel /home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt \
# --output /feature_vectors/featuresdl19.ranklib \
# --mode questions \
# --no-one-hot \
# --max-query 1 \
# --max-passage 1




python3 build_feature_vectors.py \
--judgements /home/nf1104/work/data/rubric_format_inputs/flant5large/dl2019_test_4prompts_flant5large_wo_system_message.jsonl.gz \
--qrel /home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt \
--output ./feature_vectors/flant5_exactness.ranklib \
--criteria-run-dir ./feature_vectors/ \
--mode multi_criteria \
--criteria-run-dir train \
--criterion Exactness \
--max-query 1 \
--max-passage 1 \
--no-one-hot \


