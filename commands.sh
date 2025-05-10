python3 build_feature_vectors.py \
--judgements /home/nf1104/work/data/pencils-down/dl19/Thomas-Sun_few-Sun-HELM-FagB_few-FagB-questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2019-qrels-runs-with-text.jsonl.gz \
--qrel /home/nf1104/work/data/dl/data/dl2019/2019qrels-pass.txt \
--output featuresdl19.ranklib \
--mode questions \
--no-one-hot \
--max-query 1 \
--max-passage 1