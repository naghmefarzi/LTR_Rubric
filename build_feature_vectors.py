import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import gzip

# Parse JSONL and Extract Grades
def parse_jsonl(file_path, eval_type='nugget', max_features=20):
    data = []
    all_ids = {}  # Track nugget/question IDs and their frequency
    # Use gzip.open for .gz files
    open_func = gzip.open if file_path.endswith('.gz') else open

    with open_func(file_path, 'rt', encoding='utf-8') as f:

        for line in f:
            try:
                record = json.loads(line.strip())
                query_id = record[0]
                for passage in record[1]:
                    passage_id = passage['paragraph_id']
                    relevance = next(
                        (j['relevance'] for j in passage['paragraph_data']['judgments'] if j['query'] == query_id),
                        0
                    )  # Default to 0 if no judgment
                    
                    # Extract grades
                    grades = {}
                    found_eval = False
                    for exam in passage['exam_grades']:
                        if exam['prompt_type'] == eval_type and 'self_ratings' in exam:
                            found_eval = True
                            for rating in exam['self_ratings']:
                                item_id = rating.get('nugget_id' if eval_type == 'nugget' else 'question_id')
                                score = rating.get('self_rating', 0)
                                if item_id:
                                    grades[item_id] = score
                                    all_ids[item_id] = all_ids.get(item_id, 0) + 1
                    
                    if not found_eval:
                        print(f"Warning: No {eval_type} evaluations found for query {query_id}, passage {passage_id}")
                    
                    data.append({
                        'query_id': query_id,
                        'passage_id': passage_id,
                        'grades': grades,
                        'relevance': relevance
                    })
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    # Filter IDs to those appearing in at least 10% of pairs
    min_count = len(data) * 0.1
    id_list = sorted([id_ for id_, count in all_ids.items() if count >= min_count])[:max_features]
    if not id_list:
        raise ValueError(f"No {eval_type} IDs found with sufficient coverage")
    
    print(f"Selected {len(id_list)} {eval_type} IDs for features")
    return pd.DataFrame(data), id_list

# Build Feature Vector from Grades
def build_feature_vector(grades, id_list):
    vector = np.zeros(len(id_list))
    for i, item_id in enumerate(id_list):
        vector[i] = grades.get(item_id, 0)  # Default to 0 if ID not present
    if vector.sum() == 0:
        print(f"Warning: All-zero vector for grades: {grades}")
    return vector

# Format for RankLib
def save_ranklib_file(feature_vectors, relevances, query_ids, passage_ids, output_file):
    with open(output_file, 'w') as f:
        for i, (vector, relevance, qid, pid) in enumerate(zip(feature_vectors, relevances, query_ids, passage_ids)):
            features = ' '.join([f'{j+1}:{v}' for j, v in enumerate(vector)])
            line = f"{int(relevance)} qid:{qid} {features} # docid:{pid}\n"
            f.write(line)
    print(f"RankLib file saved to {output_file}")

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Extract grades as feature vectors for LTR")
    parser.add_argument('--input', type=str, required=True, help="Path to JSONL file")
    parser.add_argument('--output', type=str, required=True, help="Path to save RankLib file")
    parser.add_argument('--eval_type', type=str, choices=['nugget', 'question'], default='nugget', help="Evaluation type: nugget or question")
    parser.add_argument('--max_features', type=int, default=20, help="Maximum number of features (nuggets/questions)")
    args = parser.parse_args()

    # Parse JSONL
    df, id_list = parse_jsonl(args.input, eval_type=args.eval_type, max_features=args.max_features)
    
    # Generate feature vectors
    feature_vectors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building feature vectors"):
        vector = build_feature_vector(row['grades'], id_list)
        feature_vectors.append(vector)
    
    # Validate non-zero vectors
    non_zero_count = sum(1 for v in feature_vectors if v.sum() > 0)
    print(f"Generated {non_zero_count}/{len(feature_vectors)} non-zero feature vectors")
    
    # Save for RankLib
    save_ranklib_file(feature_vectors, df['relevance'], df['query_id'], df['passage_id'], args.output)
    


if __name__ == '__main__':
    main()