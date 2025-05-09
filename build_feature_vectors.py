#!/usr/bin/env python3

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np
import logging

# Assume exam_pp.data_model provides these
from exam_pp.data_model import QueryWithFullParagraphList, GradeFilter, parseQueryWithFullParagraphs

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Type definitions
QueryId = str
DocId = str
QuestionId = str

SELF_GRADED = GradeFilter.noFilter()
SELF_GRADED.is_self_rated = True

def rating_histogram(queries: List[QueryWithFullParagraphList]) -> Dict[QuestionId, Dict[int, int]]:
    result = defaultdict(lambda: defaultdict(lambda: 0))
    for q in queries:
        for para in q.paragraphs:
            for grades in para.retrieve_exam_grade_all(SELF_GRADED):
                for s in grades.self_ratings or []:
                    result[QuestionId(s.get_id())][int(s.self_rating)] += 1
    return dict(result)

def read_qrel(f: Path) -> Dict[Tuple[QueryId, DocId], int]:
    rels = {}
    with f.open('r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                logging.warning(f"Skipping malformed line: {line.strip()}")
                continue
            try:
                qid = QueryId(parts[0])
                did = DocId(parts[2])
                rel = int(parts[3])
                rels[(qid, did)] = rel
            except (IndexError, ValueError) as e:
                logging.error(f"Error processing line: {line.strip()}, error: {e}")
                continue
    logging.debug(f"Loaded {len(rels)} relevance labels from {f}")
    return rels

def save_ranklib_features(
    queries: List[QueryWithFullParagraphList],
    qrel_path: Path,
    output_path: Path,
    prompt_classes={'nuggets', 'questions', 'direct'}
):
    """
    Save feature vectors in RankLib format with detailed debugging output.
    
    Args:
        queries: List of QueryWithFullParagraphList objects.
        qrel_path: Path to qrel file with relevance labels.
        output_path: Path to save RankLib feature file.
        prompt_classes: Set of prompt classes to consider.
    """
    logging.info(f"Processing queries with prompt classes: {prompt_classes}")
    # Load relevance labels
    rels = read_qrel(qrel_path)

    # Compute rating histogram for sorting
    hist = rating_histogram(queries)
    mean_rating = {
        qid: sum(n * r for r, n in ratings.items()) / sum(ratings.values())
        for qid, ratings in hist.items() if sum(ratings.values()) > 0
    }
    logging.debug(f"Computed histogram for {len(hist)} questions, mean ratings for {len(mean_rating)} questions")

    # Define prompt classes and valid ratings
    PROMPT_CLASSES = {}
    if 'nuggets' in prompt_classes:
        PROMPT_CLASSES['NuggetSelfRatedPrompt'] = {0, 1, 2, 3, 4, 5}
    if 'questions' in prompt_classes:
        PROMPT_CLASSES['QuestionSelfRatedUnanswerablePromptWithChoices'] = {0, 1, 2, 3, 4, 5}
    if 'direct' in prompt_classes:
        PROMPT_CLASSES |= {
            'FagB': {0, 1},
            'FagB_few': {0, 1},
            'HELM': {0, 1},
            'Sun': {0, 1},
            'Sun_few': {0, 1},
            'Thomas': {0, 1, 2},
        }
    logging.info(f"Using prompt classes: {list(PROMPT_CLASSES.keys())}")

    with output_path.open('w') as f:
        for q in queries:
            qid = QueryId(q.queryId)
            logging.debug(f"Processing query: {qid}")
            for para in q.paragraphs:
                did = DocId(para.paragraph_id)
                logging.debug(f"Processing document: {did} for query: {qid}")
                feats = []
                feature_desc = []  # Track feature descriptions for debugging

                for pclass, valid_range in PROMPT_CLASSES.items():
                    gfilt = GradeFilter.noFilter()
                    gfilt.prompt_class = pclass
                    logging.debug(f"  Prompt class: {pclass}, valid ratings: {valid_range}")

                    # Retrieve grades and log raw self_ratings
                    grades = para.retrieve_exam_grade_all(gfilt)
                    raw_ratings = [
                        (s.get_id(), s.self_rating)
                        for grade in grades
                        for s in (grade.self_ratings or [])
                    ]
                    logging.debug(f"    Raw self_ratings: {raw_ratings}")

                    ratings = [
                        (QuestionId(s.get_id()), s.self_rating)
                        for grades in para.retrieve_exam_grade_all(gfilt)
                        for s in grades.self_ratings or []
                    ]
                    logging.debug(f"    Processed ratings: {ratings}")

                    if not ratings:
                        logging.warning(f"    No ratings found for prompt class {pclass}")
                        continue

                    expected_ratings = 10

                    def clamp(x: int) -> int:
                        return 0 if x not in valid_range else x

                    def one_hot_rating(n: int):
                        x = np.zeros(6 if len(valid_range) > 3 else len(valid_range))
                        x[clamp(n)] = 1
                        return x

                    def rating_feature(sort_key, encoding, desc_prefix):
                        sorted_ratings = sorted(ratings, key=sort_key, reverse=True)
                        padded_ratings = (
                            [clamp(rating) for _, rating in sorted_ratings][:expected_ratings] +
                            [0] * (expected_ratings - len(ratings))
                        )
                        logging.debug(f"    {desc_prefix} ratings: {padded_ratings}")
                        return [encoding(rating) for rating in padded_ratings], [
                            f"{desc_prefix}_{i}" for i in range(expected_ratings)
                        ]

                    if len(valid_range) <= 3:
                        r = clamp(ratings[0][1])
                        feats += [np.array([r]), one_hot_rating(r)]
                        feature_desc += [f"{pclass}_integer_rating", f"{pclass}_one_hot_{r}"]
                        logging.debug(f"    Added {pclass} integer rating: {r}, one-hot: {one_hot_rating(r)}")
                    else:
                        # Integer ratings sorted by mean question rating
                        f, d = rating_feature(
                            sort_key=lambda q: mean_rating.get(q[0], 0),
                            encoding=lambda x: np.array([x]),
                            desc_prefix=f"{pclass}_int_mean_rating"
                        )
                        feats += f
                        feature_desc += d

                        # One-hot ratings sorted by mean question rating
                        f, d = rating_feature(
                            sort_key=lambda q: mean_rating.get(q[0], 0),
                            encoding=one_hot_rating,
                            desc_prefix=f"{pclass}_one_hot_mean_rating"
                        )
                        feats += f
                        feature_desc += [f"{desc}_{j}" for desc in d for j in range(6)]

                        # One-hot ratings sorted by question informativeness
                        f, d = rating_feature(
                            sort_key=lambda q: hist.get(q[0], {}).get(4, 0) + hist.get(q[0], {}).get(5, 0),
                            encoding=one_hot_rating,
                            desc_prefix=f"{pclass}_one_hot_informativeness"
                        )
                        feats += f
                        feature_desc += [f"{desc}_{j}" for desc in d for j in range(6)]

                        # Integer ratings sorted by rating
                        f, d = rating_feature(
                            sort_key=lambda q: q[1],
                            encoding=lambda x: np.array([x]),
                            desc_prefix=f"{pclass}_int_rating"
                        )
                        feats += f
                        feature_desc += d

                        # One-hot ratings sorted by rating
                        f, d = rating_feature(
                            sort_key=lambda q: q[1],
                            encoding=one_hot_rating,
                            desc_prefix=f"{pclass}_one_hot_rating"
                        )
                        feats += f
                        feature_desc += [f"{desc}_{j}" for desc in d for j in range(6)]

                        # Number of questions answered with N or better
                        counts = [sum(1 for _, r in ratings if r >= n) for n in range(5)]
                        feats += [np.array([c]) for c in counts]
                        feature_desc += [f"{pclass}_count_geq_{n}" for n in range(5)]
                        logging.debug(f"    Added {pclass} counts: {counts}")

                # Flatten features
                feature_vector = np.hstack(feats)
                logging.debug(f"Final feature vector for qid:{qid}, did:{did} ({len(feature_vector)} features):")
                for i, (val, desc) in enumerate(zip(feature_vector, feature_desc)):
                    logging.debug(f"  {i+1}: {val} ({desc})")

                # Get relevance label
                label = rels.get((qid, did), 0)
                logging.debug(f"Relevance label: {label}")

                # Write to RankLib format
                feature_str = " ".join(f"{i+1}:{v}" for i, v in enumerate(feature_vector))
                f.write(f"{label} qid:{qid} {feature_str} # {did}\n")
                logging.debug(f"Wrote RankLib line: {label} qid:{qid} ... # {did}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Save features in RankLib format with debugging")
    parser.add_argument('--judgements', '-j', type=Path, required=True, help='exampp judgements file (JSONL.gz)')
    parser.add_argument('--qrel', '-q', type=Path, required=True, help='Query relevance file')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output RankLib feature file')
    args = parser.parse_args()

    logging.info(f"Loading judgements from {args.judgements}")
    queries = parseQueryWithFullParagraphs(args.judgements)
    logging.info(f"Loaded {len(queries)} queries")
    save_ranklib_features(queries, args.qrel, args.output)

if __name__ == "__main__":
    main()