#!/usr/bin/env python3

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np
import logging
import argparse

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

def rating_histogram(queries: List[QueryWithFullParagraphList], mode: str = '') -> Dict[QuestionId, Dict[int, int]]:
    """
    Compute histogram of ratings for questions or criteria.
    
    Args:
        queries: List of QueryWithFullParagraphList objects.
        mode: Feature mode ('nuggets', 'questions', 'multi_criteria', or others).
    """
    result = defaultdict(lambda: defaultdict(lambda: 0))
    for q in queries:
        for para in q.paragraphs:
            for grades in para.retrieve_exam_grade_all(SELF_GRADED):
                for s in grades.self_ratings or []:
                    result[QuestionId(s.get_id())][int(s.self_rating)] += 1
    return dict(result)


def save_criteria_run_files(
    queries: List[QueryWithFullParagraphList],
    output_dir: Path,
    max_query: int = None,
    max_passage: int = None,
    criterion: Optional[str] = None
):
    """
    Save TREC run files for criteria in multi_criteria mode, either for a specific criterion or all.
    
    Args:
        queries: List of QueryWithFullParagraphList objects.
        output_dir: Directory to save run files (e.g., train/).
        max_query: Maximum number of queries to process.
        max_passage: Maximum number of passages per query to process.
        criterion: Specific criterion to generate run file for (e.g., 'Exactness'), or None for all.
    """
    logging.info(f"Saving criteria run files to {output_dir}" + (f" for criterion: {criterion}" if criterion else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    criteria_scores = defaultdict(list)
    gfilt = GradeFilter.noFilter()
    gfilt.prompt_class = 'FourPrompts'  # Match JSON's prompt_class
    
    queries = queries[:max_query] if max_query else queries
    for q in queries:
        qid = QueryId(q.queryId)
        q.paragraphs = q.paragraphs[:max_passage] if max_passage else q.paragraphs
        for para in q.paragraphs:
            did = DocId(para.paragraph_id)
            grades = para.retrieve_exam_grade_all(gfilt)
            for grade in grades:
                for s in grade.self_ratings or []:
                    crit = s.get_id()
                    try:
                        rating = int(s.self_rating)
                        if rating in {0, 1, 2, 3}:
                            criteria_scores[crit].append((qid, did, rating))
                        else:
                            logging.warning(f"Invalid rating {rating} for criterion {crit}")
                    except ValueError:
                        logging.warning(f"Invalid rating for criterion {crit}: {s.self_rating}")
    
    if criterion:
        if criterion not in criteria_scores:
            logging.error(f"No scores found for criterion {criterion}")
            return
        run_file = output_dir / f"{criterion}.run"
        with run_file.open('w') as f:
            for qid, did, rating in criteria_scores[criterion]:
                f.write(f"{qid} 0 {did} 1 {rating} run\n")
        logging.info(f"Wrote {len(criteria_scores[criterion])} lines to {run_file}")
    else:
        for crit, scores in criteria_scores.items():
            run_file = output_dir / f"{crit}.run"
            with run_file.open('w') as f:
                for qid, did, rating in scores:
                    f.write(f"{qid} 0 {did} 1 {rating} run\n")
            logging.info(f"Wrote {len(scores)} lines to {run_file}")

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
    mode: str = '',
    use_one_hot: bool = True,
    max_query: int = None,
    max_passage: int = None,
    criteria_run_dir: Optional[Path] = None,
    criterion: Optional[str] = None
):
    """
    Save feature vectors in RankLib format with mode-based feature selection and debugging.
    
    Args:
        queries: List of QueryWithFullParagraphList objects.
        qrel_path: Path to qrel file with relevance labels.
        output_path: Path to save RankLib feature file.
        mode: Feature mode ('nuggets', 'questions', 'multi_criteria', 'all_rubric_concat', or '' for default).
        use_one_hot: Whether to include one-hot encodings.
        max_query: Maximum number of queries to process.
        max_passage: Maximum number of passages per query to process.
        criteria_run_dir: Directory to save criteria run files (for multi_criteria mode).
        criterion: Specific criterion to generate run file for (e.g., 'Exactness'), or None for all.
    """
    logging.info(f"Processing queries with mode: '{mode}', use_one_hot: {use_one_hot}")
    
    # Save criteria run files if in multi_criteria mode and directory is specified
    if mode == 'multi_criteria' and criteria_run_dir:
        save_criteria_run_files(queries, criteria_run_dir, max_query, max_passage, criterion)
    
    # Define prompt classes based on mode
    PROMPT_CLASSES = {}
    if mode == 'nuggets':
        PROMPT_CLASSES['NuggetSelfRatedPrompt'] = {0, 1, 2, 3, 4, 5}
    elif mode == 'questions':
        PROMPT_CLASSES['QuestionSelfRatedUnanswerablePromptWithChoices'] = {0, 1, 2, 3, 4, 5}
    elif mode == 'multi_criteria':
        PROMPT_CLASSES['FourPrompts'] = {0, 1, 2, 3}
    else:  # mode == 'all_rubric_concat' or default
        PROMPT_CLASSES['NuggetSelfRatedPrompt'] = {0, 1, 2, 3, 4, 5}
        PROMPT_CLASSES['QuestionSelfRatedUnanswerablePromptWithChoices'] = {0, 1, 2, 3, 4, 5}
        PROMPT_CLASSES |= {
            'FagB': {0, 1},
            'FagB_few': {0, 1},
            'HELM': {0, 1},
            'Sun': {0, 1},
            'Sun_few': {0, 1},
            'Thomas': {0, 1, 2},
        }
    logging.info(f"Using prompt classes: {list(PROMPT_CLASSES.keys())}")

    # Load relevance labels
    rels = read_qrel(qrel_path)

    # Compute rating histogram for sorting
    hist = rating_histogram(queries, mode=mode)
    mean_rating = {
        qid: sum(n * r for r, n in ratings.items()) / sum(ratings.values())
        for qid, ratings in hist.items() if sum(ratings.values()) > 0
    }
    logging.debug(f"Computed histogram for {len(hist)} questions/criteria, mean ratings for {len(mean_rating)} items")
    queries = queries[:max_query] if max_query else queries
    with output_path.open('w') as f:
        for q in queries:
            qid = QueryId(q.queryId)
            logging.debug(f"Processing query: {qid}")
            q.paragraphs = q.paragraphs[:max_passage] if max_passage else q.paragraphs
            for para in q.paragraphs:
                did = DocId(para.paragraph_id)
                logging.debug(f"Processing document: {did} for query: {qid}")
                feats = []
                feature_desc = []

                for pclass, valid_range in PROMPT_CLASSES.items():
                    gfilt = GradeFilter.noFilter()
                    gfilt.prompt_class = pclass
                    logging.debug(f"  Prompt class: {pclass}, valid ratings: {valid_range}")

                    grades = para.retrieve_exam_grade_all(gfilt)
                    ratings = [
                        (QuestionId(s.get_id()), s.self_rating)
                        for grade in grades
                        for s in (grade.self_ratings or [])
                        # if mode != 'multi_criteria' or (criterion is None or s.get_id() == criterion)
                    ]
                    logging.debug(f"    Processed ratings: {ratings}")

                    if not ratings:
                        logging.warning(f"    No ratings found for prompt class {pclass}")
                        continue

                    expected_ratings = 10 if pclass in {'NuggetSelfRatedPrompt', 'QuestionSelfRatedUnanswerablePromptWithChoices'} else 4 if pclass == 'FourPrompts' else 1

                    def clamp(x: int) -> int:
                        return 0 if x not in valid_range else x

                    def one_hot_rating(n: int):
                        x = np.zeros(max(valid_range) + 1)
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

                    if len(valid_range) <= 3 and pclass != 'FourPrompts':
                        r = clamp(ratings[0][1])
                        feats.append(np.array([r]))
                        feature_desc.append(f"{pclass}_integer_rating")
                        logging.debug(f"    Added {pclass} integer rating: {r}")
                        if (mode == 'all_rubric_concat' or mode == '') and use_one_hot:
                            feats.append(one_hot_rating(r))
                            feature_desc.append(f"{pclass}_one_hot_{r}")
                            logging.debug(f"    Added {pclass} one-hot: {one_hot_rating(r)}")
                    else:
                        feat, d = rating_feature(
                            sort_key=lambda q: mean_rating.get(q[0], 0),
                            encoding=lambda x: np.array([x]),
                            desc_prefix=f"{pclass}_int_mean_rating"
                        )
                        feats += feat
                        feature_desc += d

                        if (mode in {'all_rubric_concat', '', 'multi_criteria'} or pclass == 'FourPrompts') and use_one_hot:
                            feat, d = rating_feature(
                                sort_key=lambda q: mean_rating.get(q[0], 0),
                                encoding=one_hot_rating,
                                desc_prefix=f"{pclass}_one_hot_mean_rating"
                            )
                            feats += feat
                            feature_desc += [f"{desc}_{j}" for desc in d for j in range(max(valid_range) + 1)]

                            if pclass != 'FourPrompts':
                                feat, d = rating_feature(
                                    sort_key=lambda q: hist.get(q[0], {}).get(4, 0) + hist.get(q[0], {}).get(5, 0),
                                    encoding=one_hot_rating,
                                    desc_prefix=f"{pclass}_one_hot_informativeness"
                                )
                                feats += feat
                                feature_desc += [f"{desc}_{j}" for desc in d for j in range(max(valid_range) + 1)]

                        feat, d = rating_feature(
                            sort_key=lambda q: q[1],
                            encoding=lambda x: np.array([x]),
                            desc_prefix=f"{pclass}_int_rating"
                        )
                        feats += feat
                        feature_desc += d

                        if (mode in {'all_rubric_concat', '', 'multi_criteria'} or pclass == 'FourPrompts') and use_one_hot:
                            feat, d = rating_feature(
                                sort_key=lambda q: q[1],
                                encoding=one_hot_rating,
                                desc_prefix=f"{pclass}_one_hot_rating"
                            )
                            feats += feat
                            feature_desc += [f"{desc}_{j}" for desc in d for j in range(max(valid_range) + 1)]

                        counts = [sum(1 for _, r in ratings if r >= n) for n in range(max(valid_range))]
                        feats += [np.array([c]) for c in counts]
                        feature_desc += [f"{pclass}_count_geq_{n}" for n in range(max(valid_range))]
                        logging.debug(f"    Added {pclass} counts: {counts}")

                feature_vector = np.hstack(feats)
                logging.debug(f"Final feature vector for qid:{qid}, did:{did} ({len(feature_vector)} features):")
                for i, (val, desc) in enumerate(zip(feature_vector, feature_desc)):
                    logging.debug(f"  {i+1}: {val} ({desc})")

                label = rels.get((qid, did), 0)
                logging.debug(f"Relevance label: {label}")

                feature_str = " ".join(f"{i+1}:{v}" for i, v in enumerate(feature_vector))
                f.write(f"{label} qid:{qid} {feature_str} # {did}\n")
                logging.debug(f"Wrote RankLib line: {label} qid:{qid} ... # {did}")
def main():
    parser = argparse.ArgumentParser(description="Save features in RankLib format with mode-based selection")
    parser.add_argument('--judgements', '-j', type=Path, required=True, help='exampp judgements file (JSONL.gz)')
    parser.add_argument('--qrel', '-q', type=Path, required=True, help='Query relevance file')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output RankLib feature file')
    parser.add_argument('--mode', type=str, default='', choices=['', 'nuggets', 'questions', 'all_rubric_concat', 'multi_criteria'],
                        help='Feature mode: nuggets, questions, multi_criteria, all_rubric_concat, or empty for default')
    parser.add_argument('--criterion', type=str, required=False, choices=['Exactness', 'Topicality', 'Coverage', 'Contextual Fit'],
                        help='Specific criterion to generate run file for (multi_criteria mode only)')
    parser.add_argument('--criteria-run-dir', type=Path, required=False, help='Directory to save criteria run files (multi_criteria mode)')
    parser.add_argument('--max-query', type=int, required=False, help='Max number of queries to process')
    parser.add_argument('--max-passage', type=int, required=False, help='Max number of passages to process')
    parser.add_argument('--no-one-hot', action='store_false', dest='use_one_hot', help='Disable one-hot encodings')
    args = parser.parse_args()

    logging.info(f"Loading judgements from {args.judgements}")
    queries = parseQueryWithFullParagraphs(args.judgements)
    logging.info(f"Loaded {len(queries)} queries")
    save_ranklib_features(
        queries, args.qrel, args.output, mode=args.mode, use_one_hot=args.use_one_hot,
        max_query=args.max_query, max_passage=args.max_passage,
        criteria_run_dir=args.criteria_run_dir, criterion=args.criterion
    )
if __name__ == "__main__":
    main()