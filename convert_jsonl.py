import json
import argparse

def convert(input_json_file, output_json_file):
    with open(input_json_file, 'r') as f:
        lines = f.readlines()

    with open(output_json_file, 'w') as out_file:
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                query_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping line {i + 1} due to JSON error: {e}")
                continue

            if not isinstance(query_data, list) or len(query_data) != 2:
                print(f"[Warning] Unexpected format on line {i + 1}, skipping.")
                continue

            query_id = query_data[0]
            paragraphs = query_data[1]

            for paragraph in paragraphs:
                try:
                    paragraph_id = paragraph['paragraph_id']
                    rankings = paragraph['paragraph_data']['rankings']
                except KeyError as e:
                    print(f"[Warning] Missing expected key in paragraph on line {i + 1}: {e}")
                    continue

                for ranking in rankings:
                    try:
                        rank = ranking['rank']
                        score = ranking['score']
                        method = ranking['method']
                    except KeyError as e:
                        print(f"[Warning] Missing ranking key in line {i + 1}: {e}")
                        continue

                    run_entry = {
                        "query_id": query_id,
                        "document_id": paragraph_id,
                        "rank": rank,
                        "score": score,
                        "method": method
                    }
                    out_file.write(json.dumps(run_entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL with full data to TREC run format")
    parser.add_argument('-input', required=True, help='Path to input JSONL file with rankings')
    parser.add_argument('-output', required=True, help='Path to output JSONL file in run format')
    args = parser.parse_args()
    
    convert(input_json_file=args.input, output_json_file=args.output)
