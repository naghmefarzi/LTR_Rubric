import os
import glob
import subprocess

def get_run_name(run_path):
    return os.path.splitext(os.path.basename(run_path))[0]

def main():
    base_run_dir = "/home/nf1104/work/data/runs/runs_trecdl2019"
    feature_dir = "/home/nf1104/work/Summer 25/LTR_Rubric/train/flant5/dl19"
    output_root = os.path.join(feature_dir, "filtered_dl19")

    base_runs = glob.glob(os.path.join(base_run_dir, "*.run"))
    feature_runs = glob.glob(os.path.join(feature_dir, "*.run"))

    for base_run in base_runs:
        run_name = get_run_name(base_run)
        output_dir = os.path.join(output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing: {run_name}")
        cmd = [
            "python", "filter_features_by_system_run.py",
            "--base-run", base_run,
            "--feature-runs", *feature_runs,
            "--output-dir", output_dir,
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
