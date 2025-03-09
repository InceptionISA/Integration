import os
import re
import pandas as pd
import json
import time
from kaggle.api.kaggle_api_extended import KaggleApi


def get_latest_submission(directory):
    """Finds the latest submission file in the given directory based on the numeric index in the filename."""
    pattern = re.compile(r'submission_(\d+)_.*\.csv')
    max_i = -1
    latest_file = None
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            i = int(match.group(1))
            if i > max_i:
                max_i = i
                latest_file = filename
    if latest_file is None:
        raise FileNotFoundError(f"No submissions found in {directory}")
    return max_i, os.path.join(directory, latest_file)


def get_next_submission_number(main_dir):
    """Determines the next submission number for the main directory."""
    pattern = re.compile(r'submission_(\d+)_.*\.csv')
    max_i = -1
    for filename in os.listdir(main_dir):
        if os.path.isdir(os.path.join(main_dir, filename)):
            continue  # Skip subdirectories
        match = pattern.match(filename)
        if match:
            i = int(match.group(1))
            if i > max_i:
                max_i = i
    return max_i + 1


def get_latest_experiment_number(experiments_dir):
    """Finds the latest experiment number in the experiments directory."""
    pattern = re.compile(r'exp_(\d+)\.json')
    max_num = 0
    for filename in os.listdir(experiments_dir):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num


def main():
    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Directories
    track_dir = os.path.join('submissions', 'Track')
    face_dir = os.path.join('submissions', 'Face')
    main_dir = 'submissions'
    experiments_dir = 'experiments'

    # Ensure directories exist
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)

    # Get latest track and face submissions
    try:
        track_i, track_path = get_latest_submission(track_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    try:
        face_i, face_path = get_latest_submission(face_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Read and merge data
    track_df = pd.read_csv(track_path)
    face_df = pd.read_csv(face_path)
    merged_df = pd.concat([track_df, face_df], ignore_index=True)

    # Drop 'id' column if exists and reset index
    if 'id' in merged_df.columns:
        merged_df = merged_df.drop(columns='id')
    merged_df = merged_df.reset_index().rename(columns={'index': 'ID'})

    # Ensure required columns
    required_columns = ['ID', 'frame', 'objects', 'objective']
    if not all(col in merged_df.columns for col in required_columns):
        print("Error: Merged DataFrame columns are incorrect.")
        return

    # Determine next submission number and save
    next_i = get_next_submission_number(main_dir)
    output_filename = f'submission_{next_i}_.csv'
    output_path = os.path.join(main_dir, output_filename)
    merged_df.to_csv(output_path, index=False)

    # Submit to Kaggle
    competition_name = 'surveillance-for-retail-stores'
    message = f'Merged Track submission {track_i} and Face submission {face_i}'
    try:
        api.competition_submit(output_path, message, competition_name)
        print(f"Submitted {output_path} to Kaggle.")
    except Exception as e:
        print(f"Error submitting to Kaggle: {e}")
        return

    # Retrieve public score
    public_score = None
    try:
        # Wait for submission to process
        time.sleep(60)  # Initial wait
        max_attempts = 6
        for _ in range(max_attempts):
            submissions = api.competition_submissions(competition_name)
            latest_sub = submissions[0]
            if latest_sub.status == 'complete':
                public_score = latest_sub.publicScore if hasattr(
                    latest_sub, 'publicScore') else None
                break
            time.sleep(20)
    except Exception as e:
        print(f"Error retrieving submission score: {e}")

    # Create experiment log
    latest_exp_num = get_latest_experiment_number(experiments_dir)
    new_exp_num = latest_exp_num + 1
    exp_filename = f'exp_{new_exp_num:03d}.json'
    exp_path = os.path.join(experiments_dir, exp_filename)

    experiment_data = {
        "submission": output_path,
        "public_score": public_score,
        "trackfile": track_path,
        "facefile": face_path,
    }

    with open(exp_path, 'w') as f:
        json.dump(experiment_data, f, indent=4)

    print(f"Experiment log saved to {exp_path}")
    if public_score is not None:
        print(f"Public Score: {public_score}")
    else:
        print("Public score could not be retrieved.")


if __name__ == "__main__":
    main()


