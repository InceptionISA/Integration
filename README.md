# Kaggle Submission Utilities

A Python module for automating Kaggle competition submissions, specifically designed for merging track and face detection results for retail surveillance competitions.

## Features

- Automated submission management for Kaggle competitions
- Merging of track and face detection results
- Experiment tracking and recording
- Robust error handling and logging
- Modular design for easy extension or partial imports

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/kaggle-submission-utils.git
   cd kaggle-submission-utils
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Kaggle API Setup

This tool requires Kaggle API credentials. There are two ways to set this up:

### Option 1: Using kaggle.json (Recommended)

1. Log in to your Kaggle account
2. Go to "Account" > "API" section
3. Click "Create New API Token" to download your `kaggle.json` file
4. Place the `kaggle.json` file in one of these locations:
   - The root directory of this project
   - `~/.kaggle/kaggle.json` (Linux/macOS)
   - `C:\Users\<Windows-username>\.kaggle\kaggle.json` (Windows)

### Option 2: Using Environment Variables

Set the following environment variables:
```
export KAGGLE_USERNAME=yourusername
export KAGGLE_KEY=yourkey
```

## Directory Structure

The utility expects the following directory structure:
```
project_root/
├── submissions/
│   ├── Track/           # Track detection submissions
│   ├── Face/            # Face detection submissions
│   └── (merged submissions will be placed here)
├── experiments/         # Experiment records
└── logs/                # Log files (created automatically)
```

## Usage

### Basic Usage

```python
from submission_utils import run_submission_pipeline

# Run the complete pipeline with default settings
results = run_submission_pipeline()

# Print the public score
if results.get('public_score'):
    print(f"Public score: {results['public_score']}")
```

### Advanced Usage

```python
from submission_utils import (
    ConfigManager, 
    SubmissionManager, 
    ExperimentManager, 
    KaggleSubmitter
)

# Initialize components
config = ConfigManager(
    base_dir="./my_project",
    competition_name="my-competition"
)
submission_mgr = SubmissionManager(config)
experiment_mgr = ExperimentManager(config)
kaggle_submitter = KaggleSubmitter(config)

# Authenticate with Kaggle
kaggle_submitter.authenticate(kaggle_config_path="./my_credentials")

# Find latest submissions
track_metadata = submission_mgr.get_latest_submission(config.get_path('track'))
face_metadata = submission_mgr.get_latest_submission(config.get_path('face'))

# Merge and submit
merged_df = submission_mgr.merge_submissions(
    track_metadata.file_path, 
    face_metadata.file_path
)
output_path = submission_mgr.save_submission(merged_df, config.get_path('main'))
```

## Running as a Script

You can also run the utility as a standalone script:

```
python submission_utils.py
```

## Configuration Options

When calling `run_submission_pipeline()`, you can specify the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_dir` | Base directory for operations | `.` (current directory) |
| `competition_name` | Name of the Kaggle competition | `surveillance-for-retail-stores` |
| `kaggle_config_path` | Path to Kaggle credentials | `None` (use default) |
| `wait_time` | Time to wait after submission before checking score (seconds) | 90 |
| `max_retries` | Maximum number of retry attempts for checking score | 8 |
| `retry_interval` | Time between retry attempts (seconds) | 15 |

## Error Handling

The utility includes comprehensive error handling and logging. Logs are saved to the `logs` directory and also output to the console.

