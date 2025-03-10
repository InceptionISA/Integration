"""
Kaggle Submission Utilities

A module for handling Kaggle competition submissions, particularly for merging
track and face detection results.
"""

import os
import sys
import re
import pandas as pd
import json
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass
from kaggle.api.kaggle_api_extended import KaggleApi
from utils.configuration import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('kaggle_submitter')


@dataclass
class SubmissionMetadata:
    """Class for storing submission file metadata."""
    index: int
    model_name: str
    file_path: Path


class SubmissionManager:
    """Manages finding, creating, and handling submission files."""

    def __init__(self, config: ConfigManager) -> None:
        """
        Initialize the SubmissionManager.

        Args:
            config: Configuration manager instance
        """
        self.config = config

    def get_latest_submission(self, directory: Path) -> SubmissionMetadata:
        """
        Find the latest submission file in the given directory.

        Args:
            directory: Directory to search for submission files

        Returns:
            SubmissionMetadata object with information about the latest submission

        Raises:
            FileNotFoundError: If no valid submission files are found
        """
        pattern = re.compile(r'^submission_(\d+)_(.*?)\.csv$')
        max_i = -1
        latest_file = None
        best_model = None

        try:
            for filename in directory.iterdir():
                if not filename.is_file():
                    continue

                match = pattern.match(filename.name)
                if match:
                    i = int(match.group(1))
                    model = match.group(2)
                    if i > max_i:
                        max_i = i
                        latest_file = filename
                        best_model = model

            if latest_file is None:
                raise FileNotFoundError(f"No valid submissions in {directory}")

            return SubmissionMetadata(max_i, best_model, latest_file)

        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Error accessing directory {directory}: {e}")
            raise

    def get_specific_submission(self, directory: Path, filename: str) -> SubmissionMetadata:
        pattern = re.compile(r'^submission_(\d+)_(.*?)\.csv$')

        try:
            file_path = directory / filename
            if not file_path.exists() or not file_path.is_file():
                logger.error(f"File {filename} not found in {directory}")
                sys.exit(1)

            match = pattern.match(filename)
            if match:
                i = int(match.group(1))
                model = match.group(2)
                return SubmissionMetadata(i, model, file_path)
            else:
                # For files that don't match the pattern, use default values
                logger.warning(
                    f"File {filename} doesn't match the expected pattern")
                return SubmissionMetadata(0, "custom", file_path)

        except (PermissionError) as e:
            logger.error(f"Error accessing file {filename} in {directory}: {e}")
            sys.exit(1)

    def get_next_submission_number(self, directory: Path) -> int:
        """
        Determine the next submission number with zero-padding.

        Args:
            directory: Directory to check for existing submissions

        Returns:
            Next submission number
        """
        pattern = re.compile(r'^submission_(\d+)\.csv$')
        max_i = -1

        try:
            for filename in directory.iterdir():
                if not filename.is_file():
                    continue

                if filename.name.startswith('submission_') and filename.name.endswith('.csv'):
                    match = pattern.match(filename.name)
                    if match:
                        i = int(match.group(1))
                        max_i = max(max_i, i)

            return max_i + 1

        except (FileNotFoundError, PermissionError):
            # If the directory doesn't exist or can't be accessed, start from 0
            return 0

    def merge_submissions(self, track_path: Path, face_path: Path) -> pd.DataFrame:
        """
        Merge track and face submissions into a single DataFrame.

        Args:
            track_path: Path to the track submission file
            face_path: Path to the face submission file

        Returns:
            Merged DataFrame

        Raises:
            ValueError: If the data is invalid or incompatible
        """
        try:
            # Read input files
            track_df = pd.read_csv(track_path)
            face_df = pd.read_csv(face_path)

            # Validate dataframes
            required_columns = ['frame', 'objects', 'objective']
            for df, name in [(track_df, 'track'), (face_df, 'face')]:
                missing = [
                    col for col in required_columns if col not in df.columns]
                if missing:
                    raise ValueError(
                        f"{name} data is missing required columns: {missing}")

            # Merge data
            merged = pd.concat([track_df, face_df], ignore_index=True)

            # Format merged submission
            # First, check if 'id' column exists before dropping
            if 'id' in merged.columns:
                merged = merged.drop(columns=['id'])

            merged = merged.reset_index()
            merged = merged.rename(columns={'index': 'ID'})

            # Ensure correct column order
            merged = merged[['ID', 'frame', 'objects', 'objective']]

            return merged

        except Exception as e:
            logger.error(f"Error merging submissions: {e}")
            raise

    def save_submission(self, merged_df: pd.DataFrame, directory: Path) -> Path:
        """
        Save the merged submission to a file.

        Args:
            merged_df: DataFrame to save
            directory: Directory to save the file in

        Returns:
            Path to the saved file
        """
        submission_num = self.get_next_submission_number(directory)
        output_path = directory / f'submission_{submission_num:02d}.csv'

        try:
            merged_df.to_csv(output_path, index=False)
            logger.info(f"Saved merged submission to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving submission: {e}")
            raise


class ExperimentManager:
    """Manages experiment tracking and records."""

    def __init__(self, config: ConfigManager) -> None:
        """
        Initialize the ExperimentManager.

        Args:
            config: Configuration manager instance
        """
        self.config = config

    def get_latest_experiment_number(self, directory: Path) -> int:
        """
        Find the highest existing experiment number.

        Args:
            directory: Directory to search for experiment files

        Returns:
            Highest experiment number found, or 0 if none exist
        """
        pattern = re.compile(r'^exp_(\d{3})__.*\.json$')
        max_num = 0

        try:
            for filename in directory.iterdir():
                if not filename.is_file():
                    continue

                match = pattern.match(filename.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)

            return max_num

        except (FileNotFoundError, PermissionError):
            # If the directory doesn't exist or can't be accessed, start from 0
            return 0

    def save_experiment_record(self,
                               submission_path: Path,
                               track_metadata: SubmissionMetadata,
                               face_metadata: SubmissionMetadata,
                               public_score: Optional[float] = None,
                               submitted_by: str = "automated") -> Path:
        """
        Create and save an experiment record.

        Args:
            submission_path: Path to the merged submission file
            track_metadata: Metadata for the track submission
            face_metadata: Metadata for the face submission
            public_score: Public score from Kaggle, if available
            submitted_by: GitHub username of the person who triggered the action

        Returns:
            Path to the saved experiment record
        """
        exp_dir = self.config.get_path('experiments')
        exp_num = self.get_latest_experiment_number(exp_dir) + 1

        exp_filename = f'exp_{exp_num:03d}__{face_metadata.model_name}__and__{track_metadata.model_name}__.json'
        exp_path = exp_dir / exp_filename

        experiment_data = {
            "submission": str(submission_path),
            "public_score": public_score,
            "trackfile": str(track_metadata.file_path),
            "facefile": str(face_metadata.file_path),
            "track_model": track_metadata.model_name,
            "face_model": face_metadata.model_name,
            "submitted_by": submitted_by,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            with open(exp_path, 'w') as f:
                json.dump(experiment_data, f, indent=2)

            logger.info(f"Experiment summary saved to {exp_path}")
            return exp_path

        except Exception as e:
            logger.error(f"Error saving experiment record: {e}")
            raise


class KaggleSubmitter:
    """Handles interactions with the Kaggle API."""

    def __init__(self, config: ConfigManager) -> None:
        """
        Initialize the KaggleSubmitter.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.api = None

    def authenticate(self, kaggle_config_path: Optional[str] = None) -> None:
        """
        Authenticate with the Kaggle API.

        Args:
            kaggle_config_path: Path to the Kaggle API credentials file

        Raises:
            RuntimeError: If authentication fails
        """
        kaggle_username = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")

        if not kaggle_username or not kaggle_key:
            logger.error(f"Kaggle API authentication failed")
            sys.exit(1)

        logger.info(
            "Using Kaggle API credentials from environment variables")

        # Create Kaggle API credentials file
        kaggle_config = {
            "username": kaggle_username,
            "key": kaggle_key
        }

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)

        config_path = os.path.join(kaggle_dir, "kaggle.json")
        with open(config_path, "w") as f:
            json.dump(kaggle_config, f)

        os.chmod(config_path, 0o600)  # Secure file permissions

        # Authenticate with Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()
        logger.info("Kaggle API authentication successful")

    def submit_to_kaggle(self,
                         file_path: Path,
                         message: str,
                         wait_time: int = 60,
                         max_retries: int = 5,
                         retry_interval: int = 15) -> Optional[float]:
        """
        Submit a file to Kaggle and wait for the score.

        Args:
            file_path: Path to the submission file
            message: Submission message
            wait_time: Initial wait time before checking score (seconds)
            max_retries: Maximum number of retries for checking score
            retry_interval: Time between retries (seconds)

        Returns:
            Public score if available, otherwise None

        Raises:
            RuntimeError: If submission fails
        """
        if not self.api:
            raise RuntimeError(
                "Kaggle API not authenticated. Call authenticate() first.")

        try:
            logger.info(
                f"Submitting {file_path} to Kaggle competition {self.config.competition_name}")
            self.api.competition_submit(
                file_name=str(file_path),
                message=message,
                competition=self.config.competition_name
            )
            logger.info("Submission successful!")

            # Wait for the submission to be processed
            logger.info(f"Waiting {wait_time} seconds for processing...")
            time.sleep(wait_time)

            # Check for the score with retries
            public_score = None
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Checking score (attempt {attempt+1}/{max_retries})...")
                    submissions = self.api.competition_submissions(
                        self.config.competition_name)

                    if not submissions:
                        logger.warning("No submissions found")
                        time.sleep(retry_interval)
                        continue

                    latest = submissions[0]

                    if latest.status == 'complete':
                        public_score = float(latest.publicScore)
                        logger.info(f"Public score: {public_score}")
                        break
                    else:
                        logger.info(f"Submission status: {latest.status}")

                    time.sleep(retry_interval)

                except Exception as e:
                    logger.error(
                        f"Error checking score (attempt {attempt+1}): {e}")
                    time.sleep(retry_interval)

            return public_score

        except Exception as e:
            logger.error(f"Submission failed: {e}")
            raise RuntimeError(f"Kaggle submission failed: {e}")


def run_submission_pipeline(
    base_dir: str = ".",
    competition_name: str = "surveillance-for-retail-stores",
    kaggle_config_path: Optional[str] = None,
    wait_time: int = 90,
    max_retries: int = 8,
    retry_interval: int = 15,
    submitted_by: str = "automated",
    track_file: str = "",
    face_file: str = ""
) -> Dict[str, Any]:
    """
    Run the complete submission pipeline.

    Args:
        base_dir: Base directory for all operations
        competition_name: Name of the Kaggle competition
        kaggle_config_path: Path to Kaggle credentials
        wait_time: Time to wait after submission before checking score
        max_retries: Maximum number of retry attempts for checking score
        retry_interval: Time between retry attempts
        submitted_by: GitHub username of the person who triggered the action
        track_file: Optional specific track file name to use
        face_file: Optional specific face file name to use

    Returns:
        Dictionary with pipeline results (paths, metadata, score)

    Raises:
        Exception: If any step of the pipeline fails
    """
    # Initialize components
    config = ConfigManager(base_dir, competition_name)
    submission_mgr = SubmissionManager(config)
    experiment_mgr = ExperimentManager(config)
    kaggle_submitter = KaggleSubmitter(config)

    results = {}

    try:
        # Step 1: Authenticate with Kaggle
        logger.info("Step 1: Authenticating with Kaggle API")
        kaggle_submitter.authenticate(kaggle_config_path)
        results['authentication'] = "success"

        # Step 2: Get track and face submissions (either specified or latest)
        logger.info("Step 2: Finding submissions")

        # Get track submission (specific or latest)
        if track_file and track_file.strip():
            logger.info(f"Using specified track file: {track_file}")
            track_metadata = submission_mgr.get_specific_submission(
                config.get_path('track'), track_file)
        else:
            logger.info("Using latest track submission")
            track_metadata = submission_mgr.get_latest_submission(
                config.get_path('track'))

        # Get face submission (specific or latest)
        if face_file and face_file.strip():
            logger.info(f"Using specified face file: {face_file}")
            face_metadata = submission_mgr.get_specific_submission(
                config.get_path('face'), face_file)
        else:
            logger.info("Using latest face submission")
            face_metadata = submission_mgr.get_latest_submission(
                config.get_path('face'))

        logger.info(
            f"Track submission: {track_metadata.file_path.name} (model: {track_metadata.model_name})")
        logger.info(
            f"Face submission: {face_metadata.file_path.name} (model: {face_metadata.model_name})")

        results['track_metadata'] = {
            'index': track_metadata.index,
            'model': track_metadata.model_name,
            'file': str(track_metadata.file_path)
        }
        results['face_metadata'] = {
            'index': face_metadata.index,
            'model': face_metadata.model_name,
            'file': str(face_metadata.file_path)
        }

        # Step 3: Merge submissions
        logger.info("Step 3: Merging submissions")
        merged_df = submission_mgr.merge_submissions(
            track_metadata.file_path, face_metadata.file_path)
        results['merge'] = "success"

        # Step 4: Save merged submission
        logger.info("Step 4: Saving merged submission")
        output_path = submission_mgr.save_submission(
            merged_df, config.get_path('main'))
        results['output_path'] = str(output_path)

        # Step 5: Submit to Kaggle
        logger.info("Step 5: Submitting to Kaggle")
        message = f"Merged Track ({track_metadata.model_name}) and Face ({face_metadata.model_name}) - Submitted by {submitted_by}"
        public_score = kaggle_submitter.submit_to_kaggle(
            output_path,
            message,
            wait_time=wait_time,
            max_retries=max_retries,
            retry_interval=retry_interval
        )
        results['submission'] = "success"
        results['public_score'] = public_score
        results['submitted_by'] = submitted_by

        # Step 6: Save experiment record
        logger.info("Step 6: Saving experiment record")
        exp_path = experiment_mgr.save_experiment_record(
            output_path,
            track_metadata,
            face_metadata,
            public_score,
            submitted_by
        )
        results['experiment_path'] = str(exp_path)

        logger.info("Submission pipeline completed successfully!")

        return results

    except Exception as e:
        logger.error(f"Submission pipeline failed: {e}")
        results['error'] = str(e)
        return results


if __name__ == "__main__":
    # Run the pipeline with default settings
    results = run_submission_pipeline()

    # Print summary
    print("\nSubmission Summary:")

    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    else:
        print(f"Track model: {results['track_metadata']['model']}")
        print(f"Face model: {results['face_metadata']['model']}")
        print(f"Output file: {results['output_path']}")
        print(f"Experiment record: {results['experiment_path']}")

        if results.get('public_score'):
            print(f"Public score: {results['public_score']}")
        else:
            print("Public score: Not available")
