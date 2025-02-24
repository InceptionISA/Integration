# Integration Repository

This repository serves as the central hub for merging outputs from the **tracking** and **face re-identification** projects, tracking experiments, and generating submission files for the competition.

## Directory Structure

- **experiments/**: Stores logs for each experiment (e.g., hyperparameters, results).
- **reports/**: Contains generated reports and visualizations (optional).
- **merge_submission.py**: Script that merges results from the other two repositories into a final submission file.
- **submissions/**: Directory to store submission files.
- **requirements.txt**: Python dependencies required to run the scripts.

## How to Use

1. **Ensure outputs are ready:**  
   From the tracking repo, produce an output file (e.g., `submissions/tracking_results{last_number}.csv`).  
   From the face reid repo, produce an output file (e.g., `face_reid_results.csv`).

2. **Merge Outputs:**  
   Run the `merge_submission.py` script to combine the outputs into the required submission format:
   ```bash
   python merge_submission.py
   ```

## Track Experiments

Each experiment’s configuration and results should be stored in the experiments/ folder (see sample file `exp_001.json` for format).

## Experiment Logging Format

Each experiment file (e.g., `exp_001.json`) should include:

```json
{
  "experiment_id": "exp_001",
  "date": "2025-02-25",
  "tracking_model": "ModelA",
  "face_reid_model": "ModelX",
  "public_score": 0.82,
  "notes": "Baseline experiment with initial hyperparameters."
}
```
