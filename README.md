# Membership Inference Attacks in Knowledge Distillation

## Overview
This is the official repository for Membership Inference Attacks in Knowledge Distillation.

## Installation
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run MIA for a specific model using the following command:

```bash
cd src
python run.py --target_model <TARGET_MODEL> --output_dir <OUTPUT_PATH> --dataset <DATASET>
```

Example:
```bash
python run.py --target_model "google-bert/bert-base-uncased" -output_dir ./out --dataset "bookcorpus"
```