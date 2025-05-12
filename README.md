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
python run.py \
    --target_model <TARGET_MODEL> \
    --output_dir <OUTPUT_PATH> \
    --dataset <DATASET> \
    --num_data_points <NUM_DATA_POINTS> \
    --post_distillation_step <POST_DISTILLATION_STEP>
```

Example:
```bash
python run.py \
    --target_model "google-bert/bert-base-uncased" \
    --output_dir ./out \
    --dataset "bookcorpus" \
    --num_data_points 100000 \
    --post_distillation_step "none"
```

To analyze the testing results for a teacher-student pair, you can use the following command:

```bash
python analysis.py \
    --teacher_model_name <TEACHER_MODEL> \
    --teacher_model_testing_results_path <TEACHER_MODEL_TESTING_PATH> \
    --student_model_names <STUDENT_MODELS> \
    --student_model_testing_results_paths <STUDENT_MODEL_TESTING_PATHS> \
    --num_data_points <NUM_DATA_POINTS> \
    --post_distillation_step <POST_DISTILLATION_STEP>
```

Example:
```bash
python analysis.py \
    --teacher_model_name "BERT" \
    --teacher_model_testing_results_path "out/bookcorpus/bert-base-uncased/128/2_shot_128.csv" \
    --student_model_names "DistilBERT" \
    --student_model_testing_results_paths "out/bookcorpus/distilbert-base-uncased/128/2_shot_128.csv" \
    --num_data_points "100000" \
    --post_distillation_step ""
```
