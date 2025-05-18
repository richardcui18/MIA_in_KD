# On Membership Inference Attacks in Knowledge Distillation

## Overview
This is the official repository for On Membership Inference Attacks in Knowledge Distillation.

## Installation
   ```bash
   pip install -r requirements.txt
   ```

## Running MIA
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

If you want to pass `"red_list"` or `"temp"` for `--post_distillation_step`, you will need an additional argument `--vulnerable_file_path`, which include the vulnerable subset that the model penalizes on. Details on generating the vulnerable and non-vulnerable subsets can be found in the section below.

For example, to run MIA on BERT using Bookcorpus dataset without any post-distillation privacy-preserving distillation steps, you can use:
```bash
python run.py \
    --target_model "google-bert/bert-base-uncased" \
    --output_dir ./out \
    --dataset "bookcorpus" \
    --num_data_points 100000 \
    --post_distillation_step "none"
```

## Generating Vulnerable and Non-vulnerable Subsets
Given MIA result for a teacher model, you can obtain vulnerable and non-vulnerable subsets of the training dataset, as described in the paper. The following command will generate the vulnerable subset, non-vulnerable subset, vulnerable subset plus 5% of non-vulnerable subset, and vulnerable subset plus 10% of non-vulnerable subset:
```bash
python get_vulnerable_data.py \
    --target_model <TARGET_MODEL_NAME> \
    --output_dir <OUTPUT_PATH> \
    --dataset <DATASET> \
    --teacher_model_testing_results_path <TEACHER_MODEL_TESTING_PATH>
```

Here is an example command on the Bookcorpus dataset using MIA result on BERT:
```bash
python get_vulnerable_data.py \
    --target_model "BERT" \
    --output_dir ./out \
    --dataset "bookcorpus" \
    --teacher_model_testing_results_path "out/bookcorpus/bert-base-uncased/128/2_shot_128_limit_100000_each.csv"
```

After obtaining the vulnerable subset, you can use it for the red list sampling and temperature scaling methods for privacy-preserving distillation. An example command to run MIA on DistilBERT using Bookcorpus dataset with red list sampling is:
```bash
python run.py \
    --target_model "distilbert/distilbert-base-uncased" \
    --output_dir ./out \
    --dataset "bookcorpus" \
    --num_data_points 100000 \
    --vulnerable_file_path "out/vulnerable_subsets/bookcorpus/loss/vulnerable.txt" \
    --post_distillation_step "red_list"
```

## Analysis
To analyze the testing results for one teacher-student pair, you can use the following command:

```bash
python analysis.py \
    --teacher_model_name <TEACHER_MODEL_NAME> \
    --teacher_model_testing_results_path <TEACHER_MODEL_TESTING_PATH> \
    --student_model_names <STUDENT_MODEL_NAMES> \
    --student_model_testing_results_paths <STUDENT_MODEL_TESTING_PATHS> \
    --post_distillation_step <POST_DISTILLATION_STEP>
```

An example command on BERT as teacher model and DistilBERT as student model is:
```bash
python analysis.py \
    --teacher_model_name "BERT" \
    --teacher_model_testing_results_path "out/bookcorpus/bert-base-uncased/128/2_shot_128_limit_100000_each.csv" \
    --student_model_names "DistilBERT" \
    --student_model_testing_results_paths "out/bookcorpus/distilbert-base-uncased/128/2_shot_128_limit_100000_each.csv" \
    --post_distillation_step "none"
```

## Analysis with Ensemble
To ensemble multiple student models or ensemble multiple privacy-preserving distillation methods (in which case the original student model should be passed in the "teacher" fields and the student models distilled from privacy-preserving distillation methods should be passed in the "student" fields), you can use the following command for ensemble analysis:

```bash
python analysis_ensemble.py \
    --teacher_model_name <TEACHER_MODEL_NAME> \
    --teacher_model_testing_results_path <TEACHER_MODEL_TESTING_PATH> \
    --student_model_names <STUDENT_MODEL_NAMES> \
    --student_model_testing_results_paths <STUDENT_MODEL_TESTING_PATHS>
```

An example command ensembling MIA results from DistilBERT and TinyBERT is:
```bash
python analysis_ensemble.py \
    --teacher_model_name "BERT" \
    --teacher_model_testing_results_path "out/bookcorpus/bert-base-uncased/128/2_shot_128_limit_100000_each.csv" \
    --student_model_names "DistilBERT" "TinyBERT" \
    --student_model_testing_results_paths "out/bookcorpus/distilbert-base-uncased/128/2_shot_128_limit_100000_each.csv" "out/bookcorpus/TinyBERT_General_4L_312D/128/2_shot_128_limit_100000_each.csv"
```