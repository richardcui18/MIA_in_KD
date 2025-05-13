import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
import sys
import json
from analysis_options import Options
import random
import hashlib


def read_tables_from_file_one_model(input_path):
    """
    Read results tables from CSV file and reconstruct the original data structure
    """
    csv.field_size_limit(sys.maxsize)
    all_output = {}
    optimal_thresholds_original = {}
    optimal_thresholds_ensembled = {}
    current_metric = None

    with open(input_path, "r", newline="") as f:
        reader = csv.reader(f)
        
        for row in reader:
            if not row:
                continue  # Skip empty rows
            
            if row[0].startswith("Table for Metric:"):
                # Extract metric and thresholds
                metric = row[0].split("Metric: ")[1]
                current_metric = metric
                
            elif row[0] == "Data Point":
                # Skip header row
                continue
                
            else:
                # Extract data point information
                input_key = row[0]
                label = int(row[1])
                pred_value = float(row[2])
                
                # Check if this data point already exists in all_output
                if input_key in all_output:
                    # Update existing data point with new metric information
                    all_output[input_key]["pred"][current_metric] = pred_value
                else:
                    # Add new data point to all_output
                    if 'bert' in input_path.lower():
                        all_output[input_key] = {
                        "input": input_key,
                        "label": label,
                        "pred": {current_metric: pred_value, 'recall': 0}
                        }
                    else:
                        all_output[input_key] = {
                            "input": input_key,
                            "label": label,
                            "pred": {current_metric: pred_value}
                        }
    
    # Convert the dictionary back to a list
    all_output_list = list(all_output.values())

    print("Done importing, start calculating threshold...")

    bert_in_model = 'bert' in input_path.lower()

    return classify_data_points(all_output_list, bert_in_model)


def calculate_optimal_threshold(labels, scores, seed=42):
    """
    Calculate the optimal threshold by first selecting the best threshold based on a metric,
    and if it doesn't satisfy the 20% class proportion requirement, randomly sample thresholds
    until a valid one is found.
    """
    np.random.seed(seed)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    valid_thresholds = np.isfinite(thresholds)
    if not np.any(valid_thresholds):
        return np.median(scores)
    
    fpr = fpr[valid_thresholds]
    tpr = tpr[valid_thresholds]
    thresholds = thresholds[valid_thresholds]
    
    # Calculate the metric: 1 - (FPR + (1 - TPR)) / 2
    metric = 1 - (fpr + (1 - tpr)) / 2
    
    # Select the threshold with the highest metric
    optimal_idx = np.argmax(metric)
    optimal_threshold = thresholds[optimal_idx]
    
    # Check if the optimal threshold satisfies the 20% class proportion requirement
    y_pred = (scores >= optimal_threshold).astype(int)
    class_proportions = np.bincount(y_pred, minlength=2) / len(y_pred)
    
    if np.all(class_proportions >= 0.2):
        return optimal_threshold
    else:
        # randomly sample thresholds until a valid one is found
        tries = 0
        while tries <= 30:
            random_idx = np.random.randint(0, len(thresholds))
            threshold = thresholds[random_idx]
            
            y_pred = (scores >= threshold).astype(int)
            
            class_proportions = np.bincount(y_pred, minlength=2) / len(y_pred)
            
            if np.all(class_proportions >= 0.2):
                return threshold
            
            tries += 1
        return threshold

def classify_data_points(all_output, bert_in_model):
    metrics = all_output[0]["pred"].keys()

    # calculate the optimal threshold for each metric
    optimal_thresholds = {}
    for metric in metrics:
        if bert_in_model and metric == 'recall':
            optimal_thresholds[metric] = 0
        else:
            for ex in all_output:
                if np.isnan(ex["pred"][metric]).all():
                    ex["pred"][metric] = 0
            
            scores = [ex["pred"][metric] for ex in all_output]
            labels = [ex["label"] for ex in all_output]
            optimal_thresholds[metric] = calculate_optimal_threshold(labels, scores)

    # calculate predicted labels to each example based on all metrics
    for ex in all_output:
        ex["predicted_label"] = {}
        for metric in metrics:
            if ex["pred"][metric] >= optimal_thresholds[metric]:
                ex["predicted_label"][metric] = 1
            else:
                ex["predicted_label"][metric] = 0

    return all_output, optimal_thresholds

def calculate_metrics(all_output):
    # Initialize lists to store true labels and predictions
    true_labels = []
    original_recall_preds = []
    original_ll_preds = []
    original_zlib_preds = []
    ensembled_recall_preds = []
    ensembled_ll_preds = []
    ensembled_zlib_preds = []

    # Single loop to extract data
    for ex in all_output:
        true_labels.append(int(ex['label']))
        original_recall_preds.append(int(ex['predicted_label_original']['recall']))
        original_ll_preds.append(int(ex['predicted_label_original']['ll']))
        original_zlib_preds.append(int(ex['predicted_label_original']['zlib']))
        ensembled_recall_preds.append(int(ex['predicted_label_ensembled']['recall']))
        ensembled_ll_preds.append(int(ex['predicted_label_ensembled']['ll']))
        ensembled_zlib_preds.append(int(ex['predicted_label_ensembled']['zlib']))

    def get_accuracy(true, pred):
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tpr + tnr) / 2
        
        return acc

    # Calculate accuracy
    original_recall_acc = get_accuracy(true_labels, original_recall_preds)
    original_ll_acc = get_accuracy(true_labels, original_ll_preds)
    original_zlib_acc = get_accuracy(true_labels, original_zlib_preds)
    ensembled_recall_acc = get_accuracy(true_labels, ensembled_recall_preds)
    ensembled_ll_acc = get_accuracy(true_labels, ensembled_ll_preds)
    ensembled_zlib_acc = get_accuracy(true_labels, ensembled_zlib_preds)

    original_accuracy = [original_recall_acc, original_ll_acc, original_zlib_acc]
    ensembled_accuracy = [ensembled_recall_acc, ensembled_ll_acc, ensembled_zlib_acc]

    prediction_summary = [true_labels, original_recall_preds, original_ll_preds, original_zlib_preds, ensembled_recall_preds, ensembled_ll_preds, ensembled_zlib_preds]

    return original_accuracy, ensembled_accuracy, prediction_summary

def plot_metrics(original_accuracy, ensembled_accuracy, model_name):
    if 'BERT' in model_name:
        labels = ['Log Likelihood', 'Zlib']
        original_accuracy = original_accuracy[1:]
        ensembled_accuracy = ensembled_accuracy[1:]
        x = range(len(labels))
    else:
        labels = ['ReCall', 'Log Likelihood', 'Zlib']
        x = range(len(labels))
    bar_width = 0.35

    os.makedirs('../fig/ensemble/' + model_name, exist_ok=True)

    accuracy_data = {
        "original_model": model_name,
        "ensembled_model": model_name+"_ensemble",
        "metrics": []
    }
    
    for i, label in enumerate(labels):
        accuracy_data["metrics"].append({
            "metric": label,
            "original_accuracy": round(float(original_accuracy[i]),3),
            "ensembled_accuracy": round(float(ensembled_accuracy[i]),3)
        })

    json_output_path = os.path.join('../fig/ensemble', model_name, 'accuracy_values.json')
    print("Accuracy values json file to", json_output_path)
    with open(json_output_path, 'w') as f:
        json.dump(accuracy_data, f, indent=4)


def plot_distribution_matrix(all_output, metric, model_name):
    matrix_distribution = {
        'original_1': {'ensembled_1': 0, 'ensembled_0': 0},
        'original_0': {'ensembled_1': 0, 'ensembled_0': 0}
    }

    for ex in all_output:
        label = int(ex['label'])
        pred_original = int(ex['predicted_label_original'][metric])
        pred_ensembled = int(ex['predicted_label_ensembled'][metric])

        if pred_original == 1 and pred_ensembled == 1:
            matrix_distribution['original_1']['ensembled_1'] += 1
        elif pred_original == 0 and pred_ensembled == 1:
            matrix_distribution['original_0']['ensembled_1'] += 1
        elif pred_original == 1 and pred_ensembled == 0:
            matrix_distribution['original_1']['ensembled_0'] += 1
        else:
            matrix_distribution['original_0']['ensembled_0'] += 1

    total = len(all_output)
    
    data = [
        [f"Proportion: {(matrix_distribution['original_1']['ensembled_1']/total):.3f}",
         f"Proportion: {(matrix_distribution['original_1']['ensembled_0']/total):.3f}"],
        [f"Proportion: {(matrix_distribution['original_0']['ensembled_1']/total):.3f}",
         f"Proportion: {(matrix_distribution['original_0']['ensembled_0']/total):.3f}"]
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=data,
        colLabels=['MIA predict \nMember \non ensembled Model', 'MIA predict \nNon-member \non ensembled Model'],
        rowLabels=['MIA predict \nMember \non original Model', 'MIA predict \nNon-member \non original Model'],
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.3]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 2)

    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_facecolor('seagreen')
            cell.set_text_props(color='white', weight='bold', verticalalignment='center', horizontalalignment='center')
        else:
            cell.set_facecolor('#F1F8E9')
        
        cell.set_height(0.24)

    caption_text = "Each cell shows the proportion of training data in each category"
    ax.text(0.35, 0.05, caption_text, ha='center', va='center', fontsize=14, transform=ax.transAxes)
    
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    plt.title(f"{model_name} Prediction Class Distribution Comparison\nusing {metric.upper()} with Ensemble", fontsize=16, y=0.92, x=0.35)

    os.makedirs(os.path.join('../fig/ensemble', model_name), exist_ok=True)
    plt.savefig(os.path.join('../fig/ensemble', model_name, f'{metric}_distribution_comparison_matrix.png'))
    plt.close()

def plot_performance_matrix(all_output, metric, model_name):
    matrix_performance = {
        'both_correct': {'positive': 0, 'negative': 0, 'texts': []},
        'original_correct_ensembled_wrong': {'positive': 0, 'negative': 0, 'texts': []},
        'original_wrong_ensembled_correct': {'positive': 0, 'negative': 0, 'texts': []},
        'both_wrong': {'positive': 0, 'negative': 0, 'texts': []}
    }

    for ex in all_output:
        label = int(ex['label'])
        pred_original = int(ex['predicted_label_original'][metric])
        pred_ensembled = int(ex['predicted_label_ensembled'][metric])

        if label == 1:
            gt_category = 'positive'
        else:
            gt_category = 'negative'

        # Compare predictions and update the matrix
        if pred_original == label and pred_ensembled == label:
            matrix_performance['both_correct'][gt_category] += 1
            matrix_performance['both_correct']['texts'].append(ex['input'])
        elif pred_original == label and pred_ensembled != label:
            matrix_performance['original_correct_ensembled_wrong'][gt_category] += 1
            matrix_performance['original_correct_ensembled_wrong']['texts'].append(ex['input'])
        elif pred_original != label and pred_ensembled == label:
            matrix_performance['original_wrong_ensembled_correct'][gt_category] += 1
            matrix_performance['original_wrong_ensembled_correct']['texts'].append(ex['input'])
        else:
            matrix_performance['both_wrong'][gt_category] += 1
            matrix_performance['both_wrong']['texts'].append(ex['input'])

    total = len(all_output)

    proportions = {
        'both_correct': {
            'total': (matrix_performance['both_correct']['positive'] + matrix_performance['both_correct']['negative']) / total,
            'positive': matrix_performance['both_correct']['positive'] / total,
            'negative': matrix_performance['both_correct']['negative'] / total
        },
        'original_correct_ensembled_wrong': {
            'total': (matrix_performance['original_correct_ensembled_wrong']['positive'] + matrix_performance['original_correct_ensembled_wrong']['negative']) / total,
            'positive': matrix_performance['original_correct_ensembled_wrong']['positive'] / total,
            'negative': matrix_performance['original_correct_ensembled_wrong']['negative'] / total
        },
        'original_wrong_ensembled_correct': {
            'total': (matrix_performance['original_wrong_ensembled_correct']['positive'] + matrix_performance['original_wrong_ensembled_correct']['negative']) / total,
            'positive': matrix_performance['original_wrong_ensembled_correct']['positive'] / total,
            'negative': matrix_performance['original_wrong_ensembled_correct']['negative'] / total
        },
        'both_wrong': {
            'total': (matrix_performance['both_wrong']['positive'] + matrix_performance['both_wrong']['negative']) / total,
            'positive': matrix_performance['both_wrong']['positive'] / total,
            'negative': matrix_performance['both_wrong']['negative'] / total
        }
    }

    # Save proportions to a JSON file
    os.makedirs(os.path.join('../fig/ensemble', model_name), exist_ok=True)
    json_file_path = os.path.join('../fig/ensemble', model_name, f'{metric}_attack_performance_proportions.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(proportions, json_file, indent=4)

    # Visualize result
    data = [
        [f"{proportions['both_correct']['total']:.3f}\n({proportions['both_correct']['positive']:.3f}, {proportions['both_correct']['negative']:.3f})",
         f"{proportions['original_correct_ensembled_wrong']['total']:.3f}\n({proportions['original_correct_ensembled_wrong']['positive']:.3f}, {proportions['original_correct_ensembled_wrong']['negative']:.3f})"],
        [f"{proportions['original_wrong_ensembled_correct']['total']:.3f}\n({proportions['original_wrong_ensembled_correct']['positive']:.3f}, {proportions['original_wrong_ensembled_correct']['negative']:.3f})",
         f"{proportions['both_wrong']['total']:.3f}\n({proportions['both_wrong']['positive']:.3f}, {proportions['both_wrong']['negative']:.3f})"]
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=data,
        colLabels=['MIA Success on \nensembled Model', 'MIA Fail on \nensembled Model'],
        rowLabels=['MIA Success on \noriginal Model', 'MIA Fail on \noriginal Model'],
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.3]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 2)

    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_facecolor('seagreen')
            cell.set_text_props(color='white', weight='bold', verticalalignment='center', horizontalalignment='center')
        else:
            cell.set_facecolor('#F1F8E9')
    
    for key, cell in table.get_celld().items():
        cell.set_height(0.2)
    
    caption_text = "Each cell shows the proportion of training data \nin each category, with parenthesis \n(propertion of member data, proportion of non-member data)"
    ax.text(0.35, 0.08, caption_text, ha='center', va='center', fontsize=14, transform=ax.transAxes)
    
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    plt.title(f"{metric.upper()} Attack Performance Comparison on {model_name} with Ensemble", fontsize=16, y=0.85, x=0.35)

    os.makedirs(os.path.join('../fig/ensemble', model_name), exist_ok=True)
    plt.savefig(os.path.join('../fig/ensemble', model_name, f'{metric}_attack_performance_comparison_matrix.png'))
    plt.close()


def read_performance_proportions(model_name, metric):
    file_path = os.path.join('../fig/ensemble', model_name, f'{metric}_attack_performance_proportions.json')
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8-sig') as json_file:
        proportions = json.load(json_file)
    return proportions

def do_ensemble(original_model_name, original_ensemble_model_map, input_path_dict):
    # random number generator for draws among even number of ensembled models
    seed = int(hashlib.sha256(original_model_name.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    all_output_original, optimal_thresholds_original = read_tables_from_file_one_model(input_path_dict[original_model_name])
    optimal_thresholds_ensembles = {}
    all_output_ensembles = {}

    for ensemble_model_name in original_ensemble_model_map[original_model_name]:
        all_output_ensemble, optimal_thresholds_ensemble = read_tables_from_file_one_model(input_path_dict[ensemble_model_name])
        optimal_thresholds_ensembles[ensemble_model_name] = optimal_thresholds_ensemble
        all_output_ensembles[ensemble_model_name] = all_output_ensemble

    pairwise_agreement_counts = {}
    ensemble_models = list(original_ensemble_model_map[original_model_name])
    num_ensembles = len(ensemble_models)

    metrics = list(all_output_original[0]['pred'].keys())

    for metric in metrics:
        pairwise_agreement_counts[metric] = {}
        for i in range(num_ensembles):
            pairwise_agreement_counts[metric][('original', i)] = 0
        for i in range(num_ensembles):
            for j in range(i + 1, num_ensembles):
                pairwise_agreement_counts[metric][(i, j)] = 0

    all_output_ensembled = []
    for i, ex in enumerate(all_output_original):
        pred_stud_label_ex = {}
        pred_stud_ex = {}

        ensemble_predictions = {}
        for metric in metrics:
            ensemble_predictions[metric] = []
            for ensemble_model_name in ensemble_models:
                all_output_ensemble = all_output_ensembles[ensemble_model_name]
                if all_output_ensemble[i]['input'] == ex['input']:
                    ensemble_predictions[metric].append(all_output_ensemble[i]['predicted_label'][metric])
                else:
                    print("ERROR: Input text mismatch")
                    
        for metric in metrics:
            # Count original-ensemble agreements
            for ensemble_idx in range(num_ensembles):
                if ex['predicted_label'][metric] == ensemble_predictions[metric][ensemble_idx]:
                    pairwise_agreement_counts[metric][('original', ensemble_idx)] += 1

        for metric in metrics:
            for idx_i in range(num_ensembles):
                for idx_j in range(idx_i + 1, num_ensembles):
                    if ensemble_predictions[metric][idx_i] == ensemble_predictions[metric][idx_j]:
                        pairwise_agreement_counts[metric][(idx_i, idx_j)] += 1


        # Loop through each metric
        for metric in ex['pred'].keys():
            pos_vote_ensembles = 0
            neg_vote_ensembles = 0
            pred_score_ensembles = 0

            for ensemble_model_name in original_ensemble_model_map[original_model_name]:
                all_output_ensemble = all_output_ensembles[ensemble_model_name]
                
                if all_output_ensemble[i]['input'] == ex['input'] or "Yes, a geometric explanation exists. One such explanation is related to the derivative" in ex['input']:
                    if all_output_ensemble[i]['predicted_label'][metric] == 0:
                        neg_vote_ensembles += 1
                    else:
                        pos_vote_ensembles += 1
                    pred_score_ensembles += all_output_ensemble[i]['pred'][metric]
                
                else:
                    print("ERROR: Input text mismatch")
            
            if pos_vote_ensembles > neg_vote_ensembles:
                pred_stud_label_ex[metric] = 1
            elif neg_vote_ensembles > pos_vote_ensembles:
                pred_stud_label_ex[metric] = 0
            else:
                # draw among even number of ensembles -- random guessing
                input_bytes = ex['input'].encode('utf-8')
                combined = original_model_name.encode('utf-8') + input_bytes
                tie_seed = int(hashlib.sha256(combined).hexdigest(), 16) % (2**32)
                tie_rng = random.Random(tie_seed)
                pred_stud_label_ex[metric] = tie_rng.choice([0, 1])
            
            pred_stud_ex[metric] = pred_score_ensembles / len(ex['pred'])
        
        ex_ensemble = {}
        ex_ensemble['input'] = ex['input']
        ex_ensemble['label'] = ex['label']
        ex_ensemble['pred_original'] = ex['pred']
        ex_ensemble['predicted_label_original'] = ex['predicted_label']
        ex_ensemble['pred_ensembled'] = pred_stud_ex
        ex_ensemble['predicted_label_ensembled'] = pred_stud_label_ex

        all_output_ensembled.append(ex_ensemble)

    return all_output_ensembled


if __name__ == "__main__":
    args = Options()
    args = args.parser.parse_args()
    
    original_model_name = args.teacher_model_name
    original_model_testing_results_path = args.teacher_model_testing_results_path
    ensembled_model_names = args.student_model_names
    ensembled_model_testing_results_paths = args.student_model_testing_results_paths

    original_ensemble_model_map = {original_model_name: ensembled_model_names}
    input_path_dict = {
        original_model_name: original_model_testing_results_path
    }
    for i, model in enumerate(ensembled_model_names):
        input_path_dict[model] = ensembled_model_testing_results_paths[i]
    
    all_output = do_ensemble(original_model_name, original_ensemble_model_map, input_path_dict)

    original_accuracy, ensembled_accuracy, prediction_summary = calculate_metrics(all_output)
    plot_metrics(original_accuracy, ensembled_accuracy, original_model_name)

    for metric in ['recall', 'll', 'zlib']:
        if metric == 'recall' and original_model_name == "BERT":
            continue
        plot_distribution_matrix(all_output, metric, original_model_name)
        print("Finish distribution matrix plotting...")
        plot_performance_matrix(all_output, metric, original_model_name)
        print("Finish performance matrix plotting...")