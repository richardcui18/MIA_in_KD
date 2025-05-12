import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import sys
import json
from analysis_options import Options
import analysis_ensemble

def read_tables_from_file(input_path):
    """
    Read results tables from CSV file and reconstruct the original data structure
    """
    csv.field_size_limit(sys.maxsize)
    all_output = {}
    current_metric = None

    with open(input_path, "r", newline="") as f:
        reader = csv.reader(f)
        
        for row in reader:
            if not row:
                continue
            
            if row[0].startswith("Table for Metric:"):
                # Extract metric and thresholds
                metric = row[0].split("Metric: ")[1]
                current_metric = metric
                
            elif row[0] == "Data Point":
                continue
                
            else:
                # Extract data point information
                input_key = row[0]
                label = int(row[1])
                pred_teacher_value = float(row[2])
                pred_student_value = float(row[3])
                
                # Check if this data point already exists in all_output
                if input_key in all_output:
                    existing_data_point = all_output[input_key]
                    existing_data_point["pred_teacher"][current_metric] = pred_teacher_value
                    existing_data_point["pred_student"][current_metric] = pred_student_value
                else:
                    # Add new data point to all_output
                    if 'bert' in input_path.lower():
                        all_output[input_key] = {
                        "input": input_key,
                        "label": label,
                        "pred_teacher": {current_metric: pred_teacher_value, 'recall': 0},
                        "pred_student": {current_metric: pred_student_value, 'recall': 0}
                        }
                    else:
                        all_output[input_key] = {
                            "input": input_key,
                            "label": label,
                            "pred_teacher": {current_metric: pred_teacher_value},
                            "pred_student": {current_metric: pred_student_value}
                        }
    
    all_output_list = list(all_output.values())

    bert_in_model = 'bert' in input_path.lower()
    
    print("Done importing, start calculating threshold...")
    return classify_data_points(all_output_list, bert_in_model)

def read_tables_from_multiple_files(input_path_teacher, input_path_student):
    all_output_teacher, optimal_thresholds_teacher = analysis_ensemble.read_tables_from_file_one_model(input_path_teacher)
    all_output_student, optimal_thresholds_student = analysis_ensemble.read_tables_from_file_one_model(input_path_student)

    all_output = []
    for i, ex in enumerate(all_output_teacher):
        pred_stud_label_ex = {}
        pred_stud_ex = {}
        
        # Loop through each metric
        for metric in ex['pred'].keys():
            if all_output_student[i]['input'] == ex['input']:
                pred_stud_label_ex[metric] = int(all_output_student[i]['predicted_label'][metric])
                pred_stud_ex[metric] = float(all_output_student[i]['pred'][metric])
            else:
                print("ERROR: Input text mismatch")
        
        ex_teacher_student = {}
        ex_teacher_student['input'] = ex['input']
        ex_teacher_student['label'] = ex['label']
        ex_teacher_student['pred_teacher'] = ex['pred']
        ex_teacher_student['predicted_label_teacher'] = ex['predicted_label']
        ex_teacher_student['pred_student'] = pred_stud_ex
        ex_teacher_student['predicted_label_student'] = pred_stud_label_ex

        all_output.append(ex_teacher_student)

    return all_output, optimal_thresholds_teacher, optimal_thresholds_student

def classify_data_points(all_output, bert_in_model):
    metrics = all_output[0]["pred_teacher"].keys()

    # calculate the optimal threshold for each metric
    optimal_thresholds_teacher = {}
    optimal_thresholds_student = {}
    for metric in metrics:
        if bert_in_model and metric == 'recall':
            optimal_thresholds_teacher[metric] = 0
            optimal_thresholds_student[metric] = 0
        else:
            for ex in all_output:
                if np.isnan(ex["pred_teacher"][metric]).all():
                    ex["pred_teacher"][metric] = 0
            for ex in all_output:
                if np.isnan(ex["pred_student"][metric]).all():
                    ex["pred_student"][metric] = 0
            
            scores_teacher = [ex["pred_teacher"][metric] for ex in all_output]
            scores_student = [ex["pred_student"][metric] for ex in all_output]
            labels = [ex["label"] for ex in all_output]
            optimal_thresholds_teacher[metric] = analysis_ensemble.calculate_optimal_threshold(labels, scores_teacher)
            optimal_thresholds_student[metric] = analysis_ensemble.calculate_optimal_threshold(labels, scores_student)

    # calculate predicted labels to each example based on all metrics
    for ex in all_output:
        ex["predicted_label_teacher"] = {}
        for metric in metrics:
            if ex["pred_teacher"][metric] >= optimal_thresholds_teacher[metric]:
                ex["predicted_label_teacher"][metric] = 1
            else:
                ex["predicted_label_teacher"][metric] = 0
        
        ex["predicted_label_student"] = {}
        for metric in metrics:
            if ex["pred_student"][metric] >= optimal_thresholds_student[metric]:
                ex["predicted_label_student"][metric] = 1
            else:
                ex["predicted_label_student"][metric] = 0

    return all_output, optimal_thresholds_teacher, optimal_thresholds_student

def calculate_metrics(all_output):
    true_labels = []
    teacher_recall_preds = []
    teacher_ll_preds = []
    teacher_zlib_preds = []
    student_recall_preds = []
    student_ll_preds = []
    student_zlib_preds = []

    for ex in all_output:
        true_labels.append(int(ex['label']))
        teacher_recall_preds.append(int(ex['predicted_label_teacher']['recall']))
        teacher_ll_preds.append(int(ex['predicted_label_teacher']['ll']))
        teacher_zlib_preds.append(int(ex['predicted_label_teacher']['zlib']))
        student_recall_preds.append(int(ex['predicted_label_student']['recall']))
        student_ll_preds.append(int(ex['predicted_label_student']['ll']))
        student_zlib_preds.append(int(ex['predicted_label_student']['zlib']))
    
    def get_accuracy_and_se(true, pred):
        correct = np.array(true) == np.array(pred)
        accuracy = np.mean(correct)
        std = np.std(correct, ddof=1)
        n = len(correct)
        se = std / np.sqrt(n)
        return accuracy, se

    # Calculate accuracy and SE
    teacher_recall_acc, teacher_recall_se = get_accuracy_and_se(true_labels, teacher_recall_preds)
    teacher_ll_acc, teacher_ll_se = get_accuracy_and_se(true_labels, teacher_ll_preds)
    teacher_zlib_acc, teacher_zlib_se = get_accuracy_and_se(true_labels, teacher_zlib_preds)
    student_recall_acc, student_recall_se = get_accuracy_and_se(true_labels, student_recall_preds)
    student_ll_acc, student_ll_se = get_accuracy_and_se(true_labels, student_ll_preds)
    student_zlib_acc, student_zlib_se = get_accuracy_and_se(true_labels, student_zlib_preds)

    teacher_accuracy = [teacher_recall_acc, teacher_ll_acc, teacher_zlib_acc]
    student_accuracy = [student_recall_acc, student_ll_acc, student_zlib_acc]
    teacher_se = [teacher_recall_se, teacher_ll_se, teacher_zlib_se]
    student_se = [student_recall_se, student_ll_se, student_zlib_se]
    
    prediction_summary = [true_labels, teacher_recall_preds, teacher_ll_preds, teacher_zlib_preds, student_recall_preds, student_ll_preds, student_zlib_preds]

    return teacher_accuracy, student_accuracy, teacher_se, student_se, prediction_summary

def plot_metrics(teacher_accuracy, student_accuracy, teacher_se, student_se, teacher_model_name, student_model_name, extra_step):
    if teacher_model_name == 'BERT' or teacher_model_name == 'BERT_not_vulnerable':
        labels = ['Log Likelihood', 'Zlib']
        teacher_accuracy = teacher_accuracy[1:]
        student_accuracy = student_accuracy[1:]
        x = range(len(labels))
    else:
        labels = ['ReCall', 'Log Likelihood', 'Zlib']
        x = range(len(labels))
    bar_width = 0.35

    os.makedirs(os.path.join('../fig/single_student', teacher_model_name, student_model_name), exist_ok=True)

    # Create accuracy output text file
    accuracy_output_path = os.path.join('../fig/single_student', teacher_model_name, student_model_name)
    
    accuracy_data = {
        "teacher_model": teacher_model_name,
        "student_model": student_model_name,
        "metrics": []
    }
    
    for i, label in enumerate(labels):
        accuracy_data["metrics"].append({
            "metric": label,
            "teacher_accuracy": {
                "mean": float(teacher_accuracy[i]),
                "se": float(teacher_se[i]),
                "ci_95": f"({teacher_accuracy[i]-1.96*teacher_se[i]:.4f}, {teacher_accuracy[i]+1.96*teacher_se[i]:.4f})"
            },
            "student_accuracy": {
                "mean": float(student_accuracy[i]),
                "se": float(student_se[i]),
                "ci_95": f"({student_accuracy[i]-1.96*student_se[i]:.4f}, {student_accuracy[i]+1.96*student_se[i]:.4f})"
            }
        })
    
    # Write to JSON file
    json_output_path = os.path.join(accuracy_output_path, f'accuracy_values{extra_step}.json')
    print("Accuracy values json file to", json_output_path)
    with open(json_output_path, 'w') as f:
        if extra_step == "_red_list" or extra_step == "_temp":
            accuracy_data['student_model'] = accuracy_data['student_model']+extra_step
        json.dump(accuracy_data, f, indent=4)


    # Plot for Accuracy
    plt.figure(figsize=(10,7))
    rects1 = plt.bar([i - bar_width / 2 for i in x], teacher_accuracy, bar_width, 
                     yerr=teacher_se, capsize=5, label='Teacher', color='lightblue')
    rects2 = plt.bar([i + bar_width / 2 for i in x], student_accuracy, bar_width, 
                     yerr=student_se, capsize=5, label='Student', color='darkgrey')
    plt.xlabel('Metrics', labelpad=10, fontsize=14)
    plt.ylabel('Accuracy', labelpad=10, fontsize=14)
    if extra_step != "none":
        title_suffix = f" (with{extra_step.lower().replace('_', ' ')})"
    else:
        title_suffix = ""
    plt.title(f'{teacher_model_name}/{student_model_name} Accuracy by Metric and Model\n{title_suffix}', 
              pad=20, fontsize=18)
    plt.xticks(x, labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join('../fig/single_student', teacher_model_name, student_model_name, 
                            f'accuracy_comparison{extra_step}.png'))
    print(f"Accuracy plot to {os.path.join('../fig/single_student', teacher_model_name, student_model_name, f'accuracy_comparison{extra_step}.png')}")
    plt.close()

def plot_distribution_matrix(all_output, metric, teacher_model_name, student_model_name, extra_step):
    matrix_distribution = {
        'teacher_1': {'student_1': 0, 'student_0': 0},
        'teacher_0': {'student_1': 0, 'student_0': 0}
    }

    for ex in all_output:
        label = int(ex['label'])
        pred_teacher = int(ex['predicted_label_teacher'][metric])
        pred_student = int(ex['predicted_label_student'][metric])

        if pred_teacher == 1 and pred_student == 1:
            matrix_distribution['teacher_1']['student_1'] += 1
        elif pred_teacher == 0 and pred_student == 1:
            matrix_distribution['teacher_0']['student_1'] += 1
        elif pred_teacher == 1 and pred_student == 0:
            matrix_distribution['teacher_1']['student_0'] += 1
        else:
            matrix_distribution['teacher_0']['student_0'] += 1

    total = len(all_output)

    data = [
        [f"Proportion: {(matrix_distribution['teacher_1']['student_1']/total):.3f}",
         f"Proportion: {(matrix_distribution['teacher_1']['student_0']/total):.3f}"],
        [f"Proportion: {(matrix_distribution['teacher_0']['student_1']/total):.3f}",
         f"Proportion: {(matrix_distribution['teacher_0']['student_0']/total):.3f}"]
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.axis('off')

    table = ax.table(
        cellText=data,
        colLabels=['MIA predict \nMember \non Student Model', 'MIA predict \nNon-member \non Student Model'],
        rowLabels=['MIA predict \nMember \non Teacher Model', 'MIA predict \nNon-member \non Teacher Model'],
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

    plt.title(f"{teacher_model_name}/{student_model_name} Prediction Class Distribution Comparison using {metric.upper()}", fontsize=16, y=0.92, x=0.35)

    os.makedirs(os.path.join('../fig/single_student', teacher_model_name, student_model_name), exist_ok=True)
    plt.savefig(os.path.join('../fig/single_student', teacher_model_name, student_model_name, f'{metric}_distribution_comparison_matrix{extra_step}.png'))
    plt.close()

def plot_performance_matrix(all_output, metric, teacher_model_name, student_model_name, extra_step):
    matrix_performance = {
        'both_correct': {'positive': 0, 'negative': 0, 'texts': []},
        'teacher_correct_student_wrong': {'positive': 0, 'negative': 0, 'texts': []},
        'teacher_wrong_student_correct': {'positive': 0, 'negative': 0, 'texts': []},
        'both_wrong': {'positive': 0, 'negative': 0, 'texts': []}
    }

    for ex in all_output:
        label = int(ex['label'])
        pred_teacher = int(ex['predicted_label_teacher'][metric])
        pred_student = int(ex['predicted_label_student'][metric])

        if label == 1:
            gt_category = 'positive'
        else:
            gt_category = 'negative'

        # Compare predictions and update the matrix
        if pred_teacher == label and pred_student == label:
            matrix_performance['both_correct'][gt_category] += 1
            matrix_performance['both_correct']['texts'].append(ex['input'])
        elif pred_teacher == label and pred_student != label:
            matrix_performance['teacher_correct_student_wrong'][gt_category] += 1
            matrix_performance['teacher_correct_student_wrong']['texts'].append(ex['input'])
        elif pred_teacher != label and pred_student == label:
            matrix_performance['teacher_wrong_student_correct'][gt_category] += 1
            matrix_performance['teacher_wrong_student_correct']['texts'].append(ex['input'])
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
        'teacher_correct_student_wrong': {
            'total': (matrix_performance['teacher_correct_student_wrong']['positive'] + matrix_performance['teacher_correct_student_wrong']['negative']) / total,
            'positive': matrix_performance['teacher_correct_student_wrong']['positive'] / total,
            'negative': matrix_performance['teacher_correct_student_wrong']['negative'] / total
        },
        'teacher_wrong_student_correct': {
            'total': (matrix_performance['teacher_wrong_student_correct']['positive'] + matrix_performance['teacher_wrong_student_correct']['negative']) / total,
            'positive': matrix_performance['teacher_wrong_student_correct']['positive'] / total,
            'negative': matrix_performance['teacher_wrong_student_correct']['negative'] / total
        },
        'both_wrong': {
            'total': (matrix_performance['both_wrong']['positive'] + matrix_performance['both_wrong']['negative']) / total,
            'positive': matrix_performance['both_wrong']['positive'] / total,
            'negative': matrix_performance['both_wrong']['negative'] / total
        }
    }

    # Save proportions to a JSON file
    os.makedirs(os.path.join('../fig/single_student', teacher_model_name, student_model_name), exist_ok=True)
    json_file_path = os.path.join('../fig/single_student', teacher_model_name, student_model_name, f'{metric}_attack_performance_proportions.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(proportions, json_file, indent=4)


    data = [
        [f"{proportions['both_correct']['total']:.3f}\n({proportions['both_correct']['positive']:.3f}, {proportions['both_correct']['negative']:.3f})",
         f"{proportions['teacher_correct_student_wrong']['total']:.3f}\n({proportions['teacher_correct_student_wrong']['positive']:.3f}, {proportions['teacher_correct_student_wrong']['negative']:.3f})"],
        [f"{proportions['teacher_wrong_student_correct']['total']:.3f}\n({proportions['teacher_wrong_student_correct']['positive']:.3f}, {proportions['teacher_wrong_student_correct']['negative']:.3f})",
         f"{proportions['both_wrong']['total']:.3f}\n({proportions['both_wrong']['positive']:.3f}, {proportions['both_wrong']['negative']:.3f})"]
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=data,
        colLabels=['MIA Success on \nStudent Model', 'MIA Fail on \nStudent Model'],
        rowLabels=['MIA Success on \nTeacher Model', 'MIA Fail on \nTeacher Model'],
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

    plt.title(f"{metric.upper()} Attack Performance Comparison on {teacher_model_name}/{student_model_name}", fontsize=16, y=0.85, x=0.35)

    os.makedirs(os.path.join('../fig/single_student', teacher_model_name, student_model_name), exist_ok=True)
    plt.savefig(os.path.join('../fig/single_student', teacher_model_name, student_model_name, f'{metric}_attack_performance_comparison_matrix{extra_step}.png'))
    plt.close()


def read_performance_proportions(model_name, metric):
    file_path = os.path.join('../fig/single_student', model_name, f'{metric}_attack_performance_proportions.json')
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8-sig') as json_file:
        proportions = json.load(json_file)
    return proportions


if __name__ == "__main__":
    args = Options()
    args = args.parser.parse_args()
    
    teacher_model_name = args.teacher_model_name
    teacher_model_testing_results_path = args.teacher_model_testing_results_path
    student_model_names = args.student_model_names
    student_model_testing_results_paths = args.student_model_testing_results_paths
    num_data_points = args.num_data_points
    extra_step = args.post_distillation_step

    input_path_dict = {
        teacher_model_name: teacher_model_testing_results_path
    }
    for i, model in enumerate(student_model_names):
        input_path_dict[model] = student_model_testing_results_paths[i]

    for student_model_name in student_model_names:
        input_path_student = input_path_dict[student_model_name]
        all_output, optimal_thresholds_teacher, optimal_thresholds_student = read_tables_from_multiple_files(teacher_model_testing_results_path, input_path_student)
        
        teacher_accuracy, student_accuracy, teacher_se, student_se, prediction_summary = calculate_metrics(all_output)
        plot_metrics(teacher_accuracy, student_accuracy, teacher_se, student_se, teacher_model_name, student_model_name, extra_step)
        
        for metric in ['recall', 'll', 'zlib']:
            if metric == 'recall' and teacher_model_name == "BERT":
                continue
            plot_distribution_matrix(all_output, metric, teacher_model_name, student_model_name, extra_step)
            plot_performance_matrix(all_output, metric, teacher_model_name, student_model_name, extra_step)
            