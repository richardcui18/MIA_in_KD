import analysis_ensemble
from options import Options
import os
import random

def export_filtered_texts(input_path, target_model, output_dir, dataset):
    data_points = analysis_ensemble.read_tables_from_file_one_model(input_path)[0]

    if target_model.lower() == "bert":
        new_text_vulnerable = {
            "loss": [],
            "zlib": []
        }
        new_text_not_vulnerable = {
            "loss": [],
            "zlib": []
        }
    else:
        new_text_vulnerable = {
            "recall": [],
            "loss": [],
            "zlib": []
        }
        new_text_not_vulnerable = {
            "recall": [],
            "loss": [],
            "zlib": []
        }
    
    for point in data_points:
        for metric in new_text_vulnerable.keys():
            # Vulnerable data point
            if point['predicted_label'][metric] == point['label'] and point['label'] == 1:
                input_text = point['input']
                if input_text:
                    new_text_vulnerable[metric].append(input_text)
            # Not vulnerable data point
            elif point['predicted_label'][metric] != point['label'] and point['label'] == 1:
                input_text = point['input']
                if input_text:
                    new_text_not_vulnerable[metric].append(input_text)
    
    # Write to files with empty lines between texts
    output_prefix = os.path.join(output_dir, "vulnerable_subsets", dataset)

    for metric in new_text_vulnerable.keys():
        metric_dir = os.path.join(output_prefix, metric)
        os.makedirs(metric_dir, exist_ok=True)

    if target_model.lower() != "bert":
        with open(f"{output_prefix}/recall/not_vulnerable.txt", "w") as f:
            f.write("\n\n".join(new_text_not_vulnerable['recall']) + "\n")
        with open(f"{output_prefix}/recall/vulnerable.txt", "w") as f:
            f.write("\n\n".join(new_text_vulnerable['recall']) + "\n")
    
    with open(f"{output_prefix}/loss/not_vulnerable.txt", "w") as f:
        f.write("\n\n".join(new_text_not_vulnerable['loss']) + "\n")
    
    with open(f"{output_prefix}/zlib/not_vulnerable.txt", "w") as f:
        f.write("\n\n".join(new_text_not_vulnerable['zlib']) + "\n")
    
    with open(f"{output_prefix}/loss/vulnerable.txt", "w") as f:
        f.write("\n\n".join(new_text_vulnerable['loss']) + "\n")
    
    with open(f"{output_prefix}/zlib/vulnerable.txt", "w") as f:
        f.write("\n\n".join(new_text_vulnerable['zlib']) + "\n")

    # Get vulnerable plus 5% and 10% non-vulnerable data
    for metric in new_text_vulnerable.keys():
        vulnerable = new_text_vulnerable[metric]
        non_vulnerable = new_text_not_vulnerable[metric]
        
        # Calculate 5% and 10% of vulnerable count
        non_vulnerable_count = len(non_vulnerable)
        five_percent_count = int(non_vulnerable_count * 0.05)
        ten_percent_count = int(non_vulnerable_count * 0.10)
        
        random.seed(42)
        five_percent_non_vuln = non_vulnerable[:five_percent_count]
        ten_percent_non_vuln = non_vulnerable[:ten_percent_count]
        
        mixed_5_percent = vulnerable + five_percent_non_vuln
        mixed_10_percent = vulnerable + ten_percent_non_vuln
        
        # Shuffle the mixed datasets
        random.shuffle(mixed_5_percent)
        random.shuffle(mixed_10_percent)
        
        # Write mixed datasets to files
        with open(f"{output_prefix}/{metric}/vulnerable_plus_5percent.txt", "w") as f:
            f.write("\n\n".join(mixed_5_percent) + "\n")
        
        with open(f"{output_prefix}/{metric}/vulnerable_plus_10percent.txt", "w") as f:
            f.write("\n\n".join(mixed_10_percent) + "\n")

    print("Vulnerable and Non-vulnerable subsets saved to folder:", output_prefix)

if __name__ == "__main__":
    args = Options()
    args = args.parser.parse_args()

    output_dir = args.output_dir
    dataset = args.dataset
    target_model = args.target_model
    teacher_model_testing_results_path = args.teacher_model_testing_results_path

    export_filtered_texts(teacher_model_testing_results_path, target_model, output_dir, dataset)