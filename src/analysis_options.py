import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--teacher_model_name', type=str, default="BERT", help="teacher model name")
        self.parser.add_argument('--teacher_model_testing_results_path', type=str, default=None, help="path to teacher model testing results")
        self.parser.add_argument(
            '--student_model_names', 
            type=str, 
            nargs='+',
            default=["DistilBERT"], 
            help="List of student model names"
        )
        self.parser.add_argument(
            '--student_model_testing_results_paths', 
            type=str, 
            nargs='+',
            default=None, 
            help="List of path to student model testing results"
        )
        self.parser.add_argument('--num_data_points', type=str, default='10000', help="subset size of testing data")
        self.parser.add_argument("--extra_step", type=str, default="", help="extra post-distillation privacy distillation step. Choose from '', 'red_list', and 'temp'.")