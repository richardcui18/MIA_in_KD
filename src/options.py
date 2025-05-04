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
        self.parser.add_argument('--target_model', type=str, default="google-bert/bert-base-uncased", help="the model to attack")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument("--num_data_points", type=int, default=-1, help="subset size of test dataset")
        self.parser.add_argument('--dataset', type=str, help="dataset name")
        self.parser.add_argument('--sub_dataset', type=int, default=128, help="the length of the input text to evaluate (for wikimia). Choose from 32, 64, 128, 256")
        self.parser.add_argument('--num_shots', type=str, default="2", help="number of shots to evaluate (for recall).")
        self.parser.add_argument('--pass_window', type=bool, default=True, help="whether to pass the window to the model.")
        self.parser.add_argument("--synehtic_prefix", type=bool, default=False, help="whether to use synehtic prefix.")
        self.parser.add_argument("--api_key_path", type=str, default=None, help="path to the api key file for OpenAI API if using synehtic prefix.")
        self.parser.add_argument("--vulnerable_file_path", type=str, default=None, help="path to file with vulnerable data.")
        self.parser.add_argument("--extra_step", type=str, default="none", help="extra post-distillation privacy distillation step. Choose from 'none', 'red_list', and 'temp'.")