# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
# TODO: use sns package (fancy plot)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser
from utils import get_analyzer, find_files_recursive
#import seaborn as sns

#sns.set()

parser = ArgumentParser()
parser.add_argument("--tb_dir", required=True, type=str, help="Directory for tensorboard output")
parser.add_argument("--analyzer", default="universal_checkpointing", type=str, choices=["universal_checkpointing"], help="Specify the analyzer to use")
parser.add_argument("--tb_event_key", required=False, default="lm-loss-training/lm loss", type=str, help="Optional override of the TensorBoard event key")
parser.add_argument("--plot_title", required=False, default="Megatron-GPT Universal Checkpointing", type=str, help="Optional override of the plot title")
parser.add_argument("--plot_x_label", required=False, default="Training Step", type=str, help="Optional override of the plot x-label")
parser.add_argument("--plot_y_label", required=False, default="LM Loss", type=str, help="Optional override of the plot y-label")
parser.add_argument("--plot_name", required=False, default="uni_ckpt_char.png", type=str, help="Optional override of the plot file name")
parser.add_argument("--skip_plot", action='store_true', help="Skip generation of plot file")
parser.add_argument("--skip_csv", action='store_true', help="Skip generation of csv files")
args = parser.parse_args()

log_dir = args.tb_dir

target_affix = 'events.out.tfevents'
tb_log_paths = find_files_recursive(log_dir, target_affix)

analyzer = get_analyzer(args.analyzer)

for tb_path in tb_log_paths:
    print(f"tb_path: {tb_path}")
    analyzer.set_names(tb_path)

    event_accumulator = EventAccumulator(tb_path)
    event_accumulator.Reload()

    events = event_accumulator.Scalars(args.tb_event_key)
    #events = event_accumulator.Scalars('lm-loss-training/lm loss')
    #events = event_accumulator.Scalars('lm-loss-validation/lm loss validation')

    # TODO: make tb key arg to script
    # iteration time
    # validation loss
    # sample/sec (throughput)

    x = [x.step for x in events]
    y = [x.value for x in events]

    plt.plot(x, y, label=f'Training Run: {analyzer.get_label_name()}')

    if not args.skip_csv:
        df = pd.DataFrame({"step": x, "value": y})
        df.to_csv(f"{analyzer.get_csv_filename()}.csv")
        print(df)

if not args.skip_plot:
    plt.legend()
    plt.title(args.plot_title)
    plt.xlabel(args.plot_x_label)
    plt.ylabel(args.plot_y_label)
    plt.savefig(args.plot_name)
print(tb_log_paths)
