# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
# TODO: use sns package (fancy plot)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import get_analyzer, find_files_recursive
from arguments import parser

args = parser.parse_args()

if args.use_sns:
    try:
        import seaborn as sns
        sns.set()
    except ImportError as e:
        print(e)

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
