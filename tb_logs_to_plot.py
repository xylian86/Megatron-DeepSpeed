import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--tb_dir", required=True, type=str, help="Directory for tensorboard output")
args = parser.parse_args()

def find_files_recursive(directory, file_affix):
    """
    Recursively searches for files with a specific affix in a directory.

    Args:
        directory (str): The path to the directory to search.
        file_affix (str): The desired file affix

    Returns:
        list: A list of paths to matching files.
    """
    matching_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().startswith(file_affix.lower()):
                matching_paths.append(os.path.join(root))
    return matching_paths

log_dir = args.tb_dir

target_affix = 'events.out.tfevents'
tb_log_paths = find_files_recursive(log_dir, target_affix)

pattern = '.*tp(\d).*pp(\d).*dp(\d).*sp(\d)'

for tb_path in tb_log_paths:
    print(f"tb_path: {tb_path}")
    match = re.match(pattern, tb_path)

    label = f"TP: {match.group(1)}, PP: {match.group(2)}, DP: {match.group(3)}, SP: {match.group(4)}"

    event_accumulator = EventAccumulator(tb_path)
    event_accumulator.Reload()

    events = event_accumulator.Scalars('lm-loss-training/lm loss')

    x = [x.step for x in events]
    y = [x.value for x in events]

    plt.plot(x, y, label=f'Training Run: {label}')

    csv_filename = f"uc_out_tp_{match.group(1)}_pp_{match.group(2)}_dp_{match.group(3)}_sp_{match.group(4)}"

    df = pd.DataFrame({"step": x, "value": y})
    df.to_csv(f"{csv_filename}.csv")
    print(df)

plt.legend()
plt.title('Megatron-GPT Universal Checkpointing')
plt.ylabel("LM Loss")
plt.xlabel("Training Step")
plt.savefig("uni_ckpt_char.png")
print(tb_log_paths)
