import os
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

i = 0
for tb_path in tb_log_paths:
    print(f"tb_path: {tb_path}")

    event_accumulator = EventAccumulator(tb_path)
    event_accumulator.Reload()

    events = event_accumulator.Scalars('lm-loss-training/lm loss')

    x = [x.step for x in events]
    y = [x.value for x in events]

    plt.plot(x, y, label=f'UC Training Run {i}')

    df = pd.DataFrame({"step": x, "value": y})
    df.to_csv(f"file{i}.csv")
    i = i + 1
    print(df)

plt.legend()
plt.ylabel("LM Loss")
plt.xlabel("Training Step")
plt.savefig("uni_ckpt_char.png")
print(tb_log_paths)
