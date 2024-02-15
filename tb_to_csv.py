import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--tb_dir", required=True, type=str, help="Directory for tensorboard output")
#parser.add_argument("--tb_dir_single", required=True, type=str, help="Directory for single tensorboard output")
#parser.add_argument("--tb_dir_full", required=True, type=str, help="Directory for many tensorboard outputs")
args = parser.parse_args()

#def find_files_recursive(directory, file_extension):
#    """
#    Recursively searches for files with a specific extension in a directory.
#
#    Args:
#        directory (str): The path to the directory to search.
#        file_extension (str): The desired file extension (e.g., '.txt', '.jpg').
#
#    Returns:
#        list: A list of absolute paths to matching files.
#    """
#    matching_files = []
#    for root, _, files in os.walk(directory):
#        for filename in files:
#            if filename.lower().startswith(file_extension.lower()):
#                matching_files.append(os.path.join(root, filename))
#    return matching_files
#
log_dir = args.tb_dir
#
#
#
## Example usage:
#directory_path = './'
#target_affix = 'events.out.tfevents'
#result_files = find_files_recursive(directory_path, target_extension)
#
#for filename in result_files:
#    #if '.pt' in filename:
#	#print(f"LOADING CKPT FOR: {filename}")
#	ckpt = torch.load(filename)
#	print(f"filename: {filename}, ckpt.keys(): {ckpt.keys()}")
#
#print(result_files)


event_accumulator = EventAccumulator(log_dir)
event_accumulator.Reload()

events = event_accumulator.Scalars('lm-loss-training/lm loss')

x = [x.step for x in events]
y = [x.value for x in events]


df = pd.DataFrame({"step": x, "value": y})
df.to_csv("train_loss_uc.csv")
print(df)
