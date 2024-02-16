# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from uc_analysis import UniversalCheckpointingAnalysis


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
            if root not in matching_paths and filename.lower().startswith(file_affix.lower()):
                matching_paths.append(os.path.join(root))
    return matching_paths

def get_analyzer(analyzer_name):
    if analyzer_name == 'universal_checkpointing':
        analyzer = UniversalCheckpointingAnalysis()
        # TODO: can this work with just the class init call being returned?
        return analyzer
    else:
        raise ValueError(f"Unsupported analyzer {analyzer_name}")
