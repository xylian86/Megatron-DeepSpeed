# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
from abstract_analysis import TensorBoardAnalysis


class UniversalCheckpointingAnalysis(TensorBoardAnalysis):

    def __init__(self):
        self._name = "universal_checkpointing"

    def set_names(self, path_name):
        match = re.match(self.path_regex(), path_name)
        tp = match.group(1)
        pp = match.group(2)
        dp = match.group(3)
        sp = match.group(4)

        self._label_name = f"TP: {tp}, PP: {pp}, DP: {dp}"
        self._csv_name = f"uc_out_tp_{tp}_pp_{pp}_dp_{dp}_sp_{sp}_val_loss"

    def get_label_name(self):
        return self._label_name

    def get_csv_filename(self):
        return self._csv_name

    def path_regex(self):
        return '.*tp(\d+).*pp(\d+).*dp(\d+).*sp(\d+)'
