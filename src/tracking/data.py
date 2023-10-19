from __future__ import annotations

import numpy as np
import pandas as pd


class GazeData(object):
    def __init__(self, path):
        self.df = pd.read_csv(path, error_bad_lines=False)

    def get_no_sessions(self, subject_id):
        data = self.df[self.df.iloc[:, 0] == subject_id].to_numpy()
        return int(data.shape[0])

    def get_data(self, subject_id, session_index):
        data = self.df[self.df.iloc[:, 0] == subject_id].to_numpy()
        known = data[session_index, 1]
        sid = np.delete(data[session_index, :], [0, 1])
        x = np.array(sid[::2], dtype=float)
        y = np.array(sid[1::2], dtype=float)
        return (known, x[~np.isnan(x)], y[~np.isnan(y)])
