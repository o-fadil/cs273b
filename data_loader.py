""" This file loads the UniRep data"""

import numpy as np
import pandas as pd
import json
import os
import errno
from typing import Tuple


class DataLoader(object):
    UNIREP_LEN = 1900
    UNIREP_COL = ["avg_UniRep_" + str(i) for i in range(UNIREP_LEN)]
    SEQ_COL = ["Extended Domain sequence"]
    OH_COL = ["OneHotEnc"]

    def __init__(self, file_name: str):
        """
        initiate dataloader class
        :param file_name: file name to load
        """
        self.file_name = file_name
        csv_file = pd.read_csv(file_name)
        self.y_col = [i for i in csv_file.columns if "Avg" in i][0]
        if "NucAct" in file_name: index_file = "index_BareNucAct.json"
        elif "NucRepr" in file_name: index_file = "index_BareNucRepr.json"
        elif "Tiling" in file_name: index_file = "index_BareTilingRepressors.json"
        else: raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), "index file")
        with open(index_file, "r") as f:
            self.index_dict = json.load(f)
        pass

    # test
    @staticmethod
    def addEntry(info_path: str, info_dict: dict, name: str) -> None:
        """
        Add information entry, not for use now
        :param info_path:
        :param info_dict:
        :param name:
        :return:
        """
        file_dir = os.path.join(info_path, name + ".json")
        with open(file_dir, "w") as f:
            json.dump(info_dict, f)
        return
    
    def onehot_encode(aaseq: str):
        """
        Convert AA sequence to onehot encoding numpy array.
        Args:
            aaseq - str
        Returns:
            onehotarr - (lenseq, 20) np.array
        """
        alphabet = 'GALMFWKQESPVICYHRNDT'
        aa2int = {k: i for i,k in enumerate(alphabet)}
        #int2aa = {i: k for i,k in enumerate(alphabet)}

        lenseq = len(aaseq)
        onehotarr = np.zeros((lenseq, 20), dtype=int)
        for i,aa in enumerate(aaseq):
            onehotarr[i, aa2int[aa]] = 1

        return onehotarr
    
    def getIdx(self):
        return self.index_dict['train_index'], self.index_dict['val_index'], self.index_dict['test_index']

    def loadData(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        load X, and Y
        :return: x, y as numpy array
        """
        csv_file = pd.read_csv(self.file_name)
        return csv_file.loc[:, self.UNIREP_COL].to_numpy(), csv_file.loc[:, self.y_col].to_numpy()

    def loadTrainValTest(self) -> Tuple:
        """
        load Train, Validation set and Test set
        :return: Tuple[x_train, x_val, x_test], Tuple[y_train, y_val, y_test]
        """
        x, y = self.loadData()
        x_train, x_val, x_test = x[self.index_dict['train_index'],:], x[self.index_dict['val_index'],:], x[self.index_dict['test_index'],:]
        y_train, y_val, y_test = y[self.index_dict['train_index']], y[self.index_dict['val_index']], y[self.index_dict['test_index']]
        return (x_train, x_val, x_test), (y_train, y_val, y_test)
    
    def loadSeqData(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        load X, and Y
        :return: x, y as numpy array
        """
        csv_file = pd.read_csv(self.file_name)
        return csv_file.loc[:, self.SEQ_COL].to_numpy(), csv_file.loc[:, self.y_col].to_numpy()

    def loadSeqTrainValTest(self) -> Tuple:
        """
        load Train, Validation set and Test set
        :return: Tuple[x_train, x_val, x_test], Tuple[y_train, y_val, y_test]
        """
        x, y = self.loadSeqData()
        x[self.OH_COL] = x[self.SEQ_COL].apply(onehot_encode)
        
        x_train, x_val, x_test = x[self.index_dict['train_index'],:], x[self.index_dict['val_index'],:], x[self.index_dict['test_index'],:]
        y_train, y_val, y_test = y[self.index_dict['train_index']], y[self.index_dict['val_index']], y[self.index_dict['test_index']]
        return (x_train, x_val, x_test), (y_train, y_val, y_test)


