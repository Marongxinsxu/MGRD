import math

import numpy as np
import pandas as pd

val_window = 300           # Verification set length
cold_window = 200           # Cold window length
win_size = 100
step = 1

def parse_data(data_dir: str) -> tuple:
    '''
    parse_data:Read training data and test data from the data_dir directory, which must contain the train.csv and test.csv files.

    train.csv The file format is as follows: (First column: sample time Other columns: sample characteristics)
    The first 7*(n+2) examples
    datatime    feature_1   feature_2   feature_3   feature_4   ...     feature_n   labels
        0	    0.4583	        1	        1	        0	    ...     0.6272          0
        1	    0.4572	        1	        1	        0	    ...     0.6272          0
        2	    0.4583	        1	        1	        0	    ...     0.6272          0
        3	    0.4592	        1	        1	        0	    ...     0.6272          0
        4	    0.4605	        1	        1	        0	    ...     0.6272          0
        5	    0.4605	        1	        1	        0	    ...     0.6272          0
        6	    0.4594	        1	        1	        0	    ...     0.6272          0

    test.csvThe file format is as follows：(First column: sample time Last column: exception label (0/1 represents normal point/abnormal point) Remaining columns: sample characteristics)
    The first 7*(n+2) columns
    datatime    feature_1   feature_2   feature_3   feature_4   ...     feature_n   labels
        0	    0.4811	        1	        0	        0	    ...     0.6329          0
        1	    0.4847	        1	        0	        0	    ...     0.6329          1
        2	    0.4879	        1	        0	        0	    ...     0.6329          1
        3	    0.4950	        1	        0	        0	    ...     0.6329          1
        4	    0.5023	        1	        0	        0	    ...     0.6352          1
        5	    0.5055	        1	        0	        0	    ...     0.6352          1
        6	    0.5097	        1	        0	        0	    ...     0.6386          1

    note: Only test.csv has an exception label, train.csv has no label (the overall data is normal)

    :param data_dir:
    :return: （train_data:pd.Frame,test_data:pd.Frame）
    train_data : Rows represent time points and columns represent features
    test_data : The row represents the point in time, the column represents the feature, and the last column is the exception label
    '''

    train_data = pd.read_csv(f'{data_dir}/train.csv', sep=',', index_col=0, low_memory=False)
    test_data = pd.read_csv(f'{data_dir}/test.csv', sep=',', index_col=0, low_memory=False)

    return train_data, test_data

def split(data):
    """
    :param data: DataFrame
    :return: train_set, val_set, test_set
    """
    train_dataset = data[0]
    test_dataset = data[1]
    '''
    train_dataset、test_dataset Example of the first n+1 column (first n column features, last column exception label)
    0.4583	        1	        1	        0	    ...     0.6272          0
    0.4572	        1	        1	        0	    ...     0.6272          0
    0.4583	        1	        1	        0	    ...     0.6272          0
    0.4592	        1	        1	        0	    ...     0.6272          0
    0.4605	        1	        1	        0	    ...     0.6272          0
    0.4605	        1	        1	        0	    ...     0.6272          0
    0.4594	        1	        1	        0	    ...     0.6272          0

    '''
    dataset_len = int(len(test_dataset))
    assert  cold_window < val_window ,'The cold window must be smaller than the validation window'
    assert val_window*2 < len(test_dataset) ,'A valid test set must be greater than twice the size of the validation window'
    # test_start_index

    train_set = TSADDataset(cold_data=train_dataset[:cold_window],
                            win_size=win_size,
                            step =step,
                            data=train_dataset[cold_window:-val_window],
                            cold_size = cold_window)
    val_set = TSADDataset(cold_data=train_dataset[-val_window-cold_window:-val_window],
                            win_size=win_size,
                            step=step,
                            data=train_dataset[-val_window:],
                            cold_size=cold_window)
    test_set = TSADDataset(cold_data=test_dataset[0:cold_window],
                          win_size=win_size,
                          step=step,
                          data=test_dataset[cold_window:],
                          cold_size=cold_window)

    # val_set=train_dataset[-self.val_window-self.cold_window:-self.val_window],train_dataset[-self.val_window:]
    # test_set = test_dataset[0:self.cold_window],test_dataset[self.cold_window:]

    ''' cw=cold_window
                           |←cold_set→|                               |←cold_set→|             
    train_dataset:          ■■■■■■■■■■|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■|■■■■■■■■■■|■■■■■■■■■■■■■■■■■■■
                           |←--cw-----,----------train_set----------------------→|
                                                                      |←--cw-----,-----val_set-----→|
    test_dataset:           ■■■■■■■■■|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
                           |←cold_set→|
                           |←---cw----,-------------test_set---------------------------------------→|
    '''



    return train_set, val_set, test_set


class TSADDataset(object):
    def __init__(self, data, win_size, step, cold_data, cold_size):
        self.data = data
        self.step = step
        self.win_size = win_size
        self.step = step
        self.cold_data = cold_data
        self.cold_win = cold_size

    def __len__(self):
        # return math.ceil(self.data_len / self.step)
        return math.ceil(self.data.shape[0] / self.step)

    def __getitem__(self, index):
        """
        train_dataset:
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ | ■■■■■■■■■■ | ■■■■■■■■■■■■■■■■■■■
        |(0) ←----sample 1- ------→ (30)|
                |(5) ←----sample 2 - ------→ (35)| (step = 5)
                            | (10)←---- sample 3 -------→ (40)|(step = 5)

        """
        index = index * self.step

        if index + self.win_size <= self.data.shape[0]:
            data_seg = self.data[index:index + self.win_size]
        else:
            shape = list(self.data.shape)
            shape[0] = index + self.win_size - self.data.shape[0]
            extend_data = np.full(shape, np.nan)
            data_seg = np.concatenate((self.data[index:], extend_data))
        y = data_seg[:, -1]  # The last column is the label
        x = data_seg[:, :-1]

        cold_index_left = min(0, index - self.cold_win) + len(self.cold_data)
        cold_from_cold_data = self.cold_data[cold_index_left:]
        cold_index_left_on_data = max(0, index - self.cold_win)
        cold_from_data = self.data[cold_index_left_on_data:index]
        cold_data = np.concatenate([cold_from_cold_data, cold_from_data], axis=0)
        return cold_data, x, y

        """ Test code 
        cold_data = np.random.random((5,3))
        data = np.random.random((100,3))
        index = 0 
        cold_win = 5
        cold_index_left = min(0,index - cold_win) + len(cold_data)
        cold_from_cold_data = cold_data[cold_index_left:]
        cold_index_left_on_data = max(0,index - cold_win)
        cold_from_data=data[cold_index_left_on_data:index]
        cold_data = np.concatenate([cold_from_cold_data,cold_from_data],axis=0)


        index = 3 
        cold_win = 5
        cold_index_left = min(0,index - cold_win) + len(cold_data)
        cold_from_cold_data = cold_data[cold_index_left:]
        cold_index_left_on_data = max(0,index - cold_win)
        cold_from_data=data[cold_index_left_on_data:index]
        cold_data = np.concatenate([cold_from_cold_data,cold_from_data],axis=0)

        index = 10
        cold_win = 5
        cold_index_left = min(0,index - cold_win) + len(cold_data)
        cold_from_cold_data = cold_data[cold_index_left:]
        cold_index_left_on_data = max(0,index - cold_win)
        cold_from_data=data[cold_index_left_on_data:index]
        cold_data = np.concatenate([cold_from_cold_data,cold_from_data],axis=0)

        """

