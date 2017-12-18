import numpy as np
import pandas as pd

def loadDataSet(fileName):
    data_frame = pd.read_cdv(fileName, header=None, prefix='X')
    return data_frame

if __name__ == '__main__':
    pass

