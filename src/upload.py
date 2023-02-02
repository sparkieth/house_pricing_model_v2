# Script para cargar los datos

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def download_data(doc_train="train.csv",doc_test="test.csv"):
    train_data = pd.read_csv(doc_train)
    test_data = pd.read_csv(doc_test)
    test_ids = test_data['Id']
    return train_data,test_data