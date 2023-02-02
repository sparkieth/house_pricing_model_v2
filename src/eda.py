import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#script para EDA

def EDA1(train_data):
    fig, ax = plt.subplots(figsize=(25,10))
    sns.heatmap(data=train_data.isnull(), yticklabels=False, ax=ax)

    fig, ax = plt.subplots(figsize=(25,10))
    sns.countplot(x=train_data['SaleCondition'])
    sns.histplot(x=train_data['SaleType'], kde=True, ax=ax)
    sns.violinplot(x=train_data['HouseStyle'], y=train_data['SalePrice'],ax=ax)
    sns.scatterplot(x=train_data["Foundation"], y=train_data["SalePrice"], palette='deep', ax=ax)
    plt.grid()

