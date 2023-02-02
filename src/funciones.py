import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Funciones de preprocesamiento

#función para llenar los valores faltantes

set1=['FireplaceQu','BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2']
def missing_data1(trainset,testset,vars_na=set1):
    for i in vars_na:
        trainset[i].fillna("No", inplace=True)
        testset[i].fillna("No", inplace=True)
    
    def fill_all_missing_values(data):
        for col in data.columns:
            if((data[col].dtype == 'float64') or (data[col].dtype == 'int64')):
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
        return data

    trainset=fill_all_missing_values(trainset)
    testset=fill_all_missing_values(testset) 
    
    return trainset,testset

#función para descartar variables

drop_col = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold', 'YrSold', 'MSSubClass',
            'GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'YearRemodAdd', 'LandSlope',
            'BsmtUnfSF', 'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1', 'Condition2', 'Heating',
             'Exterior1st', 'Exterior2nd', 'HouseStyle', 'LotShape', 'LandContour', 'LotConfig', 'Functional',
             'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu', 'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond'
           ]
def dropping1(dataset,cols_todrop=drop_col):
    dataset.drop(cols_todrop, axis=1, inplace=True)
    return dataset

#función que aplica ordinal encoder a las variables seleccionadas para las categorías seleccionadas

cat1=['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
cat2=['N', 'P', 'Y']
cat3=['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']
cat4=['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
cat5=['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
cat6=['C (all)', 'RH', 'RM', 'RL', 'FV']
cat7=['Slab', 'BrkTil', 'Stone', 'CBlock', 'Wood', 'PConc']
cat8=['MeadowV', 'IDOTRR', 'BrDale', 'Edwards', 'BrkSide', 'OldTown', 'NAmes', 'Sawyer', 'Mitchel', 'NPkVill', 'SWISU', 'Blueste', 'SawyerW', 'NWAmes', 'Gilbert', 'Blmngtn', 'ClearCr', 'Crawfor', 'CollgCr', 'Veenker', 'Timber', 'Somerst', 'NoRidge', 'StoneBr', 'NridgHt']
cat9=['None', 'BrkCmn', 'BrkFace', 'Stone']
cat10=['AdjLand', 'Abnorml','Alloca', 'Family', 'Normal', 'Partial']
cat11=['Gambrel', 'Gable','Hip', 'Mansard', 'Flat', 'Shed']
cat12=['ClyTile', 'CompShg', 'Roll','Metal', 'Tar&Grv','Membran', 'WdShake', 'WdShngl']

ordinal_col = {'BsmtQual': cat1, 'BsmtCond': cat1, 'ExterQual': cat1,
               'ExterCond': cat1,
               'KitchenQual': cat1, 'PavedDrive': cat2, 'Electrical': cat3,
               'BsmtFinType1': cat4, 'BsmtFinType2': cat4, 'Utilities': cat5}
ohe_col = {'MSZoning': cat6, 'Foundation': cat7, 'Neighborhood': cat8,
           'MasVnrType': cat9, 'SaleCondition': cat10, 'RoofStyle': cat11,
           'RoofMatl': cat12}

def ordinal_encoder(train,test,ordinal_cols=ordinal_col):
    for var,cat in ordinal_col.items():
        OE = OrdinalEncoder(categories='auto')
        train[var] = OE.fit_transform(train[[var]])
        test[var] = OE.transform(test[[var]])
    return train,test

#función para OHE

def OHE1(train,test,ordinal_cols=ohe_col):
    for var,cat in ordinal_col.items():
        OE = OneHotEncoder(categories='auto')
        train[var] = OE.fit_transform(train[[var]])
        test[var] = OE.transform(test[[var]])
    return train,test

#función que aplica label encoder

Level_col = ['Street' ,'BldgType', 'SaleType', 'CentralAir']

def encode_categorical_columns(train, test, Level_col1=Level_col):
    encoder = LabelEncoder()
    for col in Level_col1:
        train[col] = encoder.fit_transform(train[col])
        test[col]  = encoder.transform(test[col])

#función para hacer variables muy particulares nuevas a partir de variables ya existentes

def new_variables1(df):
    df['BsmtRating']= df['BsmtCond'] * df['BsmtQual']
    df['ExterRating'] = df['ExterCond'] * df['ExterQual']
    df['BsmtFinTypeRating'] = df['BsmtFinType1'] * df['BsmtFinType2']
    
    df['BsmtBath'] = df['BsmtFullBath'] + df['BsmtHalfBath']
    df['Bath'] = df['FullBath'] + df['HalfBath']
    df['PorchArea'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    return(df)