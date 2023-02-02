"""This scripts trains and predicts a random forest
with predetermined datasets."""

from Formulas import upload1
from Formulas import EDA1
from Formulas import funciones1
from Formulas import modelo1

# Run the following: python main_tarea03_SE.py

# 1. Upload data

train_data, test_data = upload1.download_data("train.csv", "test.csv")

# 2. perform EDA

EDA1.EDA1(train_data)

# 3. Preprocess data

# 3.1 filling missing data
set1 = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
train_data, test_data = funciones1.missing_data1(train_data, test_data, set1)

# 3.2 dropping variables

drop_col = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold',
            'YrSold', 'MSSubClass', 'GarageType', 'GarageArea', 'GarageYrBlt',
            'GarageFinish', 'YearRemodAdd', 'LandSlope', 'BsmtUnfSF',
            'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1',
            'Condition2', 'Heating', 'Exterior1st', 'Exterior2nd',
            'HouseStyle', 'LotShape', 'LandContour', 'LotConfig', 'Functional',
            'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu', 'WoodDeckSF',
            'GarageQual', 'GarageCond', 'OverallCond']

train_data = funciones1.dropping1(train_data, drop_col)
test_data = funciones1.dropping1(test_data, drop_col)

# encoding ordinal variables
cat1 = ['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
cat2 = ['N', 'P', 'Y']
cat3 = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']
cat4 = ['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
cat5 = ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
cat6 = ['C(all)', 'RH', 'RM', 'RL', 'FV']
cat7 = ['Slab', 'BrkTil', 'Stone', 'CBlock', 'Wood', 'PConc']
cat8 = ['MeadowV', 'IDOTRR', 'BrDale', 'Edwards', 'BrkSide', 'OldTown',
        'NAmes',
        'Sawyer', 'Mitchel', 'NPkVill', 'SWISU', 'Blueste',
        'SawyerW', 'NWAmes',
        'Gilbert', 'Blmngtn', 'ClearCr', 'Crawfor', 'CollgCr',
        'Veenker', 'Timber',
        'Somerst', 'NoRidge', 'StoneBr', 'NridgHt']
cat9 = ['None', 'BrkCmn', 'BrkFace', 'Stone']
cat10 = ['AdjLand', 'Abnorml', 'Alloca', 'Family', 'Normal', 'Partial']
cat11 = ['Gambrel', 'Gable', 'Hip', 'Mansard', 'Flat', 'Shed']
cat12 = ['ClyTile', 'CompShg', 'Roll', 'Metal', 'Tar&Grv',
         'Membran', 'WdShake', 'WdShngl']

# algunas variables es mejor hacerlas por oHE, otras por Ordinal encoding

ordinal_col = {'BsmtQual': cat1, 'BsmtCond': cat1, 'ExterQual': cat1,
               'ExterCond': cat1,
               'KitchenQual': cat1, 'PavedDrive': cat2, 'Electrical': cat3,
               'BsmtFinType1': cat4, 'BsmtFinType2': cat4, 'Utilities': cat5}
ohe_col = {'MSZoning': cat6, 'Foundation': cat7, 'Neighborhood': cat8,
           'MasVnrType': cat9, 'SaleCondition': cat10, 'RoofStyle': cat11,
           'RoofMatl': cat12}

train_data, test_data = funciones1.ordinal_encoder(train_data,
                                                   test_data, ordinal_col)
# train_data, test_data = funciones1.OHE1(train_data,test_data, ohe_col)
# creating new variables
train_data = funciones1.new_variables1(train_data)
test_data = funciones1.new_variables1(test_data)

# dropping columns
drop_col1 = ['OverallQual', 'ExterCond', 'ExterQual', 'BsmtCond', 'BsmtQual',
             'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'OpenPorchSF',
             'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'BsmtFullBath',
             'BsmtHalfBath', 'FullBath', 'HalfBath']

train_data = funciones1.dropping1(train_data, drop_col1)
test_data = funciones1.dropping1(test_data, drop_col1)

# 4. Create the model

modelofinal1 = modelo1.modelo_random_forest(train_data)
prediccion = modelo1.prediccion(test_data)
