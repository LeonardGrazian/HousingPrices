
import pandas as pd
import numpy as np

'''
    LotFrontage: 13.83007039 + 0.58228893 * sqrt(LotArea)
    Alley: 'None'
    MasVnrType: 'None', except where MasVnrArea!=0, then it's 'BrkFace'
    MasVnrArea: 0.0
    BsmtQual: 'None' Ordinal(None, Fa, TA, Gd, Ex)
    BsmtCond: 'None' Ordinal(Po, None, Fa, TA, Gd)
    BsmtExposure: 'None' Ordinal(None, No, Mn, Av, Gd)
    BsmtFinType1: 'None'
    BsmtFinType2: 'None'
    Electrical: 'None'
    FireplaceQu: 'None' Ordinal(Po, None, Fa, TA, Gd, Ex)
    GarageType: 'None'
    GarageYrBlt: YearBuilt
    GarageFinish: 'None'
    GarageQual: 'None' Ordinal(Po, None, Fa, TA, Gd, Ex)
    GarageCond: 'None' Ordinal(Po, None, Fa, TA, Gd, Ex)
    PoolQC: 'None' Ordinal(None, Fa, Gd, Ex)
    Fence: 'None'
    MiscFeature: 'None'
'''

fillna_dict = {
    'Alley': 'None',
    'MasVnrType': 'None',
    'MasVnrArea': 0.0,
    'BsmtQual': 'None',
    'BsmtCond': 'None',
    'BsmtExposure': 'None',
    'BsmtFinType1': 'None',
    'BsmtFinType2': 'None',
    'Electrical': 'None',
    'FireplaceQu': 'None',
    'GarageType': 'None',
    'GarageFinish': 'None',
    'GarageQual': 'None',
    'GarageCond': 'None',
    'PoolQC': 'None',
    'Fence': 'None',
    'MiscFeature': 'None'
}


cat_to_ord_maps = {
    'BsmtQual':     {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtCond':     {'Po': 0, 'None': 1, 'Fa': 2, 'TA': 3, 'Gd': 4},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'FireplaceQu':  {'Po': 0, 'None': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual':   {'Po': 0, 'None': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond':   {'Po': 0, 'None': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PoolQC':       {'None': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}
}


drop_list = ['BsmtHalfBath', '3SsnPorch', 'GarageQual', 'RoofMatl', 'PoolQC']


def custom_fillna(X):
    X = X.drop(columns=drop_list)
    for col in X.columns:
        if col in fillna_dict:
            X[col] = X[col].fillna(fillna_dict[col])

        elif col == 'LotFrontage':
            X[col] = X[col].fillna(13.83007039 + 0.58228893 * np.sqrt(X['LotArea']))

        elif col == 'GarageYrBlt':
            X[col] = X[col].fillna(X['YearBuilt'])

        if X[col].isnull().sum() > 0:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('None')
            else:
                X[col] = X[col].fillna(0.0)

    return X


def custom_get_categoricals(X, X_test):
    X, X_test = create_ordinals(X, X_test)
    X, X_test = reconcile_categoricals(X, X_test)
    return X, X_test


def fill_na_by_type(X):
    filled_values = X.select_dtypes(include='object').fillna('None')
    for col in filled_values.columns:
        X[col] = filled_values[col]
    filled_values = X.select_dtypes(exclude='object').fillna(0)
    for col in filled_values.columns:
        X[col] = filled_values[col]
    return X


def create_ordinals(X, X_test=[]):
    categorical_cols = X.select_dtypes(include='object').columns
    for cat_col in categorical_cols:
        if cat_col in cat_to_ord_maps:
            X[cat_col] = X[cat_col].replace(cat_to_ord_maps[cat_col])
            if np.any(X_test):
                X_test[cat_col] = X_test[cat_col].replace(cat_to_ord_maps[cat_col])
    if np.any(X_test):
        return X, X_test
    else:
        return X


def reconcile_categoricals(X, X_test):
    # get cols from X_test in case target is categorical
    categorical_cols = X.select_dtypes(include='object').columns
    for cat_col in categorical_cols:
        '''
        all_categories = np.unique(
                            np.append(np.unique(X[cat_col]),
                                        np.unique(X_test[cat_col])) )
        '''
        all_categories = np.unique(X[cat_col])
        X[cat_col] = pd.Categorical(X[cat_col], categories=all_categories)
        X_test[cat_col] = pd.Categorical(X_test[cat_col], categories=all_categories)
    return X, X_test


if __name__ == '__main__':
    root_dir = '/home/leonard/Desktop/Workspace/Kaggle/HousingPrices'
    train_data = pd.read_csv('{}/data/train.csv'.format(root_dir), index_col=0)
    train_data = custom_fillna(train_data)
    print(train_data.head())
