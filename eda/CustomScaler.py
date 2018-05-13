
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

class CustomScaler(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.scaler = StandardScaler(*args, **kwargs)
        self.cont_col_names = ['MSSubClass', 'LotFrontage', 'LotArea',
                    'OverallQual', 'OverallCond', 'YearBuilt',
                    'YearRemodAdd', 'MasVnrArea', 'BsmtQual', 'BsmtCond',
                    'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                    'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath',
                    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                    'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageCars',
                    'GarageArea', 'GarageCond', 'WoodDeckSF', 'OpenPorchSF',
                    'EnclosedPorch','ScreenPorch', 'PoolArea', 'MiscVal',
                    'MoSold', 'YrSold']

    # takes X_enc, a pandas dataframe where discrete vars are one hot encoded
    def fit(self, X, y=None):
        self.scaler.fit(X[self.cont_col_names], y)
        return self


    # takes X_enc, a pandas dataframe where discrete vars are one hot encoded
    def transform(self, X, y=None, copy=None):
        continuous_cols = self.scaler.transform(X[self.cont_col_names])
        discrete_cols = X.drop(columns=self.cont_col_names).values

        return np.concatenate([continuous_cols, discrete_cols], axis=1)


    def get_params(self, *args, **kwargs):
        return self.scaler.get_params(*args, **kwargs)
