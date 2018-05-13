
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import math

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error

from mlxtend.regressor import StackingRegressor

import matplotlib.pyplot as plt

from eda.FillMissing import custom_fillna, custom_get_categoricals
from eda.CustomScaler import CustomScaler


def make_submission(model, X_enc, X_test_enc, y):
    model.fit(X_enc, y)
    predictions = model.predict(X_test_enc)
    predictions = np.exp(predictions)
    predictions_df = pd.DataFrame(data=predictions, index=X_test_enc.index, columns=['SalePrice'])
    predictions_df.to_csv('output/submission4.csv')


def search_params(pipeline, param_dict, X_enc, y):
    grid = GridSearchCV(pipeline,
                        param_dict,
                        scoring='neg_mean_squared_log_error',
                        cv=KFold(n_splits=3, shuffle=True, random_state=42))

    grid.fit(X_enc, np.log(y))
    print(grid.best_params_)
    scores = grid.cv_results_['mean_test_score']
    plt.plot(scores)
    plt.show()


train_data = pd.read_csv('data/train.csv', index_col=0)
y = train_data['SalePrice']
X = custom_fillna( train_data.drop(columns=['SalePrice']) )

test_data = pd.read_csv('data/test.csv', index_col=0)
X_test = custom_fillna( test_data )

X, X_test = custom_get_categoricals(X, X_test) # reconcile_categoricals(X, X_test)

# TODO: group features and train smaller models
X_enc = pd.get_dummies(X, columns=X.select_dtypes(include='category').columns)
X_test_enc = pd.get_dummies(X_test, columns=X_test.select_dtypes(include='category').columns)

'''
(X_train, X_valid,
    y_train, y_valid) = train_test_split(X_enc,
                                        np.log(y),
                                        random_state=42)
'''

# pipeline = Pipeline([('scaler', CustomScaler()),
#                         ('model', RandomForestRegressor(n_estimators=1000))])
# # param_dict = {'model__min_impurity_decrease': np.linspace(0, .01, 100)}
# param_dict = {'model__max_depth': np.linspace(12, 21, 30)}
# search_params(pipeline, param_dict, X_enc, y)

pipe1 = Pipeline([('scaler', CustomScaler()),
                    ('model', Ridge(alpha=27.3737))])
pipe2 = Pipeline([('scaler', CustomScaler()),
                    ('model', RandomForestRegressor(n_estimators=1000,
                                                    max_depth=20))])
model = StackingRegressor(regressors=(pipe1, pipe2),
                            meta_regressor=LinearRegression())

make_submission(pipe1, X_enc, X_test_enc, np.log(y))
