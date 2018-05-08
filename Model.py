
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

from eda.FillMissing import custom_fillna, custom_get_categoricals


def make_submission(model, X_enc, y):
    model.fit(X_enc, y)
    predictions = model.predict(X_test_enc)
    predictions_df = pd.DataFrame(data=predictions, index=X_test_enc.index, columns=['SalePrice'])
    predictions_df.to_csv('output/submission_2.csv')


train_data = pd.read_csv('data/train.csv', index_col=0)
y = train_data['SalePrice']
X = fill_na_by_type( train_data.drop(columns=['SalePrice']) )

test_data = pd.read_csv('data/test.csv', index_col=0)
X_test = fill_na_by_type( test_data )

X, X_test = custom_get_categoricals(X, X_test) # reconcile_categoricals(X, X_test)

# TODO: group features and train smaller models
X_enc = pd.get_dummies(X, columns=X.select_dtypes(include='category').columns)
X_test_enc = pd.get_dummies(X_test, columns=X_test.select_dtypes(include='category').columns)

(X_train, X_valid,
    y_train, y_valid) = train_test_split(X_enc,
                                        y,
                                        random_state=42)

model = Lasso() # RandomForestRegressor()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))
print()
print(math.sqrt(mean_squared_log_error(y_train, model.predict(X_train))))
print(math.sqrt(mean_squared_log_error(y_valid, model.predict(X_valid))))

make_submission(model, X_enc, y)
