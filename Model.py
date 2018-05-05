
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error


def fill_na_by_type(X):
    filled_values = X.select_dtypes(include='object').fillna('None')
    for col in filled_values.columns:
        X[col] = filled_values[col]
    filled_values = X.select_dtypes(exclude='object').fillna(0)
    for col in filled_values.columns:
        X[col] = filled_values[col]
    return X


def reconcile_categoricals(X, X_test):
    # get cols from X_test in case target is categorical
    categorical_cols = X_test.select_dtypes(include='object').columns
    for cat_col in categorical_cols:
        all_categories = np.unique(
                            np.append(np.unique(X[cat_col]),
                                        np.unique(X_test[cat_col])) )
        X[cat_col] = pd.Categorical(X[cat_col], categories=all_categories)
        X_test[cat_col] = pd.Categorical(X_test[cat_col], categories=all_categories)
    return X, X_test


train_data = pd.read_csv('data/train.csv', index_col=0)
y = train_data['SalePrice']
X = fill_na_by_type( train_data.drop(columns=['SalePrice']) )

test_data = pd.read_csv('data/test.csv', index_col=0)
X_test = fill_na_by_type( test_data )

X, X_test = reconcile_categoricals(X, X_test)

# TODO: each feature - should it be onehot? discrete but ordered? and much later, monotonic?
# TODO: group features and train smaller models
X_enc = pd.get_dummies(X, columns=X.select_dtypes(include='category').columns)
X_test_enc = pd.get_dummies(X_test, columns=X_test.select_dtypes(include='category').columns)

(X_train, X_valid,
    y_train, y_valid) = train_test_split(X_enc,
                                        y,
                                        random_state=42)

model = RandomForestRegressor() # LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))
print()
print(math.sqrt(mean_squared_log_error(y_train, model.predict(X_train))))
print(math.sqrt(mean_squared_log_error(y_valid, model.predict(X_valid))))

# def make_submission(model, X_enc, y):
model.fit(X_enc, y)
predictions = model.predict(X_test_enc)
predictions_df = pd.DataFrame(data=predictions, index=X_test_enc.index, columns=['SalePrice'])
predictions_df.to_csv('output/submission_0.csv')
