
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge


train_data = pd.read_csv('data/train.csv', index_col=0)

y = train_data['SalePrice']
X = train_data.drop(columns=['SalePrice'])

filled_values = X.select_dtypes(include='object').fillna('None')
for col in filled_values.columns:
    X[col] = filled_values[col]
filled_values = X.select_dtypes(exclude='object').fillna(0)
for col in filled_values.columns:
    X[col] = filled_values[col]

# TODO: each feature - should it be onehot? discrete but ordered? and much later, monotonic?
le = LabelEncoder()
X_enc = X.apply(le.fit_transform)
ohe = OneHotEncoder()
X_enc = ohe.fit_transform(X_enc)

(X_train, X_test,
    y_train, y_test) = train_test_split(X_enc,
                                        y,
                                        random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
