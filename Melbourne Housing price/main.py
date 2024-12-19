# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#code
file_path = '/kaggle/input/melbourne-house-data/train.csv'
train_data = pd.read_csv(file_path)
y = train_data.SalePrice 
melb_predictors = train_data.drop(['SalePrice'], axis=1)
my_imputer = SimpleImputer()
ordinal_encoder = OrdinalEncoder()
#preprocessing Numerical data
numerical_cols = [cname for cname in melb_predictors.columns if melb_predictors[cname].dtype in ['int64', 'float64']]
numerical_data = melb_predictors[numerical_cols].copy()
imputed_numerical_data = pd.DataFrame(my_imputer.fit_transform(numerical_data))
imputed_numerical_data.columns = numerical_data.columns
#preprocessing Categorical data
categorical_cols = [cname for cname in melb_predictors.columns if melb_predictors[cname].nunique() < 10 and melb_predictors[cname].dtype == "object"]
categorical_data = melb_predictors[categorical_cols].copy()
ordinal_cat_data = pd.DataFrame(ordinal_encoder.fit_transform(categorical_data))
imputed_ordinal_cat_data = pd.DataFrame(my_imputer.fit_transform(ordinal_cat_data))
imputed_ordinal_cat_data.columns = categorical_data.columns
#total train data
total_train = pd.concat([imputed_numerical_data, imputed_ordinal_cat_data], axis=1)
#model work
my_model = XGBRegressor()
my_model.fit(total_train, y)
#test data
file_path2 ='/kaggle/input/melbourne-house-data/test.csv'
test_data =pd.read_csv(file_path2)
columns_train = total_train.columns.tolist()
red_test_data = test_data[columns_train]
categorical_testdata = red_test_data[categorical_cols].copy()
ord_cat_test_data=  pd.DataFrame(ordinal_encoder.fit_transform(categorical_testdata))
ord_cat_test_data.columns = categorical_data.columns
ultimate_test_data = pd.concat([red_test_data[numerical_cols], ord_cat_test_data], axis=1)
#predictions
predictions = my_model.predict(ultimate_test_data)
#output
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
