import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')
        self.trim()

        self.base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        
        
    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')
            

    def get_subsets(self):
        np.random.seed(2)
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]
        
        
        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()
        
        return [df_train,df_val,df_test]
    
    def get_label_data(self,df):
        y_orig = df.msrp.values
        y = np.log1p(df.msrp.values)
        del df['msrp']
        
        return y_orig, y
    
    

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]

   
    def prepare_X(self,df):
        
        df_num = df[self.base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X
    
    
    
    def validate(self,y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
    
    
    def display(self, X, y, y_pred):
        columns = ['engine_cylinders','transmission_type','driven_wheels','number_of_doors',
                   'market_category','vehicle_size','vehicle_style','highway_mpg','city_mpg','popularity']
        X = X.copy()
        X = X[columns]
        X['msrp'] =np.expm1(y.round(2))
        X['msrp_pred'] = np.expm1(y_pred.round(2))
        print(X.head(5).to_string(index=False))
    
    
if __name__ == "__main__":
    
    
    cp = CarPrice()

    df_train, df_val, df_test = cp.get_subsets()
    
    y_train_orig, y_train = cp.get_label_data(df_train)
    y_val_orig, y_val = cp.get_label_data(df_val)
    y_test_orig, y_test = cp.get_label_data(df_test)
    
    
    X_train = cp.prepare_X(df_train)
    w_0, w = cp.linear_regression(X_train, y_train)
    y_train_pred = w_0 + X_train.dot(w)
    
    
    X_val = cp.prepare_X(df_val)
    y_val_pred = w_0 + X_val.dot(w)
    
    X_test = cp.prepare_X(df_test)
    y_test_pred = w_0 + X_test.dot(w)

    perf_train = round(cp.validate(y_train, y_train_pred),4)
    perf_val = round(cp.validate(y_val, y_val_pred),4)
    perf_test = round(cp.validate(y_test, y_test_pred),4)
    
    cp.display( df_test, y_test, y_test_pred)
    print('Test rmse: ', round(perf_test,4))