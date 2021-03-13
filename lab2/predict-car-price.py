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
    
    

    
    def linear_regression_reg(self,X, y, r):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
    
        XTX = X.T.dot(X)
        reg = r * np.eye(XTX.shape[0])
        XTX = XTX + reg

        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
    
        return w[0], w[1:]

        

   
   
    
    
    def prepare_X(self,df):
        df = df.copy()
        features = self.base.copy()

        df['age'] = 2017 - df.year
        features.append('age')
    
        for v in [2, 3, 4]:
            feature = 'num_doors_%s' % v
            df[feature] = (df['number_of_doors'] == v).astype(int)
            features.append(feature)

        for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
            feature = 'is_make_%s' % v
            df[feature] = (df['make'] == v).astype(int)
            features.append(feature)

        for v in ['regular_unleaded', 'premium_unleaded_(required)', 
                  'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
            feature = 'is_type_%s' % v
            df[feature] = (df['engine_fuel_type'] == v).astype(int)
            features.append(feature)

        for v in ['automatic', 'manual', 'automated_manual']:
            feature = 'is_transmission_%s' % v
            df[feature] = (df['transmission_type'] == v).astype(int)
            features.append(feature)

        for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
            feature = 'is_driven_wheens_%s' % v
            df[feature] = (df['driven_wheels'] == v).astype(int)
            features.append(feature)

        for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
            feature = 'is_mc_%s' % v
            df[feature] = (df['market_category'] == v).astype(int)
            features.append(feature)

        for v in ['compact', 'midsize', 'large']:
            feature = 'is_size_%s' % v
            df[feature] = (df['vehicle_size'] == v).astype(int)
            features.append(feature)

        for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
            feature = 'is_style_%s' % v
            df[feature] = (df['vehicle_style'] == v).astype(int)
            features.append(feature)

        df_num = df[features]
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
        X['msrp_pred'] =np.expm1 (y_pred.round(2))
        print(X.head(5).to_string(index=False))
    
    
if __name__ == "__main__":
    
    
    cp = CarPrice()

    df_train, df_val, df_test = cp.get_subsets()
    
    y_train_orig, y_train = cp.get_label_data(df_train)
    y_val_orig, y_val = cp.get_label_data(df_val)
    y_test_orig, y_test = cp.get_label_data(df_test)
    
    
    X_train = cp.prepare_X(df_train)
    
        
    X_val = cp.prepare_X(df_val)
    print("RMSE values for different values of r ")
    for r in [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
        w_0, w = cp.linear_regression_reg(X_train, y_train, r=r)
        y_pred = w_0 + X_val.dot(w)
        print('%6s' %r, cp.validate(y_val, y_pred))
   
    # r=0.000001  is used because it gave the least rmse value
    # 1e-06   0.46022512570232416
    # 0.0001  0.460225493112593
    # 0.001   0.46022676283119085
    # 0.01    0.460239496312196
    # 0.1     0.4603700695820121
    # 1       0.461829804265233
    # 5       0.4684079627532219
    # 10      0.47572481006951994

    print("r = 0.000001  is used because it gave the least rmse value")
    w_0, w = cp.linear_regression_reg(X_train, y_train, r=0.000001)
    
       
    X_test = cp.prepare_X(df_test)
    y_test_pred = w_0 + X_test.dot(w)
    perf_test = round(cp.validate(y_test, y_test_pred),2)
    
    # printing predictions on test data set
    print("Printing original msrp vs. predicted msrp of 5 cars") 
    cp.display( df_test, y_test, y_test_pred)
    print('Test rmse: ', round(perf_test,4))