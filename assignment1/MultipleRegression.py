import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class HousePrice:

    def __init__(self):
        boston = load_boston()
        self.data = pd.DataFrame(boston.data)
        self.data.columns = boston.feature_names
        self.data['PRICE'] = boston.target
    
    def get_input_feature(self):
        base=['LSTAT','RM','B']
        return self.data[base].values

    def get_price(self):
        return self.data['PRICE']
    
    def get_train_test_datasets(self, test_fraction):
        return train_test_split(X, y, test_size = test_fraction, random_state = 4)

        
if __name__ == "__main__":
    obj = HousePrice()

    X = obj.get_input_feature()
    y = obj.get_price()

    X_train, X_test, y_train, y_test = obj.get_train_test_datasets(0.3)

    # Create a Linear regressor
    lm = LinearRegression()
    
    # Train the model using the training set 
    lm.fit(X_train, y_train)

    # Predict for training dataset
    y_pred = lm.predict(X_train)
    print('For Training Dataset:')
    print('R^2:',metrics.r2_score(y_train, y_pred))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    print('\n') 

    # For test dataset:
    # Predicted value    
    y_test_pred = lm.predict(X_test)
   
    acc_linreg = metrics.r2_score(y_test, y_test_pred)
    print('For Test Dataset:')
    print('R^2:', acc_linreg)
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    adjusted_r_squared = 1 - (1-acc_linreg)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print('Adjusted R_squared:', adjusted_r_squared)        


