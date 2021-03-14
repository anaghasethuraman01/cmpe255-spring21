import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures    

class HousePrice:

    def __init__(self):
        boston = load_boston()
        self.data = pd.DataFrame(boston.data)
        self.data.columns = boston.feature_names
        self.data['PRICE'] = boston.target
    
    def get_input_feature(self):
        X = self.data['LSTAT'].values
        return X.reshape(-1,1)

    def get_price(self):
        return self.data['PRICE']
    
    def get_train_test_datasets(self, test_fraction):
        return train_test_split(X, y, test_size = test_fraction, random_state = 4)

    def plot(self, X_test, y_pred, degree):
        coeff = np.polyfit(X_test[:,0], y_pred, degree)
        p = np.poly1d(coeff)
        plt.plot(X_test, p(X_test), '-')
        plt.xlabel('LSTAT')
        plt.ylabel('House Price')
        plt.show()

        X_coord = np.linspace(np.min(X_test), np.max(X_test), len(X_test))
        plt.plot(X_test, y_test,'o')
        plt.plot(X_coord, p(X_coord)) # 'smoother' line
        plt.xlabel('LSTAT')
        plt.ylabel('House Price')
        plt.show()
        
if __name__ == "__main__":
    obj = HousePrice()

    X = obj.get_input_feature()
    y = obj.get_price()

    X_train, X_test, y_train, y_test = obj.get_train_test_datasets(0.3)


    poly_reg = PolynomialFeatures(degree=2)
    X_train_t = poly_reg.fit_transform(X_train)
    X_test_t = poly_reg.fit_transform(X_test)
    
    # Train the model using the training set 
    lin_reg=LinearRegression()
    lin_reg.fit(X_train_t,y_train)


    # For test dataset:
    # Predicted value    
    y_pred = lin_reg.predict(X_test_t)
   
    # Model Evaluation
    acc_linreg = metrics.r2_score(y_test, y_pred)
    print('R^2:', acc_linreg)
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Plotting for degree = 2
    obj.plot(X_test, y_pred, 2)

    # Plotting for degree = 20
    obj.plot(X_test, y_pred, 20)
    
    
