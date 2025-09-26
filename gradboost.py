from criterions import mse, entropy
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
import pandas as pd 

class MyGradientBoosting:
    def __init__(self, criterion_name='mse', learning_rate=0.001, n_estimators=100, max_depth=3):
        
        criterion_name = criterion_name.lower()
        
        if criterion_name == 'mse':
            self.criterion = mse
        elif criterion_name == 'entropy':
            self.criterion = entropy
        else:
            raise ValueError('Wrong criterion name')
        self.criterion_name = criterion_name
        self.max_depth = max_depth
        self.models = []
        self.zero_pred = None
        if learning_rate > 0:
            self.learning_rate = learning_rate
        else:
            raise ValueError('Learning rate must be a positive number')
        if n_estimators > 0:
            self.n_estimators = n_estimators
        else:
            raise ValueError('The number of estimators must be a positive number')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Makes predictions lol 
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise TypeError('Wrong X type')

        X = np.array(X)

        pred = np.full(X.shape[0], fill_value=self.zero_pred)
        
        for model in self.models:
            pred += self.learning_rate * model.predict(X)

        return pred
    
    def predict_proba(self, X):
        y_pred = self.predict(X)
        return np.column_stack((1 - y_pred, y_pred))

    def grads(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        '''
        Computing gradients for MSE and Entropy
        '''
        if self.criterion_name == 'mse':
            return -2*(y_pred - y_real)
        elif self.criterion_name == 'entropy':
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1-eps)
            return (y_pred - y_real) / (y_pred * (1 - y_pred))

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Creates an array of models for future predictions
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise TypeError('Wrong X type')

        if not isinstance(y, (np.ndarray, pd.Series, list)):
            raise TypeError('Wrong y type')

        X = np.array(X)
        y = np.array(y)

        if self.criterion_name == 'mse':
            self.zero_pred = np.mean(y)
        else:
            pos_class_prob = np.mean(y) 
            eps = 1e-15
            pos_class_prob = np.clip(pos_class_prob, eps, 1-eps)
            self.zero_pred = np.log(pos_class_prob / (1 - pos_class_prob))

        cur_pred = np.full(y.shape[0], self.zero_pred)

        for _ in np.arange(self.n_estimators):
            gradient = self.grads(y_pred=cur_pred, y_real=y)

            if self.criterion_name == 'mse':
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
            else:
                tree = DecisionTreeClassifier(max_depth=self.max_depth)

            tree.fit(X, gradient)            
            cur_pred += self.learning_rate * tree.predict(X)

            self.models.append(tree)
    