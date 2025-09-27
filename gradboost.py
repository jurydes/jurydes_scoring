from criterions import mse, entropy
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
import pandas as pd 
from typing import Literal
eps = 1e-6


class MyGradientBoosting:
    def __init__(self, criterion_name: Literal['mse', 'entropy'] = 'mse', learning_rate: float=0.001, n_estimators: int=100, max_depth: int=3):
        '''
        Initialises MyGradientBoosting with given parameters. 

        Parameters
        ----------
        criterion name: {'mse', 'entropy'}, default=mse 
            The name of wanted criterion:
            mse ~ mean squared error
            entropy 
        
        learning_rate: float, default=0.001
            Learning rate of every tree in the ansamble.
        
        n_estimators: int, default=100
            The number of trees in the ansamble.

        max_depth: int, default=3
            Max depth of every tree in the ansamble.    
        '''
        criterion_name = criterion_name.lower()
        self.estimators={"mse" : DecisionTreeRegressor, "entropy" : DecisionTreeClassifier}
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
        Makes predictions using given data.

        Parameters
        ----------
        X: np.ndarray
            Data to make predictions with
        
        Returns
        -------
        np.ndarray
            Predicted values
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise TypeError(f'X type mismatch.\n Expected: one of (np.ndarray, pd.DataFrame, list)\n Got: {X.dtype}')

        X = np.array(X)

        pred = np.full(X.shape[0], fill_value=self.zero_pred)
        
        for model in self.models:
            pred += self.learning_rate * model.predict(X)

        return pred
    
    def predict_proba(self, X):
        '''
        Predicts probabilities of classes.

        Parameters
        ----------
        X: np.ndarray
            Data to make predictions with
        
        Returns
        -------
        np.ndarray
            Predicted probabilities
        '''
        y_pred = self.predict(X)
        return np.column_stack((1 - y_pred, y_pred))

    def grads(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        '''
        Computing gradients for MSE and Entropy

        Parameters
        ----------
        y_real : np.ndarray
            Array of actual values
        y_pred : np.ndarray
            Array of predicted values

        Returns
        -------
        np.ndarray
            Computed gradients
        '''
        if self.criterion_name == 'mse':
            return -2*(y_pred - y_real)
        else:
            y_pred = np.clip(y_pred, eps, 1-eps)
            return (y_pred - y_real) / (y_pred * (1 - y_pred))

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Creates an array of trained models for future predictions

        Parameters
        ----------
        X : np.ndarray
            Data
        y_pred : np.ndarray
            Target
        '''
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise TypeError(f'X type mismatch.\n Expected: one of (np.ndarray, pd.DataFrame, list)\n Got: {X.dtype}')

        if not isinstance(y, (np.ndarray, pd.Series, list)):
            raise TypeError(f'X type mismatch.\n Expected: one of (np.ndarray, pd.DataFrame, list)\n Got: {X.dtype}')

        X = np.array(X)
        y = np.array(y)

        if self.criterion_name == 'mse':
            self.zero_pred = np.mean(y)
        else:
            pos_class_prob = np.mean(y) 
            pos_class_prob = np.clip(pos_class_prob, eps, 1-eps)
            self.zero_pred = np.log(pos_class_prob / (1 - pos_class_prob))

        cur_pred = np.full(y.shape[0], self.zero_pred)

        for _ in np.arange(self.n_estimators):
            gradient = self.grads(y_pred=cur_pred, y_real=y)

            tree_class = self.estimators[self.criterion_name]
            tree = tree_class(max_depth=self.max_depth)
            tree.fit(X, gradient)            
            cur_pred += self.learning_rate * tree.predict(X)

            self.models.append(tree)
    