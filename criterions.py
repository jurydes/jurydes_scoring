import numpy as np

def entropy(y) -> float:  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    
    if len(y) == 0:
        return 0.0
    
    p = np.mean(y, axis=0)
    p = p[p > 0]
    H = -np.sum(p * np.log2(p + EPS))
    return H

def mse(pred: np.ndarray, real: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) between predicted and actual values.
    
    Parameters
    ----------
    pred : np.ndarray
        Array of predicted values
    real : np.ndarray
        Array of actual values
    
    Returns
    -------
    float
        Mean squared error value
    """
    return np.linalg.norm(pred - real)**2 / len(pred)