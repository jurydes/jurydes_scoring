import numpy as np

def entropy(y):  
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

def mse(pred, real):
    return np.mean((pred - real)**2)