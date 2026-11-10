import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """
    arr = np.zeros(len(x))
    for i in range(len(x)):
        if(x[i]<0):
            arr[i] = (alpha*x[i])
        else:
            arr[i] = x[i]

    return arr
            