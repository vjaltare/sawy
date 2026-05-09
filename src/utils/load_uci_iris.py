"""
Loading and pre-processing UCI Iris dataset for training small neural nets
"""

import jax
import jax.numpy as jnp
from sklearn.datasets import load_iris

def load_uci_iris(
        key: int = 0,
        train_test_split: float = 0.8,
        normalize: bool = True, # normalize the data to
        **kwargs):
    data = load_iris()

    X = jnp.array(data['data'])
    y = jnp.array(data['target'])

    # feature-wise normalization [0, 1]
    if normalize:
        Xmin = X.min(axis=0)  
        Xmax = X.max(axis=0)
        X = (X - Xmin) / (Xmax - Xmin + 1e-8)

    # split the dataset into train and test sets: TODO
    # shuffle the data
    key = jax.random.key(key)
    random_permutation = jax.random.permutation(key, X.shape[0])
    X = X[random_permutation]
    y = y[random_permutation]

    # split the data
    split_index = jnp.floor(train_test_split * X.shape[0]).astype(int)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Size of training data {X_train.shape}, size of test data {X_test.shape}")



    return X_train, X_test, y_train, y_test