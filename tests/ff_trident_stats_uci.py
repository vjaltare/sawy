"""
Characterizing the statistics of a network being trained with TriDENT.
To keep it simple we'll stick with UCI datasets/MNIST with a simple feedforward neural network.

Created on: 04/27/2026

Notes: TODO
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
import csv
import fcntl
import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
import json

import pickle
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from datetime import date

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris



from utils import ternary_activation, load_cifar10_augment, dual_sample_ternary
from models import TernaryStochasticActivation, DualSampleTernary

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# -----------------------------
# Auxiliary functions
# ----------------------------- 
def save_payload(state, configs, filename):
    """
    Save model and parameters to a pickle.
    state: nnx.Model state
    configs: dict, configuration parameters
    """

    payload = {
        'configs' : configs,
        'state': state
    }

    checkpoint_dir = "/local_disk/vikrant/trident/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {filename_}")


# --------------------------------------------------------------
# Parsing input arguments
# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Trident-FFN on UCI-Iris")

    # take in the levels
    parser.add_argument("--levels", nargs="+", type=float, default=[-1.0, 0.0, 1.0])

    # take in the initial thresholds
    # parser.add_argument("--thresholds", nargs="+", type=float, default=[-1.0, 1.0])
    parser.add_argument("--threshold", type=float, default=0.0)

    # take in the noise standard deviation
    parser.add_argument("--noise_std", type=float, default=1.0)

    # model architecture parameters
    # parser.add_argument("--num64_blocks", type=int, default=2) 
    # parser.add_argument("--num128_blocks", type=int, default=2)
    # parser.add_argument("--num256_blocks", type=int, default=2)
    # parser.add_argument("--num512_blocks", type=int, default=2)
    parser.add_argument("--ff_layer_sizes", nargs="+", type=int, default=[1000, 500])

    # hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=int(5e4))
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--seed_next", type=int, default=0)
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument("--checkpoint_every", type=int, default=1000)


    #TODO: Add arguments for saving results and the model.
    parser.add_argument("--plot_results", action='store_true')



    return parser.parse_args()

# ------------------------------------
# Import UCI 
# ------------------------------------
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

# ------------------------------------
# Define Model
# ------------------------------------
class FFN(nnx.Module):

    def __init__(
            self,
            rngs: nnx.Rngs,
            layers: list[int],
            noise_std: float = 0.5,
            threshold: float = 0.0,
            ActivationFunction: nnx.Module = DualSampleTernary,
            **kwargs):
        
        self.activation = ActivationFunction(threshold=threshold, noise_std=noise_std, rngs=rngs)

        self.layers = nnx.List([
            nnx.Linear(in_features=li, out_features=lo, rngs=rngs) for li, lo in zip(layers[:-1], layers[1:])
        ])
    
    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# ------------------------------------
# Training functions
# ------------------------------------
def loss_fn(
        model: FFN,
        # batch: dict,
        X: jax.Array,
        labels: jax.Array,
        loss_function: Callable = optax.softmax_cross_entropy_with_integer_labels
    ):
    # forwad pass through the model
    logits = model(X)
    # using softmax cross-entropy with integer labels
    loss = loss_function(logits, labels=labels).mean()

    return loss, logits

        
# training step
@nnx.jit
def train_step(
    model: FFN,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    X: jax.Array, # input data
    label: jax.Array, # labels
    loss_fn: Callable = loss_fn
    ):

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, X, label)
    metrics.update(loss=loss, logits=logits, labels=label)
    optimizer.update(model, grads)

# evaluation step
@nnx.jit
def eval_step(
    model: FFN,
    metrics: nnx.MultiMetric,
    X: jax.Array, # input data
    label: jax.Array, # labels
    loss_fn: Callable = loss_fn
 ):
    loss, logits = loss_fn(model, X, label)
    metrics.update(loss=loss, logits=logits, labels=label)


# ------------------------------------
# Training Pipeline
# ------------------------------------
def train(
        model: FFN,
        optimizer: nnx.Optimizer,
        train_inputs: jax.Array,
        train_labels: jax.Array,
        test_inputs: jax.Array,
        test_labels: jax.Array,
        metrics_history: dict,
        metrics: nnx.MultiMetric,
        configs: dict,
        checkpoint_every: int,
        checkpoint_flag: bool = False,
        **kwargs
):
    
    print("--"*50)
    print(f"Training FFN on UCI Iris Dataset")
    print("--"*50)

    eval_every = configs['eval_every']
    train_steps = configs['train_steps']

    for step in tqdm(range(train_steps)):

        # train the model
        train_step(model, optimizer, metrics, train_inputs, train_labels)

        # TODO: Add script to checkpoint the model
        # Checkpointing the model
        if checkpoint_flag and (step%checkpoint_every==0 or step==train_steps-1):
            file = f"ffn_uci_iris_hidden_date_{today}_{configs['layers'][1]}_noise_{configs['noise_std']}_threshold_{configs['threshold']}_step_{step}.pkl"
            graphdef, state = nnx.split(model)
            save_payload(state, configs, file)
            model = nnx.merge(graphdef, state)
                    

        # evaluate and checkpoint the model
        if step > 0 and (step%eval_every==0 or step == train_steps-1):
            metrics_history['step'].append(step)

            # log the training metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value.item())
            metrics.reset()

            # evaluate the model on validation set
            eval_step(model, metrics, test_inputs, test_labels)

            # log the evaluation metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(value.item())
            metrics.reset()

        
            print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")
    
    best_accuracy = max(metrics_history['eval_accuracy'])
    best_acc_idx = jnp.argmax(jnp.array(metrics_history['eval_accuracy'])).astype(int)
    print(f"Best Eval Accuracy: {best_accuracy:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

    return model, metrics_history


# ------------------------------------
# Setting up training
# ------------------------------------
def main():

    # parse the input arguments
    args = parse_args()

    # load the data
    X_train, X_test, y_train, y_test = load_uci_iris(normalize=True, key=101, train_test_split=0.7)
    configs = {
        'train_steps': args.train_steps,
        'eval_every': args.eval_every,
        'learning_rate': args.learning_rate,
        'threshold': args.threshold,
        'noise_std': args.noise_std,
        'layers': [X_train.shape[1], 32, 3],
        'seed': 234
    }

    seed = configs['seed']
    rngs = nnx.Rngs(
        params=seed + 0,
        dropout=seed + 1,
        activation=seed + 2,
        next = seed + 3 
    )

    model = FFN(
        layers = configs['layers'],
        noise_std = configs['noise_std'],
        threshold = configs['threshold'],
        rngs = rngs
    )

    nnx.display(model)

    # optimizer = nnx.Optimizer(
    #     model,
    #     optax.chain(
    #         optax.clip_by_global_norm(1.0),
    #         optax.sgd(
    #             learning_rate=configs['learning_rate'],
    #             momentum=0.9
    #         )
    #     ),
    #     wrt=nnx.Param
    # )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=configs['learning_rate'],
                weight_decay=1e-4
            )
        ),
        wrt=nnx.Param
    )


    # define metrics
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss')
    )

    metrics_history = defaultdict(list)

    model, metrics_history = train(
        model=model,
        optimizer=optimizer,
        train_inputs=X_train,
        train_labels=y_train,
        test_inputs=X_test,
        test_labels=y_test,
        metrics_history=metrics_history,
        metrics=metrics,
        configs=configs,
        checkpoint_flag=args.checkpoint,
        checkpoint_every=args.checkpoint_every
    )


if __name__ == "__main__":
    main()





# X_train, X_test, y_train, y_test = load_uci_iris(normalize=True, key=101)

# plotting the feature-wise distribution
# fig, ax = plt.subplots(figsize=(5, 5))

# for i in range(X_train.shape[1]):
#         ax.hist(X_train[:, i], alpha=0.3, label=f'Feature {i}')

# ax.set_xlabel("Feature Value")
# ax.set_ylabel("Frequency")
# ax.legend()
# fig.savefig(f"../plots/uci_iris_featurewise_distribution_{today}.png", bbox_inches='tight', dpi=300)