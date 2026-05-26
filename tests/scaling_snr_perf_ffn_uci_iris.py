"""
Studying scaling properties of TriDENT.
- Increase the number of hidden units and train each configutation with different noise levels.
- Evaluate the SNR, accuracy and sparsity of the trained models on the test set.

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

from utils import load_uci_iris, dual_sample_ternary
from models import TernaryStochasticActivation, DualSampleTernary, FFN

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"

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

    checkpoint_dir = MODEL_PATH
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {filename_}")


def save_data_pkl(data, metadata, filename):
    """
    Save data nad related metadata into a pickle file.

    data: dict, data to be saved.
    metadata: dict, description of the data and how to load it.
    filename: str, name of the file to save the data into. Include path to data directure in this string.
    """

    payload = {
        'metadata': metadata,
        'data': data
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(payload, f)

    print(f"Data saved to {filename}")


# --------------------------------------------------------------
# Parsing input arguments
# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Noise and Scaling | Trident-FFN on UCI-Iris")

    # extermeties of layer sizes
    parser.add_argument("--hidden_layer_range", nargs="+", type=float, default=[1.0, 5.0]) # range of hidden layer sizes to explore -- logarithmic to the base 10
    parser.add_argument("--hl_num_points", type=int, default=50) # number of points to sample in the hidden layer size range

    # extermeties of noise levels
    parser.add_argument("--noise_std", nargs="+", type=float, default=[-3, 2]) # range of noise levels to explore -- logarithmic to the base 10
    parser.add_argument("--noise_num_points", type=int, default=7) # number of points to sample in the noise level range

    # take in the comparator threshold
    parser.add_argument("--threshold", type=float, default=0.0)

    # # take in the noise standard deviation
    # parser.add_argument("--noise_std", type=float, default=1.0)

    # model architecture parameters
    parser.add_argument("--ff_layer_sizes", nargs="+", type=int, default=[1000, 500])

    # hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=int(5e4))
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--num_resamples", type=int, default=100) # resampling for statistics


    # #TODO: Add arguments for saving results and the model.
    # parser.add_argument("--plot_results", action='store_true')



    return parser.parse_args()

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

## prediction function
def pred_step(
        model: FFN, 
        inputs: jax.Array,
        labels: jax.Array,
        # metrics: nnx.MultiMetric
    ):

    logits = model(inputs)
    # print(logits)
    logits = nnx.softmax(logits, axis=-1)
    prediction = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(prediction == labels)

    return accuracy, prediction


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
        # checkpoint_every: int,
        # checkpoint_flag: bool = False,
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

        # # TODO: Add script to checkpoint the model
        # # Checkpointing the model
        # if checkpoint_flag and (step%checkpoint_every==0 or step==train_steps-1):
        #     file = f"ffn_uci_iris_hidden_date_{today}_{configs['layers'][1]}_noise_{configs['noise_std']}_threshold_{configs['threshold']}_step_{step}.pkl"
        #     graphdef, state = nnx.split(model)
        #     save_payload(state, configs, file)
        #     model = nnx.merge(graphdef, state)

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

        
            # print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")
    
    best_accuracy = max(metrics_history['eval_accuracy'])
    best_acc_idx = jnp.argmax(jnp.array(metrics_history['eval_accuracy'])).astype(int)
    eot_accuracy = metrics_history['eval_accuracy'][-1] # end of training accuracy
    print(f"Best Eval Accuracy: {best_accuracy:.4f} | Step: {metrics_history['step'][best_acc_idx]} | End of Training Accuracy: {eot_accuracy:.4f}")

    accuracies = {
        'best_accuracy': best_accuracy,
        'best_acc_step': metrics_history['step'][best_acc_idx],
        'eot_accuracy': eot_accuracy
    }

    return model, accuracies


# ------------------------------------
# Setting up training
# ------------------------------------
def main():

    # parse the input arguments
    args = parse_args()

    # data_dict to be saved
    data_dict = defaultdict(list)

    # load the data
    X_train, X_test, y_train, y_test = load_uci_iris(normalize=True, key=args.seed, train_test_split=0.7)

    # set up array of layer sizes to iterate over
    hl_range = args.hidden_layer_range
    hl_sizes = jnp.logspace(hl_range[0], hl_range[1], num=args.hl_num_points)
    print(f"HIDDEN LAYER SIZES: {hl_sizes[0].item(), hl_sizes[-1].item()}")

    # set up an array for noise levels
    noise_std_arr = jnp.logspace(args.noise_std[0], args.noise_std[1], num=args.noise_num_points)
    print(f"NOISE STD: {noise_std_arr[0]:.4f}, {noise_std_arr[-1]:.4f}")

    for noise_idx, noise_std in enumerate(noise_std_arr):
        for hl_idx, hl_size in enumerate(hl_sizes):
            configs = {
                'train_steps': args.train_steps,
                'eval_every': args.eval_every,
                'learning_rate': args.learning_rate,
                'threshold': args.threshold,
                'noise_std': noise_std.item(),
                'layers': [X_train.shape[1], int(jnp.floor(hl_size.item())), 3],
                'train_steps': args.train_steps,
                'seed': args.seed
            }

            seed = configs['seed']
            rngs = nnx.Rngs(
                params=seed + 0,
                dropout=seed + 1,
                activation=seed + 2,
                next = seed + 3 
            )


            # initialize the model
            model = FFN(
                layers=configs['layers'],
                noise_std=noise_std.item(),
                threshold=configs['threshold'],
                rngs=rngs
            )

            # compute the trainable parameters in the model
            num_params = nnx.state(model, nnx.Param)
            total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(num_params))
            # print(f"Total params = {total_params}")

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

            model, accuracies = train(
                model=model,
                optimizer=optimizer,
                train_inputs=X_train,
                train_labels=y_train,
                test_inputs=X_test,
                test_labels=y_test,
                metrics_history=metrics_history,
                metrics=metrics,
                configs=configs,
                # checkpoint_flag=args.checkpoint,
                # checkpoint_every=args.checkpoint_every
            )

            # compute SNR
            # compute preactivations for SNR calculation
            preactivations = model.layers[0](X_test)
            preactivations = preactivations.flatten()
            rms_preact = jnp.sqrt(jnp.mean(preactivations**2))

            # compute the SNR
            snr_train = 20*jnp.log10(rms_preact/noise_std)

            # enter the inference loop with resampling
            for r in tqdm(range(args.num_resamples)):
                seed = configs['seed'] + r + noise_idx*100 + hl_idx*200
                rngs = nnx.Rngs(
                    params=seed + 0,
                    dropout=seed + 1,
                    activation=seed + 2,
                    next = seed + 3 
                )

                model.rngs = rngs # update the model rngs for resampling

                # compute accuracy
                accuracy, prediction = pred_step(model, X_test, y_test)

                # compute sparsity
                sparsity = model.activation(model.layers[0](X_test))
                sparsity = jnp.mean(sparsity == 0.0)

                # dump everything into the data dictionary
                data_dict['total_params'].append(int(total_params))
                data_dict['accuracy'].append(accuracy.item())
                data_dict['sparsity'].append(sparsity.item())
                data_dict['training_snr'].append(snr_train.item())
                data_dict['noise_std'].append(noise_std.item())
                data_dict['resample_index'].append(r)

            mean_accuracy = sum(data_dict['accuracy'][-args.num_resamples:])/args.num_resamples
            mean_sparsity = sum(data_dict['sparsity'][-args.num_resamples:])/args.num_resamples
            print(f"Model size: {total_params} // Inference Accuracy: {mean_accuracy:.4f} | Sparsity: {mean_sparsity:.4f} | SNR: {snr_train:.4f}")

            
    # save the data dictionary into a pickle file
    metadata = {
        "__doc__": "Data for plotting variation of accuracy and sparsity of a network with different trainable parameters and noise levels.",
        "columns": {
            "total_params": "Total number of trainable parameters in the model",
            "accuracy": "Inference accuracy of the model on the test set",
            "sparsity": "Sparsity of the activations in the hidden layer, computed as the fraction of activations that are exactly zero",
            "training_snr": "Signal-to-noise ratio during training, computed as 20*log10(rms_preactivation/noise_std)",
            "noise_std": "Standard deviation of the noise added to the activations during training",
            "resample_index": "Index for resampling to compute statistics"
        },
        "training_steps": args.train_steps,
        "eval_every": args.eval_every,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "hidden_layer_range": args.hidden_layer_range,
        "noise_std_range": args.noise_std,
        "num_resamples": args.num_resamples,
        "hl_num_points": args.hl_num_points,
        "noise_num_points": args.noise_num_points
    }

    filename = os.path.join(DATA_PATH, f"ffn_uci_iris_scaling_snr_perf_sparsity_{today}.pkl")
    save_data_pkl(data_dict, metadata, filename)




if __name__ == "__main__":
    main()