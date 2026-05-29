"""
Training feedforward network on UCI-Iris + TriDENT using consistency regularization.
Consider the 32-hidden units network to being with,
-Modify the loss function as follows:
For a given input X_i, compute
y_1 = model(X_i. rngs1)
y_2 = model(X_i, rngs2)
total_loss = CE_loss(y_1 or y_2, labels) + alpha * MSE_loss(y_1, y_2)

Sweep over:
    - alpha is a hyperparameter
    - sweep over injected noise

NOTES: TODO

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



from utils import dual_sample_ternary, load_uci_iris
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
def save_payload(state, configs, filename, data = None):
    """
    Save model and parameters to a pickle.
    state: nnx.Model state
    configs: dict, configuration parameters.
    data: dict, optional data from simulations 
    """

    if data is None:
        payload = {
            'configs' : configs,
            'state': state
        }
    else:
        payload = {
            'configs' : configs,
            'data': data,
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
    parser = argparse.ArgumentParser(description="Consistency Regularization Trident-FFN on UCI-Iris")

    # take in the levels
    parser.add_argument("--levels", nargs="+", type=float, default=[-1.0, 0.0, 1.0])

    # take in the initial thresholds
    # parser.add_argument("--thresholds", nargs="+", type=float, default=[-1.0, 1.0])
    parser.add_argument("--threshold", type=float, default=0.0)

    # take in the noise standard deviation
    # parser.add_argument("--noise_std", type=float, default=1.0)

    # model architecture parameters
    # parser.add_argument("--ff_layer_sizes", nargs="+", type=int, default=[1000, 500])

    # hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=int(5e4))
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--seed_next", type=int, default=0)
    parser.add_argument("--checkpoint", action='store_true')
    # parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_test_split", type=float, default=0.7)


    #TODO: Add arguments for controling the hyperparameter
    # parser.add_argument("--alpha_var", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    # parser.add_argument("--noise_std", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])

    # TODO: test mode flag
    parser.add_argument("--test_mode", action='store_true')

    return parser.parse_args()

# ------------------------------------
# Training functions
# ------------------------------------
def loss_fn(
        model: FFN,
        # batch: dict,
        X: jax.Array,
        labels: jax.Array,
        alpha: float = 1e-3, # contribution of variance loss in total loss
        loss_function: Callable = optax.softmax_cross_entropy_with_integer_labels
    ):
    # two forwad passses through the model
    logits1 = model(X)
    logits2 = model(X)

    # using softmax cross-entropy with integer labels: use either logits1 or logits2
    ce_loss = loss_function(logits1, labels=labels).mean()

    # NEW: Consistency regularizer
    consistency_loss = jnp.mean((logits1 - logits2)**2)

    # compute total loss
    total_loss = ce_loss + alpha*consistency_loss

    return total_loss, [logits1, logits2]

        
# training step
@nnx.jit
def train_step(
    model: FFN,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    X: jax.Array, # input data
    label: jax.Array, # labels
    alpha: float = 1e-3, # regularizer for variance loss
    loss_fn: Callable = loss_fn
    ):

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, [logits1, logits2]), grads = grad_fn(model=model, X=X, labels=label, alpha=alpha)
    metrics.update(loss=loss, logits=logits1, labels=label)
    optimizer.update(model, grads)
    
# evaluation step
@nnx.jit
def eval_step(
    model: FFN,
    metrics: nnx.MultiMetric,
    X: jax.Array, # input data
    label: jax.Array, # labels
    alpha: float = 1e-3, # regularizer for variance loss
    loss_fn: Callable = loss_fn
 ):
    loss, [logits1, logits2] = loss_fn(model=model, X=X, labels=label, alpha=alpha)
    metrics.update(loss=loss, logits=logits1, labels=label)


# ------------------------------------
# Training Pipeline
# ------------------------------------
def train(
        train_inputs: jax.Array,
        train_labels: jax.Array,
        test_inputs: jax.Array,
        test_labels: jax.Array,
        configs: dict,
        checkpoint_flag: bool = False,
        test_mode: bool = False,
        **kwargs    
    ):

    """
    Training loop to sweep over input noise levels.
    Checkpoint models at the end of training for every noise level.
    """
    
    print("--"*50)
    print(f"Consistency Regularization Noise-alpha sweep: UCI Iris Dataset")
    print("--"*50)

    eval_every = configs['eval_every']
    train_steps = configs['train_steps']
    noise_std_arr = configs['noise_std_arr'] 
    alpha_arr = configs['alpha_arr']

    for alpha_idx, alpha in enumerate(tqdm(alpha_arr, total=len(alpha_arr))):
        for n_idx, noise_std in enumerate(noise_std_arr):
            print("--"*50)
            print(f"*** Noise Std: {noise_std:.4f} | Alpha: {alpha:.4f} ***")
            data = defaultdict(list) # dictionary to store training data for current noise level

            # DEFINE THE MODELS
            rngs = nnx.Rngs(
                params=configs['seed'] + 0,
                dropout=configs['seed'] + 1,
                activation=configs['seed'] + 2,
                next=configs['seed'] + 3
            )

            configs['rng_headers'] = ['params+0', 'dropout+1', 'activation+2', 'next+3']

            model = FFN(
                layers = configs['layers'],
                noise_std = noise_std.item(),
                threshold = configs['threshold'],
                ActivationFunction = DualSampleTernary,
                rngs = rngs
            )


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

            metrics = nnx.MultiMetric(
                accuracy=nnx.metrics.Accuracy(),
                loss=nnx.metrics.Average('loss')
            )

            metrics_history = defaultdict(list)

            for step in range(train_steps):

                # train the model
                train_step(model, optimizer, metrics, train_inputs, train_labels, alpha=alpha.item())

                # evaluate and checkpoint the model
                if step > 0 and (step%eval_every==0 or step == train_steps-1):
                    metrics_history['step'].append(step)

                    # log the training metrics
                    for metric, value in metrics.compute().items():
                        metrics_history[f"train_{metric}"].append(value.item())
                    metrics.reset()

                    # evaluate the model on validation set
                    eval_step(model, metrics, test_inputs, test_labels, alpha=alpha.item())

                    # log the evaluation metrics
                    for metric, value in metrics.compute().items():
                        metrics_history[f"eval_{metric}"].append(value.item())
                    metrics.reset()
                
                    print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")
            
            best_accuracy = max(metrics_history['eval_accuracy'])
            best_acc_idx = jnp.argmax(jnp.array(metrics_history['eval_accuracy'])).astype(int)
            print(f"Best Eval Accuracy: {best_accuracy:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

            ## TODO: determine checkpointing
            if checkpoint_flag:
                configs['noise_std'] = noise_std.item()

                # compute the SNR using hidden layer pre-activations
                act_hidden_layer = model.activation(model.layers[0](test_inputs))
                hidden_layer = model.layers[0](test_inputs) # should be a (150, 32) output

                # compute the rms across the dataset
                hidden_layer = hidden_layer.flatten()
                act_hidden_layer = act_hidden_layer.flatten()
                rms_pre_act = jnp.sqrt(jnp.mean(hidden_layer**2))
                snr = 20*jnp.log10(rms_pre_act/noise_std)
                training_sparsity = jnp.mean(act_hidden_layer == 0)
                print(f"Noise Std: {noise_std:.4f} | SNR: {snr:.2f} dB | Sparsity: {training_sparsity*100:.2f}%")

                # append SNR to data dictionary
                data['training_snr'] = snr.item()
                data['training_sparsity'] = training_sparsity.item()
                data['alpha'] = alpha.item()

                # save the model, data and configs
                filename = f"consistency_reg_snr_sweep_iris_{today}_alpha_{alpha:.4f}_noise_{noise_std:.4f}.pkl"
                graphdef_trident, state_trident = nnx.split(model)
                save_payload(state_trident, configs, filename, data)

            # if in test mode just return after the first checkpoint
            if test_mode:
                return model, metrics_history
            

    return model, metrics_history

# ------------------------------------
# Setting up training
# ------------------------------------
def main():

    # parse the input arguments
    args = parse_args()

    if args.test_mode:
        print("xx"*50)
        print("TEST MODE")
        print("xx"*50)

    # load the data
    X_train, X_test, y_train, y_test = load_uci_iris(normalize=True, key=args.seed, train_test_split=args.train_test_split)

    # noise std array
    noise_std_arr = jnp.logspace(-3, 3, 7)

    # alpha sweep
    alpha_arr = jnp.logspace(-4, 0, 5)

    configs = {
        'train_steps': args.train_steps,
        'eval_every': args.eval_every,
        'learning_rate': args.learning_rate,
        'threshold': args.threshold,
        'noise_std_arr': noise_std_arr,
        'train_test_split': args.train_test_split,
        'alpha_arr': alpha_arr,
        'layers': [X_train.shape[1], 32, 3],
        'seed': args.seed # also used for train test split
    }

    # print(args.test_mode) # is this true?
    # print(args.learning_rate)


    model, metrics_history = train(
        train_inputs=X_train,
        train_labels=y_train,
        test_inputs=X_test,
        test_labels=y_test,
        configs=configs,
        checkpoint_flag=args.checkpoint,
        test_mode=args.test_mode
    )

    # IN TEST MODE as a test, print out contents of a saved model
    if args.test_mode:
        fname = f"TEST_consistency_reg_snr_sweep_iris_{today}_alpha_{alpha_arr[0]:.4f}_noise_{noise_std_arr[0].item():.4f}.pkl"
        with open(os.path.join(MODEL_PATH, fname), "rb") as f:
            loaded_state = pickle.load(f)

        print(loaded_state['data'])
        print(loaded_state['configs'])
        print(loaded_state['state'])

        # remove the model file after loading
        os.remove(os.path.join(MODEL_PATH, fname))


if __name__ == "__main__":
    main()