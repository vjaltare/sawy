"""
Comparing theoretical vs approximate gradients for TriDENT.
- Train separate models with DuapSampleTernary and DualSampleTernaryExact
- Record gradients computed by exact and approx methods.
- To save space, compute cosine similarity over epochs and only save that instead of the full gradients.

Notes:
TODO
- Store the gradient norm!
- UPDATE: script is updated to include norms for trident and exact gradients
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
from models import TernaryStochasticActivation, DualSampleTernary, FFN, DualSampleTernaryExact

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

    checkpoint_dir = DATA_PATH
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
    parser.add_argument("--noise_std", type=float, default=1e-2)

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
    parser.add_argument("--num_resamples", type=int, default=30) # number of resamples to compute the average gradients for trident


    #TODO: Add arguments for controling the hyperparameter
    # parser.add_argument("--alpha_var", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    # parser.add_argument("--noise_std", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])

    # TODO: test mode flag
    parser.add_argument("--test_mode", action='store_true')
    parser.add_argument("--store_grads", action='store_true')
    parser.add_argument("--store_cosine", action='store_true')

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
    (loss, logits), grads = grad_fn(model=model, X=X, labels=label)
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
    loss, logits = loss_fn(model=model, X=X, labels=label)
    metrics.update(loss=loss, logits=logits, labels=label)

# record gradients for hidden layer
def compute_grads(
        model: FFN,
        X: jax.Array,
        label: jax.Array,
        loss_fn: Callable = loss_fn
    ):

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model=model, X=X, labels=label)
    # print(grads) # replace this with recording script

    return grads




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
        store_grads: bool = False,
        store_cosine: bool = False,
        **kwargs    
    ):

    """
    Training loop to sweep over input noise levels.
    Checkpoint models at the end of training for every noise level.
    """
    
    print("--"*50)
    print(f"Noise Performance Training: UCI Iris Dataset")
    print("--"*50)

    eval_every = configs['eval_every']
    train_steps = configs['train_steps']
    noise_std = configs['noise_std'] # use 0.1 or 0.01 for best results, 0.0 for no noise
    num_resamples = configs['num_resamples']


    # for n_idx, noise_std in enumerate(noise_std_arr):
    #     print(f"Training with noise std: {noise_std:.4f} | Noise level {n_idx+1}/{len(noise_std_arr)}")
    #     print("--"*50)
    #     data = defaultdict(list) # dictionary to store training data for current noise level

    # DEFINE THE MODELS
    rngs_trident = nnx.Rngs(
        params=configs['seed'] + 0,
        dropout=configs['seed'] + 1,
        activation=configs['seed'] + 2,
        next=configs['seed'] + 3
    )

    rngs_exact = nnx.Rngs(
        params=configs['seed'] + 0,
        dropout=configs['seed'] + 1,
        activation=configs['seed'] + 2,
        next=configs['seed'] + 3
    )

    configs['rng_headers'] = ['params+0', 'dropout+1', 'activation+2', 'next+3']

    model_trident = FFN(
        layers = configs['layers'],
        noise_std = noise_std,
        threshold = configs['threshold'],
        ActivationFunction = DualSampleTernary,
        rngs = rngs_trident
    )

    model_trident_exact = FFN(
        layers = configs['layers'],
        noise_std = noise_std,
        threshold = configs['threshold'],
        ActivationFunction = DualSampleTernaryExact,
        rngs = rngs_exact
    )


    optimizer_trident = nnx.Optimizer(
    model_trident,
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=configs['learning_rate'],
            weight_decay=1e-4
        )
    ),
    wrt=nnx.Param
    )

    metrics_trident = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss')
    )

    optimizer_exact = nnx.Optimizer(
    model_trident_exact,
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=configs['learning_rate'],
            weight_decay=1e-4
        )
    ),
    wrt=nnx.Param
    )

    metrics_exact = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss')
    )

    metrics_history_trident = defaultdict(list)
    metrics_history_exact = defaultdict(list)
    gradients_history = defaultdict(list)
    cosine_sims = defaultdict(list) # combine -> steps, cosine similarity, accuracy.

    for step in range(train_steps):


        # train the models
        train_step(model_trident, optimizer_trident, metrics_trident, train_inputs, train_labels)
        train_step(model_trident_exact, optimizer_exact, metrics_exact, train_inputs, train_labels)


        # evaluate and checkpoint the models
        if not test_mode:
            if step >= 0 and (step%eval_every==0 or step == train_steps-1):

                ## append the cosine similarities

                # list to store gradients
                trident_grads_list = None

                # resample over same datum to compute expected gradient for trident 
                for r in range(num_resamples): # TODO: not resampling helps! Remove this loop. Or only resample once
                    grads_trident = compute_grads(model_trident, train_inputs, train_labels)
                    if trident_grads_list is None:
                        trident_grads_list = grads_trident['layers'][0]['kernel'].flatten()
                    else:
                        trident_grads_list = trident_grads_list + grads_trident['layers'][0]['kernel'].flatten()

                # average over the number of samples to get expected gradient
                trident_grads_list = trident_grads_list/num_resamples

                # print(f"Step {step} | TriDENT Gradients (resampled :5): {trident_grads_list[:5]}")
                grads_trident = jnp.array(trident_grads_list)

                grads_exact = compute_grads(model_trident_exact, train_inputs, train_labels)
                # print(f"Step {step} | Exact Gradients (:5): {grads_exact['layers'][0]['kernel'].flatten().tolist()[:5]}")
                grads_exact = grads_exact['layers'][0]['kernel'].flatten()

                cosine_sim = jnp.dot(grads_trident, grads_exact) / (jnp.linalg.norm(grads_trident) * jnp.linalg.norm(grads_exact) + 1e-8)
                # print(f"Step {step} | Cosine sim: {cosine_sim.item()}")

                # append to the cosine similarity history
                cosine_sims['step'].append(step)
                cosine_sims['cosine_similarity'].append(cosine_sim.item())

                # compute the gradient norms
                trident_grad_norm = jnp.linalg.norm(grads_trident)
                exact_grad_norm = jnp.linalg.norm(grads_exact)
                cosine_sims['trident_grad_norm'].append(trident_grad_norm.item())
                cosine_sims['exact_grad_norm'].append(exact_grad_norm.item())


                ## compute the metrics
                print(f"STEP {step}  | COSINE SIMILARITY: {cosine_sim.item():.4f}")
                print(f"STEP {step}  | TRIDENT GRAD NORM: {trident_grad_norm.item():.4f} | EXACT GRAD NORM: {exact_grad_norm.item():.4f}")
                metrics_history_trident['step'].append(step)
                metrics_history_exact['step'].append(step)

                # log the training metrics
                for metric, value in metrics_trident.compute().items():
                    metrics_history_trident[f"train_{metric}"].append(value.item())
                metrics_trident.reset()

                for metric, value in metrics_exact.compute().items():
                    metrics_history_exact[f"train_{metric}"].append(value.item())
                metrics_exact.reset()

                # evaluate the models on validation set
                eval_step(model_trident, metrics_trident, test_inputs, test_labels)
                eval_step(model_trident_exact, metrics_exact, test_inputs, test_labels)

                # log the evaluation metrics
                for metric, value in metrics_trident.compute().items():
                    metrics_history_trident[f"eval_{metric}"].append(value.item())
                metrics_trident.reset()

                for metric, value in metrics_exact.compute().items():
                    metrics_history_exact[f"eval_{metric}"].append(value.item())
                metrics_exact.reset()

                # append accuracy to cosine similarity history
                cosine_sims["inference_accuracy_trident"].append(metrics_history_trident['eval_accuracy'][-1])
                cosine_sims["inference_accuracy_exact"].append(metrics_history_exact['eval_accuracy'][-1])
            
                print(f"Step {step}/{train_steps} | Train Accuracy (Trident): {metrics_history_trident['train_accuracy'][-1]:.4f} | Eval Accuracy (Trident): {metrics_history_trident['eval_accuracy'][-1]:.4f} | Train Accuracy (Exact): {metrics_history_exact['train_accuracy'][-1]:.4f} | Eval Accuracy (Exact): {metrics_history_exact['eval_accuracy'][-1]:.4f}")
        
    best_accuracy_trident = max(metrics_history_trident['eval_accuracy'])
    best_accuracy_exact = max(metrics_history_exact['eval_accuracy'])
    best_acc_idx_trident = jnp.argmax(jnp.array(metrics_history_trident['eval_accuracy'])).astype(int)
    best_acc_idx_exact = jnp.argmax(jnp.array(metrics_history_exact['eval_accuracy'])).astype(int)
    print(f"Best Eval Accuracy (Trident): {best_accuracy_trident:.4f} | Step: {metrics_history_trident['step'][best_acc_idx_trident]}")
    print(f"Best Eval Accuracy (Exact): {best_accuracy_exact:.4f} | Step: {metrics_history_exact['step'][best_acc_idx_exact]}")

    # storing gradients
    if store_grads:
        if test_mode:
            filename = f"TEST_gradients_accuracy_comparison_uci_iris_{today}_noise_{noise_std:.3f}.pkl"
        else:
            # filename = f"gradients_comparison_uci_iris_{today}_noise_{noise_std:.3f}.pkl" # for raw gradients
            filename = f"grads_cosine_similarity_accuracy_comparison_uci_iris_{today}_noise_{noise_std:.3f}.pkl" #for cosine similarities

        save_payload(state=gradients_history, configs=configs, filename=filename)

    # storing cosine similarities
    if store_cosine:
        if test_mode:
            filename = f"TEST_cosine_similarity_accuracy_comparison_uci_iris_{today}_noise_{noise_std:.3f}.pkl"
        else:
            filename = f"cosine_similarity_accuracy_comparison_uci_iris_{today}_noise_{noise_std:.3f}.pkl"

            
        save_payload(state=cosine_sims, configs=configs, filename=filename)

        # plotting the gradients
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(cosine_sims['inference_accuracy_trident'], cosine_sims['cosine_similarity'], label='Trident Biased', alpha=0.5, lw=0, marker='o', markersize=3, color="C0")
        ax[0].plot(cosine_sims['inference_accuracy_exact'], cosine_sims['cosine_similarity'], label='Exact Unbiased', alpha=0.5, lw=0, marker='o', markersize=3, color="C1")
        ax[0].set_xlabel("Inference Accuracy")
        ax[0].set_ylabel("Cosine Similarity")
        ax[0].legend()
        ax[1].hist(cosine_sims['cosine_similarity'], bins=10, alpha=0.7)
        ax[1].set_xlabel("Cosine Similarity")
        ax[1].set_ylabel("Frequency")
        plt.tight_layout() 
        plt.savefig(f"../plots/tmp_grads_comp_n_{noise_std:.3f}.png")
        plt.show()


 
        # ## Save checkpoints: model and metadata
        # if checkpoint_flag:
        #     configs['noise_std'] = noise_std.item()

        #     if noise_std == 0.0:
        #         snr = 999 # placeholder for infinite SNR
        #         data['training_snr'] = snr

        #         # prepare to save the model with metadata
        #         filename = f"noise_perf_comp_uci_iris_{today}_noise_{noise_std:.3f}.pkl"
        #         graphdef, state = nnx.split(model)
        #         save_payload(state, configs, filename, data)
        #     else:
        #         # compute the SNR using hidden layer pre-activations
        #         hidden_layer_preact = model.layers[0](test_inputs)
        #         rms_preact = jnp.sqrt(jnp.mean(hidden_layer_preact**2))
        #         snr = 20*jnp.log10(rms_preact/noise_std)
        #         data['training_snr'] = snr.item()

        #         # prepare to save the model with metadata
        #         filename = f"noise_perf_comp_uci_iris_{today}_noise_{noise_std:.3f}.pkl"
        #         graphdef, state = nnx.split(model)
        #         save_payload(state, configs, filename, data)


        # # if in test mode just return after the first checkpoint
        # if test_mode:
        #     return model, metrics_history
            

    return model_trident, model_trident_exact, metrics_history_trident, metrics_history_exact, gradients_history


# ------------------------------------
# Setting up training
# ------------------------------------
def main():
    # parse the input arguments
    args = parse_args()

    # load the data
    X_train, X_test, y_train, y_test = load_uci_iris(normalize=True, key=args.seed, train_test_split=args.train_test_split)

    # set up noise array
    # noise_std_arr = jnp.append(jnp.array([0.0]), jnp.logspace(-2, -1, num=2)) # let's keep both 0.01 and 0.1
    # noise_std_arr = jnp.array([0.1])

    configs = {
        'train_steps': args.train_steps,
        'eval_every': args.eval_every,
        'learning_rate': args.learning_rate,
        'threshold': args.threshold,
        'noise_std': args.noise_std,
        'layers': [X_train.shape[1], 32, 3],
        'num_resamples': args.num_resamples,
        'seed': args.seed
    }

    model_trident, model_trident_exact, metrics_history_trident, metrics_history_exact, gradients_history = train(
        train_inputs=X_train,
        train_labels=y_train,
        test_inputs=X_test,
        test_labels=y_test,
        configs=configs,
        checkpoint_flag=args.checkpoint,
        test_mode=args.test_mode,
        store_grads=args.store_grads,
        store_cosine=args.store_cosine
    )

    # IN TEST MODE as a test, print out contents of a saved model
    # if args.test_mode:
    #     fname = f"TEST_noise_perf_comp_uci_iris_{today}_noise_{configs['noise_std_arr'][0]:.3f}.pkl"
    #     with open(os.path.join(MODEL_PATH, fname), "rb") as f:
    #         loaded_state = pickle.load(f)

    #     print(loaded_state['data'])
    #     print(loaded_state['configs'])
    #     print(loaded_state['state'])


if __name__ == "__main__":
    main()