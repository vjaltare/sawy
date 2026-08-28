"""
Testing trident functions -- sampling activation and softmax on synthetic datasets.

Parameters to sweep:
- noise std (optional but might be a good idea)
- integration window (nu)
- number of classes (num_classes)

* Teacher Network
- assign an explicit teacher_key to initialize the parameters of the teacher FFN!

"""

import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import glob

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



from utils import dual_sample_ternary, load_uci_iris, dual_sample_binary_softmax, generate_gaussian_noise, generate_logistic_noise
from models import IntegratedCELoss, IntegratedTrident

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"


# -----------------------------
# SAVE DATA
# ----------------------------- 
def save_payload(data, filename, configs):
    """
    Save model and parameters to a pickle.
    configs: dict, configuration parameters.
    data: dict, optional data from simulations 
    """

    payload = {
        'configs': configs,
        'data': data
    }

    # if data is None:
    #     payload = {
    #         'configs' : configs,
    #         'state': state
    #     }
    # else:
    #     payload = {
    #         'configs' : configs,
    #         'data': data,
    #         'state': state
    #     }

    checkpoint_dir = "/local_disk/vikrant/trident/logs"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {filename_}")




# parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic data analysis")

    parser.add_argument("--num_resamples", type=int, default=50, help="Number of resamples for each test")
    parser.add_argument("--num_datapoints", type=int, default=int(5e4))
    parser.add_argument("--int_window", type=int, default=2, help="Number of samples to average over. Need at least 2")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes or output neurons.")
    parser.add_argument("--max_token_size", type=int, default=1024, help="Max length to sweep classes over.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for default rng stream")
    parser.add_argument("--teacher_seed", type=int, default=999, help="Random seed for teacher params rng stream")
    parser.add_argument("--preactivation_scale", type=float, default=1.0, help="Scale of preactivation noise") #  z/scale before passing it to softmax
    parser.add_argument("--use_sgd", action="store_true")
    parser.add_argument("--use_adamw", action="store_true")

    parser.add_argument("--std", type=float, default=1.0, help="Noise scale parameter")
    parser.add_argument("--mean", type=float, default=0.0, help="Noise location parameter")

    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=int(1e4))
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--base_layer_sizes", nargs="+", type=int, default=[10, 1000])


    parser.add_argument("--gaussian_noise", action="store_true", help="whether to use gaussian noise")
    # parser.add_argument("--logistic_noise", action="store_true", help="whether to use logistic noise")
    parser.add_argument("--save_results", action="store_true", help="whether to save the results")

    # parser.add_argument("--run_softmax_pipeline", action="store_true", help="run softmax pipeline")
    # parser.add_argument("--run_jacobian_pipeline", action="store_true", help="run jacobian pipeline")

    return parser.parse_args()

# -------------------------------------------------
# Auxiliary functions
# -------------------------------------------------
def sigmoid(x, loc, scale):
    return 1/(1 + jnp.exp(-(x - loc)/scale))

def expected_state_logistic(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float

    ):

    Ey = 2*sigmoid(x=x-threshold, scale=std, loc=mean) - 1
    return Ey

# -------------------------------------------------
# Defining teacher network
# -------------------------------------------------
class TeacherMLP(nnx.Module):
    def __init__(self,
                 layer_sizes: list[int],
                 rngs: nnx.Rngs,

                 ):

        self.layer_sizes = nnx.List([
            nnx.Linear(in_features=l_in, out_features=l_out, rngs=rngs)
            for l_in, l_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ])

        self.activation_function = partial(expected_state_logistic, threshold=0.0, std=1.0, mean=0.0)

    def __call__(self, x):
        for layer in self.layer_sizes[:-1]:
            x = layer(x)
            x = self.activation_function(x)

        x = nnx.softmax(self.layer_sizes[-1](x))

        return x

# -------------------------------------------------
# Defining student network
# -------------------------------------------------
class StudentMLP(nnx.Module):
    """
    Student MLP. Uses trident activations.
    Trained on the data from Teacher MLP
    """
    def __init__(
            self,
            rngs: nnx.Rngs,
            layer_sizes: list[int],
            nu: int,            # trident integration window
            noise_std: float,
            noise_mean: float,
            gauss_noise_flag: bool,
        ):

        self.layers = nnx.List([
                    nnx.Linear(in_features=l_in, out_features=l_out, rngs=rngs)
                    for l_in, l_out in zip(layer_sizes[:-1], layer_sizes[1:])
                ])


        self.trident = IntegratedTrident(
            rngs=rngs,
            nu=nu,
            noise_mean=noise_mean,
            noise_std=noise_std,
            gauss_noise_flag=gauss_noise_flag,
        )

        self.batch_norm = nnx.BatchNorm(num_features=layer_sizes[-1], rngs=rngs)


    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.trident(
                layer(x)
            )

        x = self.layers[-1](x)
        # x = self.batch_norm(x)
        return x

# -------------------------------------------------
# Make synthetic data
# -------------------------------------------------
def make_synthetic_data(
        num_datapoints: int, # no. datapoints in the dataset (train+test)
        train_test_split: float, # fraction of training samples in the total dataset
        num_classes: int, # no. output classes
        rngs: nnx.Rngs,
        teacher_params_key: int,
        layer_sizes: list,
        TeacherModel: nnx.Module = TeacherMLP,
    ):

    # append num classes to layer sizes.
    # FIX: use [*layer_sizes, num_classes] instead of layer_sizes.append(...) --
    # .append mutates the CALLER's list in place. Since test_train() passes the
    # same base_layer_sizes list used elsewhere (or a copy of it), appending in
    # place either corrupts that shared list or -- if the call site already
    # included num_classes itself -- doubles it up (that's what produced
    # "[100, 2000, 2000, 10, 10]" you saw).
    layer_sizes = [*layer_sizes, num_classes]
    print(f"layer_sizes: {layer_sizes}")

    # generate inputs
    input_size = layer_sizes[0]
    X = jax.random.normal(rngs.input_key(), shape=(num_datapoints, input_size)) # keep input key fized between teacher and student

    split = int(jnp.floor(train_test_split*num_datapoints))
    # FIX: was X[..., :split] / X[..., split:], which slices the FEATURE axis
    # (input_size), not the sample axis -- that's what produced the
    # X_test.shape[-1] == 0 you saw ("X_train.shape = (50000, 100), X_test.shape = (50000, 0)").
    # Sample axis is axis 0.
    X_train, X_test = X[:split], X[split:]
    print(f"X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}")

    # # initialize the teacher network
    # NOTE: TeacherModel must be a CLASS (constructor), not an already-built
    # instance -- this line calls TeacherModel(layer_sizes=..., rngs=...) to
    # CONSTRUCT the teacher. If a pre-built TeacherMLP instance is passed in
    # for this argument instead, this becomes `instance(layer_sizes=..., rngs=...)`,
    # i.e. it invokes the instance's __call__(self, x) with those as kwargs --
    # which is exactly the
    #   "TeacherMLP.__call__() got an unexpected keyword argument 'layer_sizes'"
    # TypeError. Don't pre-construct a TeacherMLP and pass it as TeacherModel;
    # let this function build it from teacher_params_key (that's the whole
    # point of the explicit teacher_key design noted at the top of this file).
    teacher_rngs = nnx.Rngs(params=teacher_params_key)
    TeacherNetwork = TeacherModel(layer_sizes=layer_sizes, rngs=teacher_rngs)
    # nnx.display(TeacherNetwork)

    # pass the data through the inputs
    y = TeacherNetwork(X)

    # apply hardmax to find the winning classes
    int_labels = jnp.argmax(y, axis=-1)

    labels_train = int_labels[:split]
    labels_test = int_labels[split:]

    print(f"Train labels shape = {labels_train.shape}, Test labels shape = {labels_test.shape}")
    print(f"Train labels = {labels_train[:10]}")

    return X_train, X_test, labels_train, labels_test

# ------------------------------------
# Training functions
# ------------------------------------
def loss_fn(
        model: nnx.Module,
        # batch: dict,
        X: jax.Array,
        integrated_ce_loss: nnx.Module
    ):
    # forwad pass through the model
    logits = model(X)

    # integrated_ce_loss expects preactivations + ONE-HOT labels (labels are
    # baked into the module at construction, see test_train() below) and
    # returns a per-sample loss array -- reduce it to a scalar for value_and_grad.
    per_sample_loss = integrated_ce_loss(logits)
    loss = per_sample_loss.mean()

    return loss, logits


# training step
# @nnx.jit
# def train_step(
#     model: nnx.Module,
#     optimizer: nnx.Optimizer,
#     metrics: nnx.MultiMetric,
#     X: jax.Array, # input data
#     label: jax.Array, # labels
#     integrated_ce_loss: nnx.Module,
#     loss_fn: Callable = loss_fn
#     ):

#     grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
#     (loss, logits), grads = grad_fn(model, X, integrated_ce_loss)
#     metrics.update(loss=loss, logits=logits, labels=label)
#     optimizer.update(model, grads)

# evaluation step
@nnx.jit
def eval_step(
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    int_ce_loss: nnx.Module,
    X: jax.Array, # input data
    label: jax.Array, # labels
 ):
    loss, logits = loss_fn(model, X, int_ce_loss)
    metrics.update(loss=loss, logits=logits, labels=label)
    return loss

# FOR BATCHED MODE TRAINING
def make_train_step(batch_size, num_classes):
    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        X: jax.Array,               # FULL training set -- indexed into per-batch below
        label: jax.Array,           # FULL training int labels
        integrated_ce_loss: nnx.Module,   # constructed with a batch_size-shaped dummy `labels` (see test_train)
        batch_key_base: jax.Array,
        step: jax.Array,
    ):
        batch_key = jax.random.fold_in(batch_key_base, step)
        idx = jax.random.randint(batch_key, (batch_size,), 0, X.shape[0])
        X_batch, label_batch = X[idx], label[idx]
 
        # mutate the loss module's labels in place for this step -- same shape
        # every time (batch_size, num_classes), so no jit recompilation and no
        # in/out pytree-shape mismatch across calls.
        integrated_ce_loss.labels = jax.nn.one_hot(label_batch, num_classes)
 
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, X_batch, integrated_ce_loss)
        metrics.update(loss=loss, logits=logits, labels=label_batch)
        optimizer.update(model, grads)
 
    return train_step

# ------------------------------------
# Test Training
# ------------------------------------
def test_train():
    print("--"*50)
    print(f"Test Training | Trident Synthetic Dataset")
    print("--"*50)

    args = parse_args()

    # rngs for data generation (input_key) + the STUDENT model (params + trident's
    # noise). Deliberately NOT shared with the loss's rngs below -- nnx.jit/grad
    # reject the same stateful rng node being reachable from two different
    # top-level arguments (model vs. int_ce_loss) with inconsistent diff/non-diff
    # treatment ("Inconsistent aliasing detected").
    rngs = nnx.Rngs(default=0, params=args.seed, input_key=args.seed + 1)

    # Base architecture shared by teacher & student (input -> hidden -> hidden).
    # Kept WITHOUT num_classes here -- make_synthetic_data appends num_classes
    # itself, and StudentMLP needs its own copy WITH num_classes appended for
    # its output layer (student_layer_sizes below). Passing one list that
    # already had num_classes appended into make_synthetic_data (which then
    # appends it again) is what produced the duplicated "..., 10, 10]".
    base_layer_sizes = args.base_layer_sizes
    print(f"BASE LAYER SIZES: {base_layer_sizes}")

    # generate synthetic data -- teacher is built INSIDE make_synthetic_data
    # from teacher_params_key (see the NOTE in that function).
    X_train, X_test, labels_train, labels_test = make_synthetic_data(
            num_datapoints=args.num_datapoints,
            train_test_split=0.8,
            num_classes=args.num_classes,
            rngs=rngs,
            teacher_params_key=args.teacher_seed,
            layer_sizes=list(base_layer_sizes),
        )

    student_layer_sizes = [*base_layer_sizes, args.num_classes]
    student_model = StudentMLP(
        layer_sizes=student_layer_sizes,
        rngs=rngs,
        nu=args.int_window,
        noise_std=args.std,
        noise_mean=args.mean,
        gauss_noise_flag=args.gaussian_noise
    )

    # IntegratedCELoss needs ONE-HOT labels of shape (Batch, Classes), not the
    # integer labels make_synthetic_data returns -- and needs a SEPARATE
    # instance per split (train vs. test), since labels are bound at
    # construction time rather than passed to __call__.
    labels_train_oh = jax.nn.one_hot(labels_train, args.num_classes)
    labels_test_oh = jax.nn.one_hot(labels_test, args.num_classes)

    # separate rng stream for the loss's own noise -- see the note on `rngs` above
    loss_rngs = nnx.Rngs(default=args.seed + 2)
    int_ce_loss_train = IntegratedCELoss(
        rngs=loss_rngs,
        nu=args.int_window,
        noise_std=args.std,
        noise_mean=args.mean,
        gauss_noise_flag=args.gaussian_noise,
        labels=labels_train_oh
    )
    int_ce_loss_test = IntegratedCELoss(
        rngs=loss_rngs,
        nu=args.int_window,
        noise_std=args.std,
        noise_mean=args.mean,
        gauss_noise_flag=args.gaussian_noise,
        labels=labels_test_oh
    )


    adamw_optimizer = nnx.Optimizer(
            student_model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    learning_rate=args.learning_rate,
                    weight_decay=1e-4
                )
            ),
            wrt=nnx.Param
            )
    
    sgd_optimizer = nnx.Optimizer(
        student_model,
        optax.chain(
            optax.sgd(
                learning_rate=args.learning_rate,
                momentum=0.9
            )
        ),
        wrt=nnx.Param
    )

    optimizer = sgd_optimizer if args.use_sgd else adamw_optimizer

    metrics = nnx.MultiMetric(
                accuracy=nnx.metrics.Accuracy(),
                loss=nnx.metrics.Average('loss')
            )

    metrics_history = defaultdict(list)
    train_steps = args.train_steps
    eval_every = args.eval_every

    train_step = make_train_step(batch_size=args.batch_size, num_classes=args.num_classes)
    batch_key_base = jax.random.key(args.seed + 3)

    for step in tqdm(range(args.train_steps)):

        # train the model
        # FIX: X, label, and integrated_ce_loss are required arguments (no
        # defaults) -- the original call omitted all three.
        train_step(
            model=student_model, optimizer=optimizer, metrics=metrics,
            X=X_train, label=labels_train, integrated_ce_loss=int_ce_loss_train, 
            batch_key_base=batch_key_base, step=jnp.asarray(step),

        )

        # evaluate and checkpoint the model
        if step > 0 and (step%eval_every==0 or step == train_steps-1):
            metrics_history['step'].append(step)

            # log the training metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value.item())
            metrics.reset()

            # evaluate the model on validation set
            # FIX: same missing-arguments issue, plus this now uses
            # int_ce_loss_test (bound to the TEST split's one-hot labels)
            # rather than reusing the train-bound loss module.
            eval_step(model=student_model, metrics=metrics, int_ce_loss=int_ce_loss_test,
                      X=X_test, label=labels_test)

            # log the evaluation metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(value.item())
            metrics.reset()

            print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")

    best_accuracy = max(metrics_history['eval_accuracy'])
    best_acc_idx = jnp.argmax(jnp.array(metrics_history['eval_accuracy'])).astype(int)
    print(f"Best Eval Accuracy: {best_accuracy:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

    return metrics_history, student_model


# -----------------------------------------
# Sweep nu and std
# -----------------------------------------
def sweep_gauss_nu_std():
    print("--"*50)
    print(f"Sweeping Nu and STD | Trident Synthetic Dataset")
    print("--"*50)

    args = parse_args() # NOTE: ignore the int_window and std parameters passed here as we'll be sweeping over them!

    # initialize the arrays to sweep over
    NU_ARR = jnp.array([2, 20, 200, 2000])
    STD_ARR = jnp.logspace(-3, 0, 4, base=10)
    NUM_RESAMPLES = args.num_resamples

    print(f"NU SWEEP: {NU_ARR}")
    print(f"STD SWEEP: {STD_ARR}")

    if args.gaussian_noise:
        print("** GAUSSIAN NOISE SELECTED **")

    # data storage
    data = defaultdict(list)

    # configs for the run
    configs = {
        'nu_sweep': NU_ARR.tolist(),
        'std_sweep': STD_ARR.tolist(),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'resamples': args.num_resamples,
        'layer_sizes': args.base_layer_sizes,
        'adamw_used': args.use_adamw,
        'noise_dist': "gaussian",
        'learning_rate': args.learning_rate,

    }

    for nu_idx, nu in tqdm(enumerate(NU_ARR), total=len(NU_ARR)):
        
        for s_idx, gauss_std in enumerate(STD_ARR):
            
            for r in range(NUM_RESAMPLES):
                    print(f"NU = {nu} // {nu_idx + 1}/{len(NU_ARR)}")
                    print(f"STD = {gauss_std} // {s_idx + 1}/{len(STD_ARR)}")
                    print(f"RESAMPLING... {r+1}/{NUM_RESAMPLES}")

                # TODO: train the network for changing seed -> seed + n_idx + s_idx + r + args.seed for every run. Record best train and eval accuracies for all configs.
                # TODO: Make a dict: data: columns -> nu, std, best_test_acc, best_eval_acc, resample_index (r)
                # TODO: Return the data dictionary 

                # rngs for data generation (input_key) + the STUDENT model (params + trident's
                    # noise). Deliberately NOT shared with the loss's rngs below -- nnx.jit/grad
                    # reject the same stateful rng node being reachable from two different
                    # top-level arguments (model vs. int_ce_loss) with inconsistent diff/non-diff
                    # treatment ("Inconsistent aliasing detected").
                    rngs = nnx.Rngs(default=0 + nu_idx + s_idx + r, params=args.seed + nu_idx + s_idx + r, input_key=args.seed + 10 + nu_idx + s_idx + r)
                
                    # Base architecture shared by teacher & student (input -> hidden -> hidden).
                    # Kept WITHOUT num_classes here -- make_synthetic_data appends num_classes
                    # itself, and StudentMLP needs its own copy WITH num_classes appended for
                    # its output layer (student_layer_sizes below). Passing one list that
                    # already had num_classes appended into make_synthetic_data (which then
                    # appends it again) is what produced the duplicated "..., 10, 10]".
                    base_layer_sizes = args.base_layer_sizes
                    print(f"BASE LAYER SIZES: {base_layer_sizes}")
                
                    # generate synthetic data -- teacher is built INSIDE make_synthetic_data
                    # from teacher_params_key (see the NOTE in that function).
                    X_train, X_test, labels_train, labels_test = make_synthetic_data(
                            num_datapoints=args.num_datapoints,
                            train_test_split=0.8,
                            num_classes=args.num_classes,
                            rngs=rngs,
                            teacher_params_key=args.teacher_seed,
                            layer_sizes=list(base_layer_sizes),
                        )
                
                    student_layer_sizes = [*base_layer_sizes, args.num_classes]
                    student_model = StudentMLP(
                        layer_sizes=student_layer_sizes,
                        rngs=rngs,
                        nu=nu.item(),
                        noise_std=gauss_std.item(),
                        noise_mean=args.mean,
                        gauss_noise_flag=args.gaussian_noise
                    )
                
                    # IntegratedCELoss needs ONE-HOT labels of shape (Batch, Classes), not the
                    # integer labels make_synthetic_data returns -- and needs a SEPARATE
                    # instance per split (train vs. test), since labels are bound at
                    # construction time rather than passed to __call__.
                    labels_train_oh = jax.nn.one_hot(labels_train, args.num_classes)
                    labels_test_oh = jax.nn.one_hot(labels_test, args.num_classes)
                
                    # separate rng stream for the loss's own noise -- see the note on `rngs` above
                    loss_rngs = nnx.Rngs(default=args.seed + 2)
                    int_ce_loss_train = IntegratedCELoss(
                        rngs=loss_rngs,
                        nu=nu.item(),
                        noise_std=gauss_std.item(),
                        noise_mean=args.mean,
                        gauss_noise_flag=args.gaussian_noise,
                        labels=labels_train_oh
                    )
                    int_ce_loss_test = IntegratedCELoss(
                        rngs=loss_rngs,
                        nu=nu.item(),
                        noise_std=gauss_std.item(),
                        noise_mean=args.mean,
                        gauss_noise_flag=args.gaussian_noise,
                        labels=labels_test_oh
                    )
                
                
                    adamw_optimizer = nnx.Optimizer(
                            student_model,
                            optax.chain(
                                optax.clip_by_global_norm(1.0),
                                optax.adamw(
                                    learning_rate=args.learning_rate,
                                    weight_decay=1e-4
                                )
                            ),
                            wrt=nnx.Param
                            )
                    
                    sgd_optimizer = nnx.Optimizer(
                        student_model,
                        optax.chain(
                            optax.sgd(
                                learning_rate=args.learning_rate,
                                momentum=0.9
                            )
                        ),
                        wrt=nnx.Param
                    )
                
                    optimizer = sgd_optimizer if args.use_sgd else adamw_optimizer
                
                    metrics = nnx.MultiMetric(
                                accuracy=nnx.metrics.Accuracy(),
                                loss=nnx.metrics.Average('loss')
                            )
                
                    metrics_history = defaultdict(list)
                    train_steps = args.train_steps
                    eval_every = args.eval_every
                
                    train_step = make_train_step(batch_size=args.batch_size, num_classes=args.num_classes)
                    batch_key_base = jax.random.key(args.seed + 3 + nu_idx + s_idx + r)
                
                    for step in tqdm(range(args.train_steps)):
                
                        # train the model
                        # FIX: X, label, and integrated_ce_loss are required arguments (no
                        # defaults) -- the original call omitted all three.
                        train_step(
                            model=student_model, optimizer=optimizer, metrics=metrics,
                            X=X_train, label=labels_train, integrated_ce_loss=int_ce_loss_train, 
                            batch_key_base=batch_key_base, step=jnp.asarray(step),
                
                        )
                
                        # evaluate and checkpoint the model
                        if step > 0 and (step%eval_every==0 or step == train_steps-1):
                            metrics_history['step'].append(step)
                
                            # log the training metrics
                            for metric, value in metrics.compute().items():
                                metrics_history[f"train_{metric}"].append(value.item())
                            metrics.reset()
                
                            # evaluate the model on validation set
                            # FIX: same missing-arguments issue, plus this now uses
                            # int_ce_loss_test (bound to the TEST split's one-hot labels)
                            # rather than reusing the train-bound loss module.
                            eval_step(model=student_model, metrics=metrics, int_ce_loss=int_ce_loss_test,
                                      X=X_test, label=labels_test)
                
                            # log the evaluation metrics
                            for metric, value in metrics.compute().items():
                                metrics_history[f"eval_{metric}"].append(value.item())
                            metrics.reset()
                
                            print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")
                
                    best_accuracy = max(metrics_history['eval_accuracy'])
                    best_train_acc = max(metrics_history['train_accuracy'])
                    best_acc_idx = jnp.argmax(jnp.array(metrics_history['eval_accuracy'])).astype(int)
                    print(f"Best Eval Accuracy: {best_accuracy:.4f} | Best Train Accuract: {best_train_acc:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

                    data["nu"].append(nu.item())
                    data["std"].append(gauss_std.item())
                    data["best_train_acc"].append(best_train_acc)
                    data["best_eval_acc"].append(best_accuracy)

    return data, configs


# -----------------------------------------
# Sweep nu and std: logistic
# -----------------------------------------
def sweep_log_nu_std():
    print("--"*50)
    print(f"Sweeping Nu and STD - Logistic noise | Trident Synthetic Dataset")
    print("--"*50)

    args = parse_args() # NOTE: ignore the int_window and std parameters passed here as we'll be sweeping over them!

    if not args.gaussian_noise:
        print(f"** LOGISTIC NOISE SELECTED **")

    # initialize the arrays to sweep over
    NU_ARR = jnp.array([2, 20, 200, 2000])
    STD_ARR = jnp.logspace(-3, 0, 4, base=10)
    NUM_RESAMPLES = args.num_resamples

    print(f"NU SWEEP: {NU_ARR}")
    print(f"STD SWEEP: {STD_ARR}")

    # data storage
    data = defaultdict(list)

    # configs for the run
    configs = {
        'nu_sweep': NU_ARR.tolist(),
        'std_sweep': STD_ARR.tolist(),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'resamples': args.num_resamples,
        'layer_sizes': args.base_layer_sizes,
        'adamw_used': args.use_adamw,
        'noise_dist': "logistic",
        'learning_rate': args.learning_rate,

    }

    for nu_idx, nu in tqdm(enumerate(NU_ARR), total=len(NU_ARR)):
        print(f"NU = {nu} // {nu_idx + 1}/{len(NU_ARR)}")
        for s_idx, std in enumerate(STD_ARR):
            print(f"STD = {std} // {s_idx + 1}/{len(STD_ARR)}")
            for r in range(NUM_RESAMPLES):
                    print(f"RESAMPLING... {r+1}/{NUM_RESAMPLES}")

                # TODO: train the network for changing seed -> seed + n_idx + s_idx + r + args.seed for every run. Record best train and eval accuracies for all configs.
                # TODO: Make a dict: data: columns -> nu, std, best_test_acc, best_eval_acc, resample_index (r)
                # TODO: Return the data dictionary 

                # rngs for data generation (input_key) + the STUDENT model (params + trident's
                    # noise). Deliberately NOT shared with the loss's rngs below -- nnx.jit/grad
                    # reject the same stateful rng node being reachable from two different
                    # top-level arguments (model vs. int_ce_loss) with inconsistent diff/non-diff
                    # treatment ("Inconsistent aliasing detected").
                    rngs = nnx.Rngs(default=0 + nu_idx + s_idx + r, params=args.seed + nu_idx + s_idx + r, input_key=args.seed + 10 + nu_idx + s_idx + r)
                
                    # Base architecture shared by teacher & student (input -> hidden -> hidden).
                    # Kept WITHOUT num_classes here -- make_synthetic_data appends num_classes
                    # itself, and StudentMLP needs its own copy WITH num_classes appended for
                    # its output layer (student_layer_sizes below). Passing one list that
                    # already had num_classes appended into make_synthetic_data (which then
                    # appends it again) is what produced the duplicated "..., 10, 10]".
                    base_layer_sizes = args.base_layer_sizes
                    print(f"BASE LAYER SIZES: {base_layer_sizes}")
                
                    # generate synthetic data -- teacher is built INSIDE make_synthetic_data
                    # from teacher_params_key (see the NOTE in that function).
                    X_train, X_test, labels_train, labels_test = make_synthetic_data(
                            num_datapoints=args.num_datapoints,
                            train_test_split=0.8,
                            num_classes=args.num_classes,
                            rngs=rngs,
                            teacher_params_key=args.teacher_seed,
                            layer_sizes=list(base_layer_sizes),
                        )
                
                    student_layer_sizes = [*base_layer_sizes, args.num_classes]
                    student_model = StudentMLP(
                        layer_sizes=student_layer_sizes,
                        rngs=rngs,
                        nu=nu.item(),
                        noise_std=std.item(),
                        noise_mean=args.mean,
                        gauss_noise_flag=args.gaussian_noise
                    )
                
                    # IntegratedCELoss needs ONE-HOT labels of shape (Batch, Classes), not the
                    # integer labels make_synthetic_data returns -- and needs a SEPARATE
                    # instance per split (train vs. test), since labels are bound at
                    # construction time rather than passed to __call__.
                    labels_train_oh = jax.nn.one_hot(labels_train, args.num_classes)
                    labels_test_oh = jax.nn.one_hot(labels_test, args.num_classes)
                
                    # separate rng stream for the loss's own noise -- see the note on `rngs` above
                    loss_rngs = nnx.Rngs(default=args.seed + 2)
                    int_ce_loss_train = IntegratedCELoss(
                        rngs=loss_rngs,
                        nu=nu.item(),
                        noise_std=std.item(),
                        noise_mean=args.mean,
                        gauss_noise_flag=args.gaussian_noise,
                        labels=labels_train_oh
                    )
                    int_ce_loss_test = IntegratedCELoss(
                        rngs=loss_rngs,
                        nu=nu.item(),
                        noise_std=std.item(),
                        noise_mean=args.mean,
                        gauss_noise_flag=args.gaussian_noise,
                        labels=labels_test_oh
                    )
                
                
                    adamw_optimizer = nnx.Optimizer(
                            student_model,
                            optax.chain(
                                optax.clip_by_global_norm(1.0),
                                optax.adamw(
                                    learning_rate=args.learning_rate,
                                    weight_decay=1e-4
                                )
                            ),
                            wrt=nnx.Param
                            )
                    
                    sgd_optimizer = nnx.Optimizer(
                        student_model,
                        optax.chain(
                            optax.sgd(
                                learning_rate=args.learning_rate,
                                momentum=0.9
                            )
                        ),
                        wrt=nnx.Param
                    )
                
                    optimizer = sgd_optimizer if args.use_sgd else adamw_optimizer
                
                    metrics = nnx.MultiMetric(
                                accuracy=nnx.metrics.Accuracy(),
                                loss=nnx.metrics.Average('loss')
                            )
                
                    metrics_history = defaultdict(list)
                    train_steps = args.train_steps
                    eval_every = args.eval_every
                
                    train_step = make_train_step(batch_size=args.batch_size, num_classes=args.num_classes)
                    batch_key_base = jax.random.key(args.seed + 3 + nu_idx + s_idx + r)
                
                    for step in tqdm(range(args.train_steps)):
                
                        # train the model
                        # FIX: X, label, and integrated_ce_loss are required arguments (no
                        # defaults) -- the original call omitted all three.
                        train_step(
                            model=student_model, optimizer=optimizer, metrics=metrics,
                            X=X_train, label=labels_train, integrated_ce_loss=int_ce_loss_train, 
                            batch_key_base=batch_key_base, step=jnp.asarray(step),
                
                        )
                
                        # evaluate and checkpoint the model
                        if step > 0 and (step%eval_every==0 or step == train_steps-1):
                            metrics_history['step'].append(step)
                
                            # log the training metrics
                            for metric, value in metrics.compute().items():
                                metrics_history[f"train_{metric}"].append(value.item())
                            metrics.reset()
                
                            # evaluate the model on validation set
                            # FIX: same missing-arguments issue, plus this now uses
                            # int_ce_loss_test (bound to the TEST split's one-hot labels)
                            # rather than reusing the train-bound loss module.
                            eval_step(model=student_model, metrics=metrics, int_ce_loss=int_ce_loss_test,
                                      X=X_test, label=labels_test)
                
                            # log the evaluation metrics
                            for metric, value in metrics.compute().items():
                                metrics_history[f"eval_{metric}"].append(value.item())
                            metrics.reset()
                
                            print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")
                
                    best_accuracy = max(metrics_history['eval_accuracy'])
                    best_train_acc = max(metrics_history['train_accuracy'])
                    best_acc_idx = jnp.argmax(jnp.array(metrics_history['eval_accuracy'])).astype(int)
                    print(f"Best Eval Accuracy: {best_accuracy:.4f} | Best Train Accuract: {best_train_acc:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

                    data["nu"].append(nu.item())
                    data["std"].append(std.item())
                    data["best_train_acc"].append(best_train_acc)
                    data["best_eval_acc"].append(best_accuracy)

    return data, configs


    # return metrics_history, student_model



# -------------------------------------------------
# Testing
# -------------------------------------------------
def main():
    test_make_data = False
    if test_make_data:
        rngs = nnx.Rngs(default=0, key=1, input_key=2, params=3)
        layers = [200, 2000, 2000]
        num_classes = 10
        num_datapoints = 50000
        train_test_split = 0.8
        X_train, X_test, labels_train, labels_test = make_synthetic_data(num_classes=num_classes, num_datapoints=num_datapoints, train_test_split=train_test_split, rngs=rngs, layer_sizes=layers, teacher_params_key=999)
        print(f"Input shape = {X_train.shape}")

    test_train_script = False
    if test_train_script:
        metrics_history, model = test_train()

    run_sweep_nu_std_gauss = True #TODO: rerun!
    if run_sweep_nu_std_gauss:
        data, configs = sweep_gauss_nu_std()
        filename = f"nu_std_sweep_gauss_synthetic_{today}.pkl"
        save_payload(data=data, configs=configs, filename=filename)

    run_sweep_nu_std_logistic = False
    if run_sweep_nu_std_logistic:
        data, configs = sweep_log_nu_std()
        filename = f"nu_std_sweep_log_synthetic_{today}.pkl"
        save_payload(data=data, configs=configs, filename=filename)

if __name__ == "__main__":
    main()