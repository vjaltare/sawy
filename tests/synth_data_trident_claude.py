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

# parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic data analysis")

    parser.add_argument("--num_resamples", type=int, default=50, help="Number of resamples for each test")
    parser.add_argument("--num_datapoints", type=int, default=int(5e4))
    parser.add_argument("--int_window", type=int, default=2, help="Number of samples to average over. Need at least 2")
    parser.add_argument("--chunk_size", type=int, default=None, help="How many of int_window's samples IntegratedTrident materializes at once (must divide int_window). None = int_window (old one-shot behavior). Lower = less memory, more (sequential) scan steps -- use if you still OOM with mini-batching on.")
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
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training steps (sampled with replacement each step).")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Chunk size for iterating over the eval set (whole test set is covered, just in chunks of this size).")


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
            chunk_size: int | None = None,   # see IntegratedTrident -- None = old one-shot behavior
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
            chunk_size=chunk_size,
        )



    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.trident(
                layer(x)
            )

        x = self.layers[-1](x)
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
        verbose: bool = True,   # set False to silence prints/nnx.display -- used by sweep_gauss_nu_std
                                 # so a sweep of hundreds of runs doesn't flood stdout.
    ):

    # append num classes to layer sizes.
    # FIX: use [*layer_sizes, num_classes] instead of layer_sizes.append(...) --
    # .append mutates the CALLER's list in place. Since test_train() passes the
    # same base_layer_sizes list used elsewhere (or a copy of it), appending in
    # place either corrupts that shared list or -- if the call site already
    # included num_classes itself -- doubles it up (that's what produced
    # "[100, 2000, 2000, 10, 10]" you saw).
    layer_sizes = [*layer_sizes, num_classes]
    if verbose:
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
    if verbose:
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
    if verbose:
        nnx.display(TeacherNetwork)

    # pass the data through the inputs
    y = TeacherNetwork(X)

    # apply hardmax to find the winning classes
    int_labels = jnp.argmax(y, axis=-1)

    labels_train = int_labels[:split]
    labels_test = int_labels[split:]

    if verbose:
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
    # baked into the module at construction / mutated in place per-batch, see
    # make_train_step / make_eval_step below) and returns a per-sample loss
    # array -- reduce it to a scalar for value_and_grad.
    per_sample_loss = integrated_ce_loss(logits)
    loss = per_sample_loss.mean()

    return loss, logits


# ------------------------------------
# Mini-batching
# ------------------------------------
# Both factories below close over batch_size/eval_batch_size/num_classes as
# plain Python constants (rather than passing them as jitted-function
# arguments) since they determine array SHAPES -- jax.random.randint's shape
# and jax.nn.one_hot's num_classes both need to be static at trace time. This
# is also why train_step/eval_step are built via factories instead of being
# defined once at module scope: each run's batch_size/num_classes are only
# known once parse_args() has run.
#
# Training draws a fresh mini-batch WITH REPLACEMENT each step (no
# epoch/shuffle bookkeeping needed) via jax.random.fold_in(batch_key_base,
# step) -- batch_key_base is a plain jax.random.key, passed in as an
# ordinary (non-nnx, stateless) argument, and step is the plain Python loop
# counter from test_train()'s for-loop. Deliberately NOT an nnx.Rngs stream:
# using one shared with `model`'s own rngs (or int_ce_loss's) would trip the
# same "Inconsistent aliasing detected" error from before, since it would be
# reachable from two top-level jit arguments with different diff/non-diff
# treatment. A plain immutable key array sidesteps that entirely.
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


# Evaluation walks the WHOLE test set in contiguous chunks of eval_batch_size
# (trailing remainder smaller than one chunk is dropped for simplicity),
# calling eval_step once per chunk and letting nnx.MultiMetric accumulate --
# more stable than scoring a single random eval batch, and eval only runs
# every `eval_every` steps so the extra chunks are cheap.
def make_eval_step(eval_batch_size, num_classes):
    @nnx.jit
    def eval_step(
        model: nnx.Module,
        metrics: nnx.MultiMetric,
        int_ce_loss: nnx.Module,   # constructed with an eval_batch_size-shaped dummy `labels`
        X: jax.Array,               # one eval_batch_size-sized chunk
        label: jax.Array,           # matching int labels for that chunk
    ):
        int_ce_loss.labels = jax.nn.one_hot(label, num_classes)
        loss, logits = loss_fn(model, X, int_ce_loss)
        metrics.update(loss=loss, logits=logits, labels=label)
        return loss

    def eval_full(model, metrics, int_ce_loss, X_test, labels_test):
        n_chunks = X_test.shape[0] // eval_batch_size
        for i in range(n_chunks):
            sl = slice(i * eval_batch_size, (i + 1) * eval_batch_size)
            eval_step(model, metrics, int_ce_loss, X_test[sl], labels_test[sl])

    return eval_full

# ------------------------------------
# Train + evaluate a single (nu, std, seed, ...) configuration
# ------------------------------------
def train_and_evaluate(
        args,
        nu: int,
        std: float,
        gauss_noise_flag: bool,
        seed: int,
        verbose: bool = True,
    ):
    """
    Everything test_train() used to do inline, refactored into a reusable
    function of the swept hyperparameters (nu, std, gauss_noise_flag) plus a
    per-run seed. test_train() below is now a thin wrapper around this that
    preserves its old behavior (reading nu/std/gauss/seed from args, verbose
    prints). sweep_gauss_nu_std() calls this directly, once per (nu, std, r)
    cell, with verbose=False so a sweep of hundreds of runs doesn't flood
    stdout, and with its own per-cell seed instead of args.seed.

    Returns (best_train_acc, best_eval_acc) -- the best value logged for each
    over the whole training run (matching the max-eval-accuracy bookkeeping
    test_train() already did; best_train_acc is the train accuracy computed
    at that same step-index-of-best-eval, since a single scalar "best" for
    train accuracy alone would just be a noisier estimate of the same
    ceiling. See the sweep_gauss_nu_std() docstring for how these two numbers
    map onto the "best_test_acc, best_eval_acc" columns the TODO asked for.)
    """
    if verbose:
        print("--"*50)
        print(f"Test Training | Trident Synthetic Dataset | nu={nu} std={std} gauss={gauss_noise_flag} seed={seed}")
        print("--"*50)

    # rngs for data generation (input_key) + the STUDENT model (params + trident's
    # noise). Deliberately NOT shared with the loss's rngs below -- nnx.jit/grad
    # reject the same stateful rng node being reachable from two different
    # top-level arguments (model vs. int_ce_loss) with inconsistent diff/non-diff
    # treatment ("Inconsistent aliasing detected").
    rngs = nnx.Rngs(default=0, params=seed, input_key=seed + 1)

    # Base architecture shared by teacher & student (input -> hidden -> hidden).
    # Kept WITHOUT num_classes here -- make_synthetic_data appends num_classes
    # itself, and StudentMLP needs its own copy WITH num_classes appended for
    # its output layer (student_layer_sizes below). Passing one list that
    # already had num_classes appended into make_synthetic_data (which then
    # appends it again) is what produced the duplicated "..., 10, 10]".
    base_layer_sizes = [10, 1000]

    # generate synthetic data -- teacher is built INSIDE make_synthetic_data
    # from teacher_params_key (see the NOTE in that function). Note the
    # TEACHER's key stays args.teacher_seed regardless of `seed` -- the
    # ground-truth mapping is meant to stay fixed across a sweep, only the
    # student's init/training/data-draw randomness (rngs, which depends on
    # `seed` via input_key) should vary from cell to cell / resample to resample.
    X_train, X_test, labels_train, labels_test = make_synthetic_data(
            num_datapoints=args.num_datapoints,
            train_test_split=0.8,
            num_classes=args.num_classes,
            rngs=rngs,
            teacher_params_key=args.teacher_seed,
            layer_sizes=list(base_layer_sizes),
            verbose=verbose,
        )

    student_layer_sizes = [*base_layer_sizes, args.num_classes]
    student_model = StudentMLP(
        layer_sizes=student_layer_sizes,
        rngs=rngs,
        nu=nu,
        noise_std=std,
        noise_mean=args.mean,
        gauss_noise_flag=gauss_noise_flag,
        chunk_size=args.chunk_size,
    )

    # IntegratedCELoss needs ONE-HOT labels of shape (Batch, Classes). Rather
    # than the full train/test split's labels, each is now constructed with a
    # BATCH-SIZED dummy (all-zero) placeholder -- train_step/eval_step
    # overwrite `.labels` with the real batch's one-hot labels every call, but
    # keeping the shape fixed at construction time (batch_size / eval_batch_size
    # respectively) avoids any in/out pytree-shape mismatch under nnx.jit.
    # Needs a SEPARATE instance per split since train and eval use different
    # batch sizes (and therefore different `.labels` shapes).
    #
    # separate rng stream for the loss's own noise -- see the note on `rngs` above
    loss_rngs = nnx.Rngs(default=seed + 2)
    int_ce_loss_train = IntegratedCELoss(
        rngs=loss_rngs,
        nu=nu,
        noise_std=std,
        noise_mean=args.mean,
        gauss_noise_flag=gauss_noise_flag,
        labels=jnp.zeros((args.batch_size, args.num_classes)),
    )
    int_ce_loss_test = IntegratedCELoss(
        rngs=loss_rngs,
        nu=nu,
        noise_std=std,
        noise_mean=args.mean,
        gauss_noise_flag=gauss_noise_flag,
        labels=jnp.zeros((args.eval_batch_size, args.num_classes)),
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

    train_step = make_train_step(args.batch_size, args.num_classes)
    eval_full = make_eval_step(args.eval_batch_size, args.num_classes)

    # plain, stateless key for mini-batch sampling -- see the note above
    # make_train_step for why this isn't an nnx.Rngs stream.
    batch_key_base = jax.random.key(seed + 3)

    step_iter = tqdm(range(args.train_steps)) if verbose else range(args.train_steps)
    for step in step_iter:

        # train the model on a random mini-batch (sampled with replacement)
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

            # evaluate over the WHOLE test set, in eval_batch_size chunks
            # (int_ce_loss_test is bound to the TEST split via each chunk's
            # labels, mutated in place per chunk inside eval_full/eval_step).
            eval_full(student_model, metrics, int_ce_loss_test, X_test, labels_test)

            # log the evaluation metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(value.item())
            metrics.reset()

            if verbose:
                print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")

    best_eval_accuracy = max(metrics_history['eval_accuracy'])
    best_acc_idx = int(jnp.argmax(jnp.array(metrics_history['eval_accuracy'])))
    best_train_accuracy = metrics_history['train_accuracy'][best_acc_idx]
    if verbose:
        print(f"Best Eval Accuracy: {best_eval_accuracy:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

    return best_train_accuracy, best_eval_accuracy


# ------------------------------------
# Test Training (thin wrapper around train_and_evaluate, using args directly)
# ------------------------------------
def test_train():
    args = parse_args()
    return train_and_evaluate(
        args,
        nu=args.int_window,
        std=args.std,
        gauss_noise_flag=args.gaussian_noise,
        seed=args.seed,
        verbose=True,
    )


# ------------------------------------
# Sweep over (nu, std) for gaussian noise, with resampling
# ------------------------------------
def sweep_gauss_nu_std():
    """
    Sweeps IntegratedTrident's integration window (nu) and noise std, using
    gaussian noise (gauss_noise_flag=True -- logistic can follow later as its
    own sweep_logistic_nu_std() twin, not built here since it wasn't asked
    for). For every (nu, std) cell, trains args.num_resamples independent
    students (different seed each time, so different param init AND -- since
    the data draw is coupled to `rngs`/seed via input_key in
    train_and_evaluate -- a different draw of the synthetic dataset too) and
    records that run's best train/eval accuracy.

    Per your TODOs:
      (a) seed = args.seed + nu_idx + s_idx + r for every run; train and
          record best train + eval accuracy for every config.
      (b) build a dict with columns: nu, std, best_test_acc, best_eval_acc,
          resample_index (r).
      (c) return the dict.

    NOTE on "best_test_acc" vs "best_eval_acc": this codebase only has one
    held-out split (X_test/labels_test), scored via int_ce_loss_test during
    training and logged into metrics_history as "eval_accuracy" -- there's no
    separate third split called "test" distinct from "eval". So both columns
    below are populated from the SAME best-eval-accuracy number
    (best_test_acc is an alias of best_eval_acc), to match the literal column
    names in the TODO without inventing a third data split you didn't ask
    for. best_train_acc (not requested by name in the TODO, but recorded
    since (a) asks for it) is also included for reference.

    The TEACHER stays fixed at args.teacher_seed across the entire sweep --
    only the student's init/training and the data draw vary with `seed` --
    so every cell is approximating the SAME ground-truth mapping.
    """
    args = parse_args()

    # nu (integration window) and std (noise scale) grids to sweep.
    # 7 nu values x 4 std values x args.num_resamples resamples (default 50)
    # = 1400 full training runs by default, each args.train_steps steps
    # (default 1e4) -- ~14 MILLION total training steps. Pass smaller
    # --train_steps / --num_resamples (and/or a coarser NU_ARR/STD_ARR below)
    # for an initial exploratory pass before scaling up.
    NU_ARR = jnp.array([2, 10, 20, 100, 200, 1000, 2000])
    STD_ARR = jnp.logspace(-3, 0, 4, base=10)
    NUM_RESAMPLES = args.num_resamples

    n_runs = len(NU_ARR) * len(STD_ARR) * NUM_RESAMPLES
    print("--"*50)
    print(f"Sweep (gauss): {len(NU_ARR)} nu x {len(STD_ARR)} std x {NUM_RESAMPLES} resamples "
          f"= {n_runs} runs x {args.train_steps} train_steps each "
          f"= {n_runs * args.train_steps:,} total training steps")
    print("--"*50)

    data = defaultdict(list)

    for nu_idx, nu in enumerate(NU_ARR):
        for s_idx, std in enumerate(STD_ARR):
            for r in range(NUM_RESAMPLES):
                seed = args.seed + nu_idx + s_idx + r

                best_train_acc, best_eval_acc = train_and_evaluate(
                    args,
                    nu=int(nu),
                    std=float(std),
                    gauss_noise_flag=True,
                    seed=seed,
                    verbose=False,
                )

                data['nu'].append(int(nu))
                data['std'].append(float(std))
                data['best_test_acc'].append(best_eval_acc)
                data['best_eval_acc'].append(best_eval_acc)
                data['best_train_acc'].append(best_train_acc)
                data['resample_index'].append(r)

                print(f"[{len(data['nu'])}/{n_runs}] nu={nu} std={std} r={r} seed={seed} "
                      f"-> best_train_acc={best_train_acc:.4f} best_eval_acc={best_eval_acc:.4f}")

    if args.save_results:
        os.makedirs(DATA_PATH, exist_ok=True)
        out_path = os.path.join(DATA_PATH, f"sweep_gauss_nu_std_{today}.csv")
        pd.DataFrame(dict(data)).to_csv(out_path, index=False)
        print(f"Saved sweep results to {out_path}")

    return dict(data)




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

    test_train_script = True
    if test_train_script:
        test_train()

    run_sweep = False
    if run_sweep:
        sweep_gauss_nu_std()

if __name__ == "__main__":
    main()