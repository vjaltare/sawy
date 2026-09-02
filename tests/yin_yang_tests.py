"""
Train a small MLP with Trident activations + integrated CE loss on the
Yin-Yang dataset (see yin_yang_dataset.py, delivered alongside this script).

This replaces the earlier teacher/student synthetic-data setup: the
Yin-Yang labels come from a fixed, hand-designed geometric rule
(yin_yang_dataset.classify_point), not from a random teacher FFN. So
TeacherMLP / make_synthetic_data / teacher_seed all go away -- everything
else (StudentMLP, IntegratedTrident, IntegratedCELoss, train_step/eval_step,
save_payload, the tqdm training loop) follows the same structure as the
reference test_train().

Network depth is now a CLI knob: --num_hidden_layers and --hidden_layer_size
build

    layer_sizes = [2] + [hidden_layer_size] * num_hidden_layers + [3]

(2 input features, 3 output classes -- yin / yang / dot), e.g.
--num_hidden_layers 2 --hidden_layer_size 64 gives [2, 64, 64, 3].

Because the input is 2D, test_train() also renders and saves the trained
network's decision boundary (via yin_yang_dataset.plot_decision_boundary)
so you can see directly what adding depth buys you.

IntegratedTrident / IntegratedCELoss are imported unchanged from your
models.py, exactly as in the reference script:

    from models import IntegratedCELoss, IntegratedTrident
"""
import os
import copy
import argparse
import pickle
from collections import defaultdict
from datetime import date

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import IntegratedCELoss, IntegratedTrident
from utils import get_train_test, plot_decision_boundary

today = date.today().isoformat()

# Paths for saving results: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"


# -----------------------------
# SAVE DATA
# -----------------------------
def save_payload(data, filename, configs):
    """
    Save training results + configs to a pickle. Mirrors the reference
    script's save_payload().
    """
    payload = {
        'configs': configs,
        'data': data,
    }
    checkpoint_dir = DATA_PATH
    filename_ = os.path.join(checkpoint_dir, filename)
    os.makedirs(os.path.dirname(filename_), exist_ok=True)
    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Model saved to {filename_}")


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Yin-Yang Trident MLP training")

    # dataset
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--r_small", type=float, default=0.1, help="Yin-Yang eye-dot radius")
    parser.add_argument("--r_big", type=float, default=0.5, help="Yin-Yang outer disk radius")
    parser.add_argument("--num_classes", type=int, default=3, help="yin, yang, dot -- fixed for this dataset")
    parser.add_argument("--num_resamples", type=int, default=5)

    # architecture -- controls network depth/width
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--hidden_layer_size", type=int, default=64, help="Width of each hidden layer")

    # trident / loss
    parser.add_argument("--int_window", type=int, default=2, help="Trident/loss integration window (nu). Need at least 2")
    parser.add_argument("--std", type=float, default=1.0, help="Noise scale parameter")
    parser.add_argument("--mean", type=float, default=0.0, help="Noise location parameter")
    parser.add_argument("--gaussian_noise", action="store_true", help="whether to use gaussian noise (else logistic)")
    # parser.add_argument("--num_resamples", type=int, default=5, help="No. resamples for nu-hl sweep")

    # optimization
    parser.add_argument("--use_sgd", action="store_true")
    parser.add_argument("--use_adamw", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=int(5e3))
    parser.add_argument("--eval_every", type=int, default=100)

    # misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed for default rng stream")
    parser.add_argument("--save_results", action="store_true", help="whether to save the results pickle")
    parser.add_argument("--plot_boundary", action="store_true", default=True,
                         help="plot + save the decision boundary at the end of training")
    parser.add_argument("--no_plot_boundary", dest="plot_boundary", action="store_false")
    parser.add_argument("--boundary_resolution", type=int, default=300)

    # preactivation storage
    parser.add_argument("--preact_data_split", type=str, default="train", choices=["train", "test", "both"],
                         help="which split's inputs to run through the trained network for stored "
                              "preactivations / layer-wise mutual information") # MI not yet implemented

    # choose the pipeline to run
    parser.add_argument("--run_train", action="store_true")
    parser.add_argument("--run_nu_hl_sweep", action="store_true")
    parser.add_argument("--run_store_preactivations", action="store_true")
    parser.add_argument("--run_boundaries", action="store_true")

    return parser.parse_args()


# -------------------------------------------------
# Student network (Trident MLP) -- unchanged from the reference script
# -------------------------------------------------
class StudentMLP(nnx.Module):
    """
    Student MLP. Uses trident activations.
    Trained directly on the (geometrically-labeled) Yin-Yang dataset --
    there's no teacher network here.
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
        self.batch_norm = nnx.BatchNorm(num_features=layer_sizes[-1], rngs=rngs) # if needed!

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.trident(layer(x))
        x = self.layers[-1](x)
        # x = self.batch_norm(x)
        return x


def build_layer_sizes(num_hidden_layers: int, hidden_layer_size: int, in_features: int = 2, out_features: int = 3):
    """
    [2] + [hidden_layer_size] * num_hidden_layers + [3] -- this is the
    knob for network depth requested via --num_hidden_layers /
    --hidden_layer_size. num_hidden_layers=0 gives a plain linear model
    ([in_features, out_features]) with no Trident activation applied
    (there's no hidden layer to attach it to).
    """
    return [in_features] + [hidden_layer_size] * num_hidden_layers + [out_features]


def get_preactivations(model: "StudentMLP", x: jax.Array):
    """
    Run x through a trained StudentMLP, capturing the PREACTIVATION at each
    hidden layer -- i.e. the output of each nnx.Linear, BEFORE the Trident
    activation is applied to it. Doesn't require any change to StudentMLP
    itself; just re-walks model.layers/model.trident the same way
    StudentMLP.__call__ does, but keeps every intermediate around.
 
    Returns (preacts, logits):
      preacts: {'hl1': array of shape (N, hidden_layer_size), 'hl2': ..., ...}
               one entry per hidden layer (empty dict if num_hidden_layers=0
               -- a plain linear model has no hidden layer to record).
      logits:  the model's usual final-layer output, shape (N, num_classes).
    """
    preacts = {}
    for i, layer in enumerate(model.layers[:-1]):
        x = layer(x)
        preacts[f"hl{i + 1}"] = x
        x = model.trident(x)
    logits = model.layers[-1](x)
    return preacts, logits

def compute_decision_boundary_grid(predict_fn, resolution=300):
    """
    Evaluate predict_fn on a dense grid over [0, 1]^2.
 
    predict_fn: callable mapping an (N, 2) float32 array of grid points to
    an (N,) array of predicted integer classes in {0, 1, 2}.
 
    Returns (xx, yy, preds), each a (resolution, resolution) float/int array.
    """
    xx, yy = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    preds = np.asarray(predict_fn(grid)).reshape(xx.shape)
    return xx, yy, preds


# ------------------------------------
# Training functions -- unchanged pattern from the reference script
# ------------------------------------
def loss_fn(
        model: nnx.Module,
        X: jax.Array,
        integrated_ce_loss: nnx.Module
    ):
    logits = model(X)
    # integrated_ce_loss expects preactivations + ONE-HOT labels (labels are
    # baked into the module at construction) and returns a per-sample loss
    # array -- reduce it to a scalar for value_and_grad.
    per_sample_loss = integrated_ce_loss(logits)
    loss = per_sample_loss.mean()
    return loss, logits


@nnx.jit
def eval_step(
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    int_ce_loss: nnx.Module,
    X: jax.Array,
    label: jax.Array,
 ):
    loss, logits = loss_fn(model, X, int_ce_loss)
    metrics.update(loss=loss, logits=logits, labels=label)
    return loss


def make_train_step(batch_size, num_classes):
    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        X: jax.Array,               # FULL training set -- indexed into per-batch below
        label: jax.Array,           # FULL training int labels
        integrated_ce_loss: nnx.Module,   # constructed with a batch_size-shaped dummy `labels`
        batch_key_base: jax.Array,
        step: jax.Array,
    ):
        batch_key = jax.random.fold_in(batch_key_base, step)
        idx = jax.random.randint(batch_key, (batch_size,), 0, X.shape[0])
        X_batch, label_batch = X[idx], label[idx]

        # mutate the loss module's labels in place for this step -- same
        # shape every time (batch_size, num_classes), so no recompilation.
        integrated_ce_loss.labels = jax.nn.one_hot(label_batch, num_classes)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, X_batch, integrated_ce_loss)
        metrics.update(loss=loss, logits=logits, labels=label_batch)
        optimizer.update(model, grads)

    return train_step


# --------------------------------------------------------------
# Test Training
# --------------------------------------------------------------
def train(args=None):
    print("--" * 50)
    print("Test Training | Trident Yin-Yang Dataset")
    print("--" * 50)

    if args is None:
        args = parse_args()

    # rngs for the STUDENT model (params + trident's noise). Deliberately
    # NOT shared with the loss's rngs below -- nnx.jit/grad reject the same
    # stateful rng node being reachable from two different top-level
    # arguments (model vs. int_ce_loss) with inconsistent diff/non-diff
    # treatment ("Inconsistent aliasing detected").
    rngs = nnx.Rngs(default=0, params=args.seed, input_key=args.seed + 1)

    layer_sizes = build_layer_sizes(
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_size=args.hidden_layer_size,
        in_features=2,
        out_features=args.num_classes,
    )
    print(f"LAYER SIZES: {layer_sizes}")

    # Yin-Yang data: fixed geometric labels, no teacher network involved
    # (unlike make_synthetic_data in the reference script).
    X_train_np, labels_train_np, X_test_np, labels_test_np = get_train_test(
        n_train=args.n_train, n_test=args.n_test,
        r_small=args.r_small, r_big=args.r_big,
        seed=args.seed,
    )
    X_train, labels_train = jnp.asarray(X_train_np), jnp.asarray(labels_train_np)
    X_test, labels_test = jnp.asarray(X_test_np), jnp.asarray(labels_test_np)
    print(f"X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}")

    student_model = StudentMLP(
        layer_sizes=layer_sizes,
        rngs=rngs,
        nu=args.int_window,
        noise_std=args.std,
        noise_mean=args.mean,
        gauss_noise_flag=args.gaussian_noise,
    )

    # IntegratedCELoss needs ONE-HOT labels of shape (Batch, Classes), not
    # the integer labels get_train_test returns -- and needs a SEPARATE
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
                    weight_decay=args.weight_decay
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

    for step in tqdm(range(train_steps)):

        train_step(
            model=student_model, optimizer=optimizer, metrics=metrics,
            X=X_train, label=labels_train, integrated_ce_loss=int_ce_loss_train,
            batch_key_base=batch_key_base, step=jnp.asarray(step),
        )

        # evaluate and log
        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
            metrics_history['step'].append(step)

            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value.item())
            metrics.reset()

            eval_step(model=student_model, metrics=metrics, int_ce_loss=int_ce_loss_test,
                      X=X_test, label=labels_test)

            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(value.item())
            metrics.reset()

            print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")

    best_accuracy = max(metrics_history['eval_accuracy'])
    best_acc_idx = int(np.argmax(metrics_history['eval_accuracy']))
    print(f"Best Eval Accuracy: {best_accuracy:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

    if args.plot_boundary:
        def predict_fn(grid):
            logits = student_model(jnp.asarray(grid, dtype=jnp.float32))
            return np.asarray(jnp.argmax(logits, axis=-1))

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_decision_boundary(
            predict_fn, X=X_test_np, y=labels_test_np, ax=ax,
            title=f"nu={args.int_window} | layers={layer_sizes} | eval_acc={best_accuracy:.3f}",
        )
        fig.tight_layout()
        os.makedirs(FIGURES_PATH, exist_ok=True)
        fig_path = os.path.join(
            FIGURES_PATH,
            f"yin_yang_boundary_nu{args.int_window}_L{args.num_hidden_layers}x{args.hidden_layer_size}_sd_{args.std}_{today}.png",
        )
        fig.savefig(fig_path, dpi=150)
        print(f"Decision boundary saved to {fig_path}")
        plt.close(fig)

    if args.save_results:
        configs = {
            'seed': args.seed,
            'layer_sizes': layer_sizes,
            'num_hidden_layers': args.num_hidden_layers,
            'hidden_layer_size': args.hidden_layer_size,
            'nu': args.int_window,
            'noise_std': args.std,
            'noise_mean': args.mean,
            'noise_dist': "gaussian" if args.gaussian_noise else "logistic",
            'adamw_used': not args.use_sgd,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'train_steps': args.train_steps,
            'n_train': args.n_train,
            'n_test': args.n_test,
        }
        filename = f"yin_yang_train_{today}.pkl"
        save_payload(data=metrics_history, configs=configs, filename=filename)

    return metrics_history, student_model

# --------------------------------------------------------------
# Sweep no. hidden layers + int_window
# --------------------------------------------------------------
def sweep_hl_nu(args=None):
    """
    Define an array of no. hidden layers:
    n_hl = [1, 2, 3, 4, 5]
 
    Define an array of integration windows (nu/int_window)
    nu_arr = [2, 20, 200, 2000]
 
    Repeat training for 5 independent resamples of parameter initializations and data seed.
 
    Train the student model on each of these configurations
    Save the following metrics
    data: dict. columns: best_train_accuracy, best_test_accuracy, nu, n_hl
 
    save the data+configs as payload dict in DATA_PATH. name: yin_yang_nu_hl_sweep_{today}.pkl
    """
    print("--" * 50)
    print("Sweeping Num Hidden Layers and Integration Window | Trident Yin-Yang Dataset")
    print("--" * 50)
 
    if args is None:
        args = parse_args()  # NOTE: ignore args.num_hidden_layers / args.int_window -- swept below
 
    N_HL_ARR = [1, 2, 3, 4, 5] # no. hidden layers
    NU_ARR = [2, 20, 200, 2000] # no. bipolar samples to average at each layer. 
    NUM_RESAMPLES = args.num_resamples
 
    print(f"NUM HIDDEN LAYERS SWEEP: {N_HL_ARR}")
    print(f"NU SWEEP: {NU_ARR}")
    print("** GAUSSIAN NOISE SELECTED **" if args.gaussian_noise else "** LOGISTIC NOISE SELECTED **")
 
    # data storage
    data = defaultdict(list)
 
    # configs for the run
    configs = {
        'n_hl_sweep': N_HL_ARR,
        'nu_sweep': NU_ARR,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'resamples': NUM_RESAMPLES,
        'hidden_layer_size': args.hidden_layer_size,
        'adamw_used': not args.use_sgd,
        'noise_dist': "gaussian" if args.gaussian_noise else "logistic",
        'noise_std': args.std,
        'noise_mean': args.mean,
        'learning_rate': args.learning_rate,
        'train_steps': args.train_steps,
        'n_train': args.n_train,
        'n_test': args.n_test,
    }
 
    for hl_idx, n_hl in enumerate(N_HL_ARR):
        for nu_idx, nu in enumerate(NU_ARR):
            for r in range(NUM_RESAMPLES):
                print(f"N_HL = {n_hl} // {hl_idx + 1}/{len(N_HL_ARR)}")
                print(f"NU = {nu} // {nu_idx + 1}/{len(NU_ARR)}")
                print(f"RESAMPLING... {r + 1}/{NUM_RESAMPLES}")
 
                # Unique seed per (n_hl, nu, resample) -- reseeds BOTH the
                # student model's param/trident-noise init (via train's
                # own `rngs = nnx.Rngs(..., params=args.seed, ...)`) AND the
                # yin-yang data draw (get_train_test is called with this same
                # seed inside train), matching "independent resamples of
                # parameter initializations and data seed".
                seed = args.seed + hl_idx * 10_000 + nu_idx * 100 + r
 
                # Fresh copy of args per run -- reuses train() itself
                # (rather than re-inlining the whole training loop the way
                run_args = copy.copy(args)
                run_args.num_hidden_layers = n_hl
                run_args.int_window = nu
                run_args.seed = seed
                run_args.plot_boundary = False       # don't dump a PNG per run during the sweep
                run_args.save_boundary_data = False  # ditto for the .npz
                run_args.save_results = False        # sweep saves ONE payload at the end, not per-run
 
                metrics_history, _student_model = train(run_args)
 
                best_train_acc = max(metrics_history['train_accuracy'])
                best_test_acc = max(metrics_history['eval_accuracy'])
                best_idx = int(np.argmax(metrics_history['eval_accuracy']))
 
                print(
                    f"Best Eval Accuracy: {best_test_acc:.4f} | Best Train Accuracy: {best_train_acc:.4f} "
                    f"| Step: {metrics_history['step'][best_idx]}"
                )
 
                data["nu"].append(nu)
                data["n_hl"].append(n_hl)
                data["resample"].append(r)
                data["best_train_accuracy"].append(best_train_acc)
                data["best_test_accuracy"].append(best_test_acc)
 
    filename = f"yin_yang_nu_hl_sweep_noise_{configs['noise_dist']}_{today}.pkl"
    save_payload(data=data, configs=configs, filename=filename)
 
    return data, configs

# --------------------------------------------------------------
# Store decision boundary
# --------------------------------------------------------------
def store_decision_boundary(args=None, resolution=None):
    """
    Train a network once (via train()) on the Yin-Yang data, evaluate it on
    a dense (resolution x resolution) grid over [0, 1]^2, and store the
    grid + predicted classes as flat columns:
 
        data = {'xx': (resolution**2,), 'yy': (resolution**2,),
                'preds': (resolution**2,),
                'X_test': (n_test, 2), 'y_test': (n_test,)}
 
    ("individual columns" -- xx/yy/preds are each flattened to 1D so they
    line up element-for-element and can be handed straight to a DataFrame
    or to yin_yang_dataset.plot_decision_boundary_from_grid after reshaping
    back to (resolution, resolution), which is what
    load_and_plot_decision_boundary below does.)
 
    Saves {'configs': ..., 'data': ...} to
    DATA_PATH/yin_yang_boundary_{today}.pkl via save_payload(), and also
    returns (data, configs).
    """
    print("--" * 50)
    print("Storing Decision Boundary | Trident Yin-Yang Dataset")
    print("--" * 50)
 
    if args is None:
        args = parse_args()
    resolution = args.boundary_resolution if resolution is None else resolution
 
    layer_sizes = build_layer_sizes(
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_size=args.hidden_layer_size,
        in_features=2, out_features=args.num_classes,
    )
    print(f"LAYER SIZES: {layer_sizes} | nu={args.int_window} | resolution={resolution}")
 
    run_args = copy.copy(args)
    # this run is purely to get a trained model -- suppress the one per-run
    # side effect train() can still do (the results pickle).
    run_args.save_results = False
 
    metrics_history, student_model = train(run_args)
    best_accuracy = max(metrics_history['eval_accuracy'])
    print(f"Best Eval Accuracy: {best_accuracy:.4f}")
 
    # Regenerate the EXACT data this model was trained on -- get_train_test
    # is deterministic given the seed, so this reproduces X_test/y_test
    # (used here as the scatter overlay) without train() needing to return
    # them itself.
    _X_train_np, _labels_train_np, X_test_np, labels_test_np = get_train_test(
        n_train=args.n_train, n_test=args.n_test,
        r_small=args.r_small, r_big=args.r_big, seed=args.seed,
    )
 
    def predict_fn(grid):
        logits = student_model(jnp.asarray(grid, dtype=jnp.float32))
        return np.asarray(jnp.argmax(logits, axis=-1))
 
    xx, yy, preds = compute_decision_boundary_grid(predict_fn, resolution=resolution)
 
    data = {
        'xx': np.asarray(xx).reshape(-1),
        'yy': np.asarray(yy).reshape(-1),
        'preds': np.asarray(preds).reshape(-1),
        'X_test': X_test_np,
        'y_test': labels_test_np,
    }
 
    configs = {
        'seed': args.seed,
        'layer_sizes': layer_sizes,
        'num_hidden_layers': args.num_hidden_layers,
        'hidden_layer_size': args.hidden_layer_size,
        'nu': args.int_window,
        'noise_std': args.std,
        'noise_mean': args.mean,
        'noise_dist': "gaussian" if args.gaussian_noise else "logistic",
        'eval_accuracy': best_accuracy,
        'resolution': resolution,
    }
 
    filename = f"yin_yang_boundary_noise_{configs['noise_dist']}_nu_{args.int_window}_{today}.pkl"
    save_payload(data=data, configs=configs, filename=filename)
 
    return data, configs


# --------------------------------------------------------------
# Store layer-wise inputs (preactivations)
# --------------------------------------------------------------
def store_preactivations(args=None, data_split=None):
    """
    Take in a configuration of network: hidden layer size and numbers, nu,
    std, noise distribution (all via the usual CLI args). Train the
    network on the Yin-Yang data, then pass the input data back through
    the trained network and store the preactivations layer-wise in a dict:
 
        data = {'hl1': flattened array of all preactivations, 'hl2': ..., ...}
 
    ("flattened" = pooled over both hidden units and datapoints, so
    data['hl1'] is a 1D array you can hand straight to a histogram/KDE --
    exactly what plot_preactivation_distributions below expects.)
 
 
    Saves {'configs': ..., 'data': ...} to
    DATA_PATH/yin_yang_preactivations_{today}.pkl via save_payload(), and
    also returns (data, configs).
    """
    print("--" * 50)
    print("Storing Layer-wise Preactivations | Trident Yin-Yang Dataset")
    print("--" * 50)
 
    if args is None:
        args = parse_args()
    data_split = args.preact_data_split if data_split is None else data_split
 
    layer_sizes = build_layer_sizes(
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_size=args.hidden_layer_size,
        in_features=2, out_features=args.num_classes,
    )
    print(f"LAYER SIZES: {layer_sizes} | nu={args.int_window}")
 
    run_args = copy.copy(args)
    run_args.plot_boundary = False
    run_args.save_boundary_data = False
    run_args.save_results = False
 
    metrics_history, student_model = train(run_args)
    train_acc = max(metrics_history['train_accuracy'])
    print(f"Train accuracy: {train_acc:.4f} (no threshold enforced)")
 
    # Regenerate the EXACT data this model was trained on -- get_train_test
    # is deterministic given the seed, so this reproduces X_train/X_test
    # without train() needing to return them itself.
    X_train_np, labels_train_np, X_test_np, labels_test_np = get_train_test(
        n_train=args.n_train, n_test=args.n_test,
        r_small=args.r_small, r_big=args.r_big, seed=args.seed,
    )
 
    if data_split == "train":
        X_np = X_train_np
    elif data_split == "test":
        X_np = X_test_np
    elif data_split == "both":
        X_np = np.concatenate([X_train_np, X_test_np], axis=0)
    else:
        raise ValueError(f"Unknown data_split={data_split!r}, expected 'train'/'test'/'both'")
 
    preacts, _logits = get_preactivations(student_model, jnp.asarray(X_np))
 
    if not preacts:
        print("NOTE: num_hidden_layers=0 -- no hidden layers, so there are no preactivations to store.")
 
    # pool over both hidden units and datapoints -> one 1D array per layer
    data = {name: np.asarray(arr).reshape(-1) for name, arr in preacts.items()}
 
    configs = {
        'seed': args.seed,
        'layer_sizes': layer_sizes,
        'num_hidden_layers': args.num_hidden_layers,
        'hidden_layer_size': args.hidden_layer_size,
        'nu': args.int_window,
        'noise_std': args.std,
        'noise_mean': args.mean,
        'noise_dist': "gaussian" if args.gaussian_noise else "logistic",
        'train_accuracy': train_acc,
        'data_split': data_split,
        'n_points': int(X_np.shape[0]),
    }
 
    filename = f"yin_yang_preactivations_nu_{configs['nu']}_noise_{configs['noise_dist']}_{today}.pkl"
    save_payload(data=data, configs=configs, filename=filename)
 
    return data, configs
 
 

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()

    if args.run_train:
        metrics_history, student_model = train(args)

    if args.run_nu_hl_sweep:
        data, configs = sweep_hl_nu()
        print(data)

    if args.run_store_preactivations:
        data, configs = store_preactivations(args)
        print(data)

    if args.run_boundaries:
            data, configs = store_decision_boundary(args)
            print(data)


if __name__ == "__main__":
    main()