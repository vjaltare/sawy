"""
Testing out the ternary resNet like network with ternary stochastic activation on CIFAR-10 dataset.

Created on: 02/19/2024
"""

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
import os
import pickle
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from datetime import date

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns



from utils import ternary_activation, load_cifar10_augment
from models import TernaryStochasticActivation

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

today = date.today().isoformat()

# --------------------------------------------------------------
# Parsing input arguments
# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Trident-ResNet on CIFAR-10")

    # take in the levels
    parser.add_argument("--levels", nargs="+", type=float, default=[-1.0, 0.0, 1.0])

    # take in the initial thresholds
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-1.0, 1.0])

    # take in the noise standard deviation
    parser.add_argument("--noise_std", type=float, default=0.1)

    # model architecture parameters
    parser.add_argument("--num64_blocks", type=int, default=3) 
    parser.add_argument("--num128_blocks", type=int, default=3)
    parser.add_argument("--num256_blocks", type=int, default=4)
    parser.add_argument("--ff_layer_sizes", nargs="+", type=int, default=[2000, 1000])

    # hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=int(5e4))
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--seed_next", type=int, default=0)


    #TODO: Add arguments for saving results and the model.
    parser.add_argument("--plot_results", action='store_true')



    return parser.parse_args()

# -------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------
def save_result_to_csv(result, csv_path):
    # Create directory if needed
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists to write headers
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        # Lock file for thread-safe writing
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        try:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(result)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

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

    checkpoint_dir = "/local_disk/vikrant/scrramble/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {filename_}")


# -------------------------------------------------------------------
# Residual Block: outputs shape (2048)
# -------------------------------------------------------------------
class ResidualBlock(nnx.Module):
    """
    Residual block with two convolutional layers and a residual connection.
    Order of operations:
    input -> conv -> relu -> conv + input -> relu

    TODO: Change call signature to include the TernaryStochasticActivation module instead of activation function.
    """

    def __init__(self,
                 kernel_size: tuple[int, int],
                 in_features: int,
                 out_features: int,
                #  stride: tuple[int, int],
                 rngs: nnx.Rngs,
                 padding: str = "SAME",
                 activation_fn: Callable = nnx.relu,
                 **kwargs):
        
        self.conv1 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(out_features, rngs=rngs)
        self.activation_fn = activation_fn

    def __call__(self, x):
        x_res = x
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = x + x_res
        x = self.activation_fn(x)
        return x
    
class AdaptiveResidualBlock(nnx.Module):
    """
    Residual block with two convolutional layers and a residual connection.
    Order of operations:
    input -> conv -> relu -> conv + input -> relu

    TODO: Change call signature to include the TernaryStochasticActivation module instead of activation function.
    """

    def __init__(self,
                 kernel_size: tuple[int, int],
                 in_features: int,
                 out_features: int,
                #  stride: tuple[int, int],
                 rngs: nnx.Rngs,
                 levels: list[float],
                 thresholds: list[float],
                 noise_std: float,
                 padding: str = "SAME",
                 **kwargs):
        
        self.conv1 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(out_features, rngs=rngs)
        self.activation_fn1 = TernaryStochasticActivation(levels=levels, thresholds=thresholds, noise_std=noise_std, rngs=rngs)
        self.activation_fn2 = TernaryStochasticActivation(levels=levels, thresholds=thresholds, noise_std=noise_std, rngs=rngs)

    def __call__(self, x):
        x_res = x
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.activation_fn1(x)
        x = self.conv2(x)
        x = x + x_res
        x = self.activation_fn2(x)
        return x
    
# -------------------------------------------------------------------
# Residual Network
# -------------------------------------------------------------------
class ResNet(nnx.Module):
    """
    Vanilla ResNet with residual blocks and final output layer.
    """

    def __init__(
            self,
            kernel_size: tuple[int, int],
            num64_blocks: int,
            num128_blocks: int,
            num256_blocks: int,
            ff_layer_sizes: list[int],
            # activation_function: Callable,
            levels: list[float],
            thresholds: list[float],
            noise_std: float,
            rngs: nnx.Rngs,
            **kwargs
        ):

        # unpack the rngs
        self.rngs = rngs  
        
        # bind he activation function with its parameters
        # self.activation_function = partial(activation_function, thresholds=thresholds, levels=levels, noise_std=noise_std) # pass a key every time activation is called

        # create a projection block 3 -> 64
        self.projection_block1 = nnx.Conv(in_features=3, out_features=64, kernel_size=(1, 1), padding="SAME", rngs=rngs)

        # create the first residual block with 64 channels
        self.res_block1 = nnx.List(
            [
                # ResidualBlock(in_features=64, out_features=64, kernel_size=kernel_size, padding="SAME", rngs=rngs, activation_fn=self.activation_function) 
                AdaptiveResidualBlock(in_features=64, out_features=64, kernel_size=kernel_size, padding="SAME", rngs=rngs, levels=levels, thresholds=thresholds, noise_std=noise_std)
                for _ in range(num64_blocks)
            ]
        )

        # create the second projection layer: 64 -> 128
        self.projection_block2 = nnx.Conv(in_features=64, out_features=128, kernel_size=(1, 1), padding="SAME", rngs=rngs)

        # create the residual block for 128 channels
        self.res_block2 = nnx.List(
            [
                # ResidualBlock(in_features=128, out_features=128, kernel_size=kernel_size, padding="SAME", rngs=rngs, activation_fn=self.activation_function) 
                AdaptiveResidualBlock(in_features=128, out_features=128, kernel_size=kernel_size, padding="SAME", rngs=rngs, levels=levels, thresholds=thresholds, noise_std=noise_std)
                for _ in range(num128_blocks)
            ]
        )

        # create projection block 128 -> 256
        self.projection_block3 = nnx.Conv(in_features=128, out_features=256, kernel_size=(1, 1), padding="SAME", rngs=rngs)

        # create the residual block for 256 channels
        self.res_block3 = nnx.List(
            [
                # ResidualBlock(in_features=256, out_features=256, kernel_size=kernel_size, padding="SAME", rngs=rngs, activation_fn=self.activation_function) 
                AdaptiveResidualBlock(in_features=256, out_features=256, kernel_size=kernel_size, padding="SAME", rngs=rngs, levels=levels, thresholds=thresholds, noise_std=noise_std)
                for _ in range(num256_blocks)
            ]
        )

        # activation function for linear layers
        self.activation_fn_l1 = TernaryStochasticActivation(levels=levels, thresholds=thresholds, noise_std=noise_std, rngs=rngs)
        self.activation_fn_l2 = TernaryStochasticActivation(levels=levels, thresholds=thresholds, noise_std=noise_std, rngs=rngs)

        # create the feedforward layers
        self.linear1 = nnx.Linear(in_features=4096, out_features=ff_layer_sizes[0], rngs=rngs)
        self.linear2 = nnx.Linear(in_features=ff_layer_sizes[0], out_features=ff_layer_sizes[1], rngs=rngs)
        self.classifier = nnx.Linear(in_features=ff_layer_sizes[1], out_features=10, rngs=rngs)

        # normalization (TODO: add a CIM specific normalization)

        # max pooling
        self.max_pool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x):
        # input shape (batch_size, 32, 32, 3)
        # pass through projection block
        x = self.projection_block1(x) # reshapes to (batch size, 32, 32, 64)

        # pass through the first residual block
        for res_block in self.res_block1:
            x = res_block(x)
        x = self.max_pool(x)

        # pass through project block 
        x = self.projection_block2(x) 

        # pass through second stage of residual blocks
        for res_block in self.res_block2:
            x = res_block(x)
        x = self.max_pool(x)

        # pass through projection block
        x = self.projection_block3(x)

        # pass through the third stage of residual blocks
        for res_block in self.res_block3:
            x = res_block(x)
        x = self.max_pool(x)

        # pass through the feedforward layers
        x = x.reshape((x.shape[0], -1)) # flatten along non-batch dimensions

        # print(f"Shape before feedforward layers: {x.shape}") # debug statement

        x = self.linear1(x)
        x = self.activation_fn_l1(x)
        x = self.linear2(x)
        x = self.activation_fn_l2(x)
        x = self.classifier(x)
        return x
    

# ---------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------
def loss_fn(
        model: ResNet,
        batch: dict,
        loss_function: Callable = optax.softmax_cross_entropy_with_integer_labels
    ):
    # forwad pass through the model
    logits = model(batch['image'])

    # using softmax cross-entropy with integer labels
    loss = loss_function(logits, labels=batch['label']).mean()

    return loss, logits

# training step
@nnx.jit
def train_step(
    model: ResNet,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: dict,
    loss_fn: Callable = loss_fn
    ):

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(model, grads)

# evaluation step
@nnx.jit
def eval_step(
    model: ResNet,
    metrics: nnx.MultiMetric,
    batch: dict,
    loss_fn: Callable = loss_fn
 ):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

# predfiction step
def pred_step(
        model: ResNet, 
        batch: dict
    ):

    logits = model(batch['image'])
    logits = nnx.softmax(logits, axis=-1)
    prediction = jnp.argmax(logits, axis=-1)

    return prediction

# ---------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------
def train_ternary_resnet(
        model: ResNet,
        optimizer: nnx.Optimizer,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        metrics_history: dict,
        metrics: nnx.MultiMetric,
        # dataset_configs: dict,
        configs: dict,
        **kwargs

    ):

    print("--"*50)
    print("Training ResNet with Ternary Stochastic activation: CIFAR-10 \n")
    print("--"*50)

    eval_every = configs['eval_every']
    train_steps = configs['train_steps']

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # train step
        train_step(model, optimizer, metrics, batch)

        # evaluate and record
        if step > 0 and (step%eval_every == 0 or step == train_steps - 1):
            metrics_history['step'].append(step)

            # log the training metrics
            for metric, value in metrics.compute().items():
                metrics_history[f'train_{metric}'].append(value.item())

            metrics.reset()

            # evaluate and log on validation dataset
            for valid_batch in valid_ds.as_numpy_iterator():
                eval_step(model, metrics, valid_batch)
            
            # log the validation metrics
            for metric, value in metrics.compute().items():
                metrics_history[f'valid_{metric}'].append(value.item())
            
            metrics.reset()

            # print out the results so far
            print(f" Step {step}/{train_steps} | Train Loss: {metrics_history['train_loss'][-1]:.4f} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Valid Loss: {metrics_history['valid_loss'][-1]:.4f} | Valid Accuracy: {metrics_history['valid_accuracy'][-1]:.4f}", end="\r")

    # print out the best accuracy
    best_accuracy = max(metrics_history['valid_accuracy'])
    print("__"*50)
    print(f"BEST VALIDATION ACCURACY: {best_accuracy:.4f}")
    print("__"*50)

    # find accuracy on held out test set
    for test_batch in test_ds.as_numpy_iterator():
        eval_step(model, metrics, test_batch)

    for metric, value in metrics.compute().items():
        metrics_history[f'test_{metric}'] = value.item()

    print("++"*50)
    print(f"TEST LOSS: {metrics_history['test_loss']:.4f} | TEST ACCURACY: {metrics_history['test_accuracy']:.4f}")
    print("++"*50)

    return model, metrics_history  


# ----------------------------------------
# Seting up training
# ----------------------------------------
def main():
    # parse the arguments
    args = parse_args()

    data_dir =  "/local_disk/vikrant/datasets"

    configs = {
        'levels': jnp.array(args.levels),
        'thresholds': jnp.array(args.thresholds),
        'noise_std': args.noise_std,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'train_steps': args.train_steps,
        'eval_every': args.eval_every,
        'data_dir': data_dir
    }

    seed = args.seed_next + 0

    # load CIFAR-10 dataset with augmentation
    train_ds, valid_ds, test_ds = load_cifar10_augment(
        batch_size=configs['batch_size'],
        train_steps=configs['train_steps'],
        data_dir=configs['data_dir'],
        shuffle_buffer = 1024,
        seed=seed,
        augmentation=True,
        training=False
    )

    # hyperparameters
    hyperparameters = {
        'learning_rate': configs['learning_rate'],
        'momentum': 0.9,
        'weight_decay': 1e-4
    }

    # learning rate scheduler
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=hyperparameters['learning_rate']*0.1,
        peak_value=hyperparameters['learning_rate']*1.3,
        warmup_steps=int(1e3),
        decay_steps=configs['train_steps'] - int(1e3),
        end_value=1e-6
    )

    rngs = nnx.Rngs(
        params=seed + 0,
        dropout=seed + 1,
        activation=seed + 2,
        next = seed + 3 
    )

    # activation function
    # ternary_activation_fn = partial(ternary_activation, thresholds=configs['thresholds'], levels=configs['levels'], noise_std=configs['noise_std'], key=rngs.activation())

    model_parameters = {
        'kernel_size': (3, 3),
        'rngs': rngs,
        # 'activation_function': ternary_activation_fn,
        'num64_blocks': args.num64_blocks,
        'num128_blocks': args.num128_blocks,
        'num256_blocks': args.num256_blocks,
        'ff_layer_sizes': args.ff_layer_sizes,
        'levels': configs['levels'],
        'thresholds': configs['thresholds'],
        'noise_std': configs['noise_std']

    }

    model = ResNet(**model_parameters)

    nnx.display(model)

    # optimizer setup
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=hyperparameters['weight_decay'],
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

    # train the model
    model, metrics_history = train_ternary_resnet(
        model=model,
        optimizer=optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        metrics_history=metrics_history,
        metrics=metrics,
        # dataset_configs: dict,
        configs=configs,
    )

    # TODO: add code to save the model and the results.


    # if plotting is enabled, plot the training curves
    if args.plot_results:
        fig, ax = plt.subplots(2, 1, figsize=(5, 5))
        ax1, ax2 = ax[0], ax[1]

        sns.lineplot(x=metrics_history['step'], y=metrics_history['train_loss'], label='Train Loss', ax=ax1, marker='o', lw=2.5)
        sns.lineplot(x=metrics_history['step'], y=metrics_history['valid_loss'], label='Valid Loss', ax=ax1, marker='o', lw=2.5)
        ax1.axhline(y=metrics_history['test_loss'], color='k', alpha=0.6, linestyle='--', label='Test Loss')
        ax1.set_xlabel('Training Steps', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)

        sns.lineplot(x=metrics_history['step'], y=metrics_history['train_accuracy'], label='Train Accuracy', ax=ax2, marker='o', lw=2.5)
        sns.lineplot(x=metrics_history['step'], y=metrics_history['valid_accuracy'], label='Valid Accuracy', ax=ax2, marker='o', lw=2.5)
        ax2.axhline(y=metrics_history['test_accuracy'], color='k', alpha=0.6, linestyle='--', label='Test Accuracy')
        ax2.set_xlabel('Training Steps', fontsize=14)
        ax2.set_ylabel('Accuracy', fontsize=14)
        sns.despine(ax=ax1)
        sns.despine(ax=ax2)
        plt.tight_layout()
        plt.savefig(f"TEMP_cifar10_ternary_test1_{today}.png", dpi=300)
        plt.show()







    
if __name__ == "__main__":
    main()
