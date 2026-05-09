# Utils module
from .trident import trident
from .ternary_activation import ternary_activation
from .load_cifar10_augment import load_cifar10_augment
from .dual_sample_ternary import dual_sample_ternary
from .load_uci_iris import load_uci_iris

__all__ = ['trident', 
           'ternary_activation', 
           'load_cifar10_augment',
            'dual_sample_ternary',
            'load_uci_iris']