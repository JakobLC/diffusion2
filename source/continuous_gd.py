
import enum
import math
import numpy as np
import torch
from .utils import smooth_threshold,TemporarilyDeterministic

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_gamma_schedule(schedule_name):
    pass

class MeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    EPS = enum.auto()  
    X = enum.auto() 
    V = enum.auto()
    BOTH = enum.auto()

class WeightsType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    SNR = enum.auto()
    SNR_plus1 = enum.auto()
    SNR_trunc = enum.auto()
    uniform = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    BCE = enum.auto()