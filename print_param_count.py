"""
    Notes:
        * I won't add model checkpoint averaging as mentioned in the paper - it just feels like an arbitrary heuristic
         and it won't add anything to the learning experience this repo aims to provide.

"""

import weightwatcher as ww
import argparse
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
from operator import itemgetter
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from models.definitions.transformer_model import Transformer, count_parameters
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

for layer in [4,5,6]:
    for dim in [384,768]:
        for head in [4,6,8,12,16]:
            baseline_transformer = Transformer(
                model_dimension=dim,
                src_vocab_size=58947,
                trg_vocab_size=36206,
                number_of_heads=head,
                number_of_layers=layer,
                dropout_probability=0.1
            ).to(device)
            print(count_parameters(baseline_transformer))
