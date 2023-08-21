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
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import information_process
from torch.autograd import Variable
from utils.decoding_utils import greedy_decoding
from nltk.translate.bleu_score import corpus_bleu

from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *

num_of_trg_tokens_processed = 0
bleu_scores = []
global_train_step, global_val_step = [0, 0]
best_val_loss=None

num_warmup_steps = 4000

#
# Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
#
parser = argparse.ArgumentParser()
# According to the paper I infered that the baseline was trained for ~19 epochs on the WMT-14 dataset and I got
# nice returns up to epoch ~20 on IWSLT as well (nice round number)
parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
# You should adjust this for your particular machine (I have RTX 2080 with 8 GBs of VRAM so 1500 fits nicely!)
parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

# Data related args
parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)

# Logging/debugging/checkpoint related (helps a lot with experimentation)
parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=True)
parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)

parser.add_argument("--model_dimension", type=int, help="the dimension to save a checkpoint", default=BASELINE_MODEL_DIMENSION)
parser.add_argument("--num_heads", type=int, help="number of Transformer heads", default=BASELINE_MODEL_NUMBER_OF_HEADS)
parser.add_argument("--num_layers", type=int, help="number of Transformer layers", default=BASELINE_MODEL_NUMBER_OF_LAYERS)
parser.add_argument("--dropout", type=float, help="dropout probability", default=BASELINE_MODEL_DROPOUT_PROB)
parser.add_argument("--lr_factor", type=float, help="constant to multiply lr by", default=1.0)
parser.add_argument("--checkpoint", type=str, help="checkpoint model saving path", default=CHECKPOINTS_PATH)
parser.add_argument("--temp_balance", type=str, help="if use temp balance function", default='')
parser.add_argument("--sample_evals", action='store_true', help='are we changing the ESD')
parser.add_argument("--pl_fit", type=str,default='E_TPL', help='distribution to fit ESD')
parser.add_argument("--metric", type=str,default='Lambda', help='ww metric to use')
parser.add_argument("--desc",action='store_true', help='descending reorder of layers')
parser.add_argument("--path",type=str,default='E_TPL-IWSLT_e2g_ascdesc_layers6_dim512_dropout0.1_heads8_lr0.25')

args = parser.parse_args()

# Wrapping training configuration into a dictionary
training_config = dict()
for arg in vars(args):
    training_config[arg] = getattr(args, arg)
training_config['num_warmup_steps'] = num_warmup_steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
    training_config['dataset_path'],
    training_config['language_direction'],
    training_config['dataset_name'],
    training_config['batch_size'],
    device)

pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
src_vocab_size = len(src_field_processor.vocab)
trg_vocab_size = len(trg_field_processor.vocab)

baseline_transformer = Transformer(
    model_dimension=args.model_dimension,
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    number_of_heads=args.num_heads,
    number_of_layers=args.num_layers,
    dropout_probability=args.dropout
).to(device)

PATH = args.path
print("path: ", PATH)
if os.path.exists(PATH):
    checkpoint = torch.load(PATH+'/best.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'select' in k:
            v = v.view(1, -1, 1, 1)
        new_state_dict[k.replace('module.', '')] = v
    baseline_transformer.load_state_dict(new_state_dict, strict=False)

kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # gives better BLEU score than "mean"

# Makes smooth target distributions as opposed to conventional one-hot distributions
# My feeling is that this is a really dummy and arbitrary heuristic but time will tell.
label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)

# Check out playground.py for an intuitive visualization of how the LR changes with time/training steps, easy stuff.
custom_lr_optimizer = CustomLRAdamOptimizer(
            Adam(baseline_transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
            args.model_dimension,
            training_config['num_warmup_steps'],
            args.lr_factor
        )

device = next(baseline_transformer.parameters()).device

# # https://vadim.me/publications/blackbox/
# # https://github.com/ravidziv/IDNNs
# class ZivInformationPlane():
#     # The code by Ravid Schwartz-Ziv is very hard to understand 
#     # (the reader is encouraged to try it themselves)
#     # Solution: wrap it into this class and don't touch it with a 10-meter pole
    
#     def __init__(self, X, Y, bins = np.linspace(-1, 1, 30)):
#         """
#         Inititalize information plane (set X and Y and get ready to calculate I(T;X), I(T;Y))
#         X and Y have to be discrete
#         """
        
#         plane_params = dict(zip(['pys', 'pys1', 'p_YgX', 'b1', 'b', 
#                                  'unique_a', 'unique_inverse_x', 'unique_inverse_y', 'pxs'], 
#                                 information_process.extract_probs(np.array(Y).astype(np.float), X)))
        
#         plane_params['bins'] = bins
#         plane_params['label'] = Y
#         plane_params['len_unique_a'] = len(plane_params['unique_a'])
#         del plane_params['unique_a']
#         del plane_params['pys']
        
#         self.X = X
#         self.Y = Y
#         self.plane_params = plane_params
        
#     def mutual_information(self, layer_output):
#         """ 
#         Given the outputs T of one layer of an NN, calculate MI(X;T) and MI(T;Y)
        
#         params:
#             layer_output - a 3d numpy array, where 1st dimension is training objects, second - neurons
        
#         returns:
#             IXT, ITY - mutual information
#         """
            
#         information = information_process.calc_information_for_layer_with_other(layer_output, **self.plane_params)
#         return information['local_IXT'], information['local_ITY']

# Main loop - start of the CORE PART
#
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins=1000):
    print(x.shape,y.shape)
    c_xy = np.histogram2d(x.flatten(), y.flatten(), bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

all_I=[]
I_end=[]
I_dec=[]
val_losses = []
BLEU = []
for batch_idx, token_ids_batch in enumerate(train_token_ids_loader):
    I = []
    src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
    src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
    smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities

    # log because the KL loss expects log probabilities (just an implementation detail)
    
    # predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
    
    src_embeddings_batch = baseline_transformer.src_embedding(src_token_ids_batch)  # get embedding vectors for src token ids
    src_embeddings_batch = baseline_transformer.src_pos_embedding(src_embeddings_batch)  # add positional embedding
    src_representations_batch = src_embeddings_batch
    for encoder_layer in baseline_transformer.encoder.encoder_layers:
        # src_mask's role is to mask/ignore padded token representations in the multi-headed self-attention module
        new_src_representations_batch = encoder_layer(src_representations_batch, src_mask)
        I.append(calc_MI(src_representations_batch.cpu().detach().numpy(), new_src_representations_batch.cpu().detach().numpy()))
        src_representations_batch = new_src_representations_batch
    # infoplane = ZivInformationPlane(src_token_ids_batch.cpu().detach().numpy(),trg_token_ids_batch_gt.cpu().detach().numpy())

    src_representations_batch = baseline_transformer.encoder(src_embeddings_batch, src_mask)
    I_end.append(calc_MI(src_embeddings_batch.cpu().detach().numpy(), src_representations_batch.cpu().detach().numpy()))

    # I_end.append((infoplane.mutual_information(src_representations_batch),infoplane.mutual_information(trg_representations_batch)))
    # decoder_representations = []
        # Shape (B, T, D), where B - batch size, T - longest target token-sequence length and D - model dimension
    
    trg_embeddings_batch = baseline_transformer.trg_embedding(trg_token_ids_batch_input)  # get embedding vectors for trg token ids
    trg_embeddings_batch = baseline_transformer.trg_pos_embedding(trg_embeddings_batch)  # add positional embedding
 
    trg_representations_batch = trg_embeddings_batch
    for decoder_layer in baseline_transformer.decoder.decoder_layers:
        new_trg_representations_batch = decoder_layer(trg_representations_batch,src_representations_batch,trg_mask,src_mask)
        I.append(calc_MI(trg_representations_batch.cpu().detach().numpy(), new_trg_representations_batch.cpu().detach().numpy()))
        trg_representations_batch = new_trg_representations_batch

    trg_representations_batch = baseline_transformer.decoder(trg_embeddings_batch, src_representations_batch, trg_mask, src_mask)
    I_dec.append(calc_MI(trg_embeddings_batch.cpu().detach().numpy(), trg_representations_batch.cpu().detach().numpy()))

    predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch_input, src_representations_batch, trg_mask, src_mask)
    
    # I.append(calc_MI(predicted_log_distributions.cpu().detach().numpy(), smooth_target_distributions.cpu().detach().numpy()))

    loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)
    val_losses.append(loss.item())
    all_I.append(I)

print("I_end=np.array({})".format(I_end))
print("I_dec=np.array({})".format(I_dec))
print("all_I=np.array({})".format(all_I))
print("val_losses=np.array({})".format(val_losses))