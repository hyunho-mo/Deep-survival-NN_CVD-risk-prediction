'''
Utility functions for running DeepSurv experiments
'''

import h5py
import scipy.stats as st
from collections import defaultdict
import numpy as np
import pandas as pd
import copy
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
from torchvision import datasets
from torchvision import transforms
from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn.functional as F

from sklearn.experimental import enable_iterative_imputer
# noinspection PyUnresolvedReferences
from sklearn.impute import IterativeImputer
from lifelines.utils import concordance_index

import configparser

# import lasagne
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

###################### Deepsurv ###########################
from torch.utils.data import Dataset

import tensorboard_logger 
from collections import defaultdict
import sys
import math

class DeepSurvLogger():
    def __init__(self, name):
        self.logger         = logging.getLogger(name)
        self.history = {}

    def logMessage(self,message):
        self.logger.info(message)

    def print_progress_bar(self, step, max_steps, loss = None, ci = None, bar_length = 25, char = '*'):
        progress_length = int(bar_length * step / max_steps)
        progress_bar = [char] * (progress_length) + [' '] * (bar_length - progress_length)
        space_padding = int(math.log10(max_steps))
        if step > 0:
            space_padding -= int(math.log10(step))
        space_padding = ''.join([' '] * space_padding)
        message = "Training step %d/%d %s|" % (step, max_steps, space_padding) + ''.join(progress_bar) + "|"
        if loss:
            message += " - loss: %.4f" % loss
        if ci:
            message += " - ci: %.4f" % ci

        self.logger.info(message)

    def logValue(self, key, value, step):
        pass

    def shutdown(self):
        logging.shutdown()

class TensorboardLogger(DeepSurvLogger):
    def __init__(self, name, logdir, max_steps = None, update_freq = 10):
        self.max_steps = max_steps

        self.logger         = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        format = logging.Formatter("%(asctime)s - %(message)s")
        ch.setFormatter(format)
        self.logger.addHandler(ch)

        self.update_freq    = update_freq

        self.tb_logger = tensorboard_logger.Logger(logdir)

        self.history = defaultdict(list)

    def logValue(self, key, value, step):
        self.tb_logger.log_value(key, value, step)
        self.history[key].append((step, value))


class Dataset_Converter(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, dict):
        ''' Loading data from .h5 file based on (h5_file, is_train).

        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.e, self.y = self.read_dict(dict)


        # normalizes data
        self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))
    
    def read_dict (self, dict):

        X = dict["x"]
        e = dict["e"]
        y = dict["t"]


        return X, e, y


    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)

    
        e_item = np.array(e_item)
        y_item = np.array(y_item)
        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]




def read_config(ini_file):
    ''' Performs read config file and parses it.

    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def create_logger(logs_dir):
    ''' Performs creating logger

    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    # logs settings
    log_file = os.path.join(logs_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # initialize console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


###################### Autoencoder imputation ######################################

class SampleDataset(Dataset):
    def __init__(self, X, device, clazz=0):
        self.__device = device
        self.__clazz = clazz
        self.__X = X

    def __len__(self):
        return self.__X.shape[0]

    def __getitem__(self, idx):
        item = self.__X[idx,:]

        return item, self.__clazz

class AE1(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        self.drop_out = torch.nn.Dropout(p=0.5)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size)
        )

    def forward(self, x):
        drop_out = self.drop_out(x)
        encoded = self.encoder(drop_out)
        # encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# class AE1(torch.nn.Module):
#     def __init__(self, input_size):
#         super().__init__()

#         self.input_size = input_size
#         self.drop_out = torch.nn.Dropout(p=0.5)

#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 36),
#             torch.nn.ReLU(),
#             torch.nn.Linear(36, 18)
#         )

#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(18, 36),
#             torch.nn.ReLU(),
#             torch.nn.Linear(36, 72),
#             torch.nn.ReLU(),
#             torch.nn.Linear(72, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, input_size)
#         )

#     def forward(self, x):
#         drop_out = self.drop_out(x)
#         encoded = self.encoder(drop_out)
#         # encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

class AE2(torch.nn.Module):
    def __init__(self, dim, theta=7):
        super().__init__()
        self.dim = dim
        self.theta = theta

        self.drop_out = torch.nn.Dropout(p=0.5)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim+theta*0, dim+theta*1),
            torch.nn.Tanh(),
            torch.nn.Linear(dim+theta*1, dim+theta*2),
            torch.nn.Tanh(),
            torch.nn.Linear(dim+theta*2, dim+theta*3)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(dim+theta*3, dim+theta*2),
            torch.nn.Tanh(),
            torch.nn.Linear(dim+theta*2, dim+theta*1),
            torch.nn.Tanh(),
            torch.nn.Linear(dim+theta*1, dim+theta*0)
        )

    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)

        z = self.encoder(x_missed)
        out = self.decoder(z)

        out = out.view(-1, self.dim)

        return out

def autoencoder_train(model, optimizer, N_dl, device):
    loss_function = torch.nn.MSELoss()

    epochs = 20
    loss_df = []

    for epoch in range(epochs):
        losses = []

        for (items, _) in N_dl:
            items = items.to(device)
            optimizer.zero_grad()

            reconstructed = model(items)
            loss = loss_function(reconstructed, items)

            loss.backward()

            optimizer.step()

            losses.append(loss.detach().cpu().numpy().item())

        losses = np.array(losses)

        loss_df.append({
            'epoch': epoch + 1,
            'loss': losses.mean()
        })

    loss_df = pd.DataFrame(loss_df)
    loss_df.index = loss_df['epoch']
    loss_df = loss_df.drop(columns=['epoch'])

    return loss_df

def autoencoder_predict(m, items, device):
    return m(items.to(device)).cpu().detach().numpy()


def get_imputation(m_v, p_v):
    def get_value(m, p):
        if pd.isna(m):
            return p
        else:
            return m        
    return np.array([get_value(m, p) for m, p in zip(m_v, p_v)])

############################## GAIN #############################


#%% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])        

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx



###############################################################

def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def format_dataset_to_df(dataset, duration_col, event_col, trt_idx = None):
    xdf = pd.DataFrame(dataset['x'])
    if trt_idx is not None:
        xdf = xdf.rename(columns={trt_idx : 'treat'})

    dt = pd.DataFrame(dataset['t'], columns=[duration_col])
    censor = pd.DataFrame(dataset['e'], columns=[event_col])
    cdf = pd.concat([xdf, dt, censor], axis=1)
    return cdf

def standardize_dataset(dataset, offset, scale):
    norm_ds = copy.deepcopy(dataset)
    norm_ds['x'] = (norm_ds['x'] - offset) / scale
    return norm_ds

def bootstrap_metric(metric_fxn, dataset, N=100):
    def sample_dataset(dataset, sample_idx):
        sampled_dataset = {}
        for (key,value) in dataset.items():
            sampled_dataset[key] = value[sample_idx]
        return sampled_dataset

    metrics = []
    size = len(dataset['x'])

    for _ in range(N):
        resample_idx = np.random.choice(size, size=size, replace = True)
    
        metric = metric_fxn(**sample_dataset(dataset, resample_idx))
        metrics.append(metric)
    
    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics)-1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }

# def get_optimizer_from_str(update_fn):
#     if update_fn == 'sgd':
#         return lasagne.updates.sgd
#     elif update_fn == 'adam':
#         return lasagne.updates.adam
#     elif update_fn == 'rmsprop':
#         return lasagne.updates.rmsprop

#     return None

def calculate_recs_and_antirecs(rec_trt, true_trt, dataset, print_metrics=True):
    if isinstance(true_trt, int):
        true_trt = dataset['x'][:,true_trt]

    # trt_values = zip([0,1],np.sort(np.unique(true_trt)))
    trt_values = enumerate(np.sort(np.unique(true_trt)))
    equal_trt = [np.logical_and(rec_trt == rec_value, true_trt == true_value) for (rec_value, true_value) in trt_values]
    rec_idx = np.logical_or(*equal_trt)
    # original Logic
    # rec_idx = np.logical_or(np.logical_and(rec_trt == 1,true_trt == 1),
    #               np.logical_and(rec_trt == 0,true_trt == 0))

    rec_t = dataset['t'][rec_idx]
    antirec_t = dataset['t'][~rec_idx]
    rec_e = dataset['e'][rec_idx]
    antirec_e = dataset['e'][~rec_idx]

    if print_metrics:
        print("Printing treatment recommendation metrics")
        metrics = {
            'rec_median' : np.median(rec_t),
            'antirec_median' : np.median(antirec_t)
        }
        print("Recommendation metrics:", metrics)

    return {
        'rec_t' : rec_t, 
        'rec_e' : rec_e, 
        'antirec_t' : antirec_t, 
        'antirec_e' : antirec_e
    }
    



################################################ EGAIN ###########

def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    '''

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    '''

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values
      - imputed_data: imputed data
      - data_m: indicator matrix for missingness

    Returns:
      - rmse: Root Mean Squared Error
    '''

    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    rmse_mean = rmse_benchmarks(ori_data, data_m)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse, rmse_mean


def hint_for_mar(p, mask):
    m_indices = np.where(1 - mask)
    h_indices = np.random.choice(range(len(m_indices[0])), size=int(p * len(m_indices[0])), replace=False)
    hh_indices = m_indices[0][h_indices], m_indices[1][h_indices]
    h = np.zeros_like(mask)
    h[hh_indices] = 1
    return 1 - h


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size=[rows, cols])


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
      - total: total number of samples
      - batch_size: batch size

    Returns:
      - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx