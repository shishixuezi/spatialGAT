import json
import os
import pyproj
import random
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import jismesh.utils as ju

from datetime import datetime


class EarlyStopper:
    def __init__(self, patience=7, verbose=False, delta=0.15, save_path="."):
        """
        Initialization of the early stopper
        :param patience: Number of epochs to wait
        :param verbose:
        :param delta: The minimum required change
        :param save_path:
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.val_loss_min:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.val_loss_min = val_loss
        elif val_loss >= self.val_loss_min + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, save_model=False):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        if save_model:
            save_path = os.path.join(self.save_path, 'best_model')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(save_path)


class Params:
    """
    Reference: https://cs230.stanford.edu/blog/hyperparameters
    loading: params = Params("experiments/base_model/params.json")
    accessing: params.model_name
    updating: params.update("other_params.json")
    defining the model:
    if params.model_name == "homo":
        logits = model_homo(inputs, params)
    elif params.model_name == "hetero":
        logits = model_hetero(inputs, params)
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class Metrics:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def rmse(self):
        return F.mse_loss(self.y_pred, self.y_true).sqrt()

    def mae(self):
        return F.l1_loss(self.y_pred, self.y_true).cpu().detach().numpy()

    def pcc(self):
        return np.corrcoef(self.y_pred.squeeze().cpu().detach().numpy(),
                           self.y_true.squeeze().cpu().detach().numpy())[0][1]


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        # d = {k: float(v) for k, v in d.items()}
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, folder):
    """
    Example:
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optim_dict': optimizer.state_dict()}
        utils.save_checkpoint(state,
                              is_best=is_best,     # True if this is the model with the best metrics
                              folder=model_dir)    # Path to the folder
    """
    path = os.path.join(folder, 'last.pth.tar')
    if not os.path.exists(folder):
        print("Directory does not exist! Making directory {}. Save the checkpoint!".format(folder))
        os.mkdir(folder)
    else:
        print("Directory exists! Save the checkpoint!")
        torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(folder, 'best.pth.tar'))


def load_checkpoint(folder, model, optimizer=None):
    """
    The optimizer argument is optional, and it can choose to restart with a new optimizer.
    Example:
        utils.load_checkpoint(restore_path, model, optimizer)
    Args:
        folder: folder name
        model: model
        optimizer: optimizer

    Returns:

    """
    if not os.path.exists(folder):
        raise ("File does not exist {}!".format(folder))
    checkpoint = torch.load(folder)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return model, optimizer


def mesh_pair_distance(m1, m2):
    """
    Return distance for central point of m1 and m2.
    Args:
        m1: mesh id of m1
        m2: mesh id of m2

    Returns:
        d: distance
    """
    m1_lat, m1_lon = ju.to_meshpoint(m1, lat_multiplier=0.5, lon_multiplier=0.5)
    m2_lat, m2_lon = ju.to_meshpoint(m2, lat_multiplier=0.5, lon_multiplier=0.5)
    geodesic = pyproj.Geod(ellps='WGS84')
    faz, baz, d = geodesic.inv(m1_lon, m1_lat, m2_lon, m2_lat)
    d /= 1000.0
    return d


def get_path(relative_path):
    src = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(src, relative_path)


def create_output_dir(root='result', experiment_name=None):
    def generate_folder_with_index(f):
        for i in range(1000):
            if not os.path.isdir(f + '.' + str(i)):
                return f + '.' + str(i)

    if experiment_name is not None:
        folder_name = os.path.join(root, 'experiment_' + experiment_name)
    else:
        folder_name = os.path.join(root, datetime.today().strftime('%Y%m%d'))
    if os.path.isdir(folder_name):
        folder_name = generate_folder_with_index(folder_name)
    os.makedirs(get_path(os.path.join(folder_name, 'diagram')))
    return folder_name


def gaussian_normalize(x):
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-12)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # ensure reproducibility using GPU, but will deteriorate the training speed and performance
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    """
