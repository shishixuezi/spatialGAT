import copy
import logging
import sys

import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

import utils
from args import args
from graph import HomogeneousODGraph
from model import SpatialGAT
from preprocessing import preprocessing
from utils import get_path, gaussian_normalize
from loss import choose_criterion_type


def train(model, data, optimizer, criterion, scheduler=None, clip_threshold=None):
    model.train()
    optimizer.zero_grad()

    if hasattr(data, 'edge_label_index'):
        # Using RandomLinkSplit
        edge_label_index = data.edge_label_index
        edge_label = data.edge_label.unsqueeze(-1)
    else:
        # No Split or Using Inductive Learning (train, validate and test sets are completely disjoint)
        edge_label_index = data.edge_index
        edge_label = data.edge_label

    edge_index = data.edge_index

    in_embed, out_embed, volume, general_embed = model.forward(data.x,
                                                               edge_index,
                                                               data.edge_weight,
                                                               edge_label_index)
    loss = criterion(volume.squeeze(), edge_label.squeeze())
    loss.backward()
    if clip_threshold is not None:
        nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return float(loss)


@torch.no_grad()
def test(model, data, criterion, return_embed=False, return_prediction=False):
    model.eval()

    if hasattr(data, 'edge_label_index'):
        edge_label_index = data.edge_label_index
        edge_label = data.edge_label.unsqueeze(-1)
    else:
        edge_label_index = data.edge_index
        edge_label = data.edge_label

    edge_index = data.edge_index

    in_embed, out_embed, out, general_embed = model.forward(data.x,
                                                            edge_index,
                                                            data.edge_weight,
                                                            edge_label_index)

    out[out < 0] = 0
    test_loss = criterion(out.squeeze(), edge_label.squeeze())

    if return_prediction:
        if return_embed:
            return float(test_loss), out, edge_label, in_embed, out_embed, general_embed
        else:
            return float(test_loss), out, edge_label
    else:
        if return_embed:
            return float(test_loss), in_embed, out_embed, general_embed
        else:
            return float(test_loss)


def model_homo(result_folder):
    utils.seed_everything(args.seed)
    utils.save_dict_to_json(vars(args), os.path.join(result_folder, 'param.json'))
    logger = logging.getLogger(__name__)
    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = torch.device(args.device)
    logger.info('Device: {}'.format(device))
    criterion = choose_criterion_type(args.loss_type)

    T_list = [T.ToDevice(device),
              T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=False,
                                add_negative_train_samples=False, neg_sampling_ratio=0.0)]
    transform = T.Compose(T_list)

    name_list = ['静岡市葵区', '静岡市駿河区', '浜松市中区', '富士市', '沼津市', '裾野市']

    dict_df_data = preprocessing(city_name=name_list[args.city])
    logger.info(f'The target city name is {name_list[args.city]}')
    g = HomogeneousODGraph(dict_df_data['xs'], dict_df_data['edges']['od'])
    data = g.graph

    data.x = gaussian_normalize(data.x)
    data.edge_weight = gaussian_normalize(data.edge_weight)

    if args.is_inductive:
        num = data.num_nodes
        perm = torch.randperm(num)
        train_mask = perm[:int(num * 0.8)]
        val_mask = perm[int(num * 0.8):int(num * 0.9)]
        test_mask = perm[int(num * 0.9):]
        train_data = data.subgraph(train_mask).to(device)
        val_data = data.subgraph(val_mask).to(device)
        test_data = data.subgraph(test_mask).to(device)
    else:
        train_data, val_data, test_data = transform(data)

    model = SpatialGAT(in_channels=data.x.shape[1],
                       hidden_channels=args.hidden_channels,
                       out_channels=args.embedding_size,
                       heads=args.heads,
                       dropout=args.dropout,
                       layer_type=args.layer_type).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.lr_weight_decay)

    if args.is_scheduled:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.scheduler_step,
                                                    gamma=args.scheduler_gamma)
    else:
        scheduler = None

    best_loss = float('inf')
    best_model = None
    best_epoch = 1
    train_losses = []
    val_losses = []

    start = time.time()
    for epoch in range(1, args.epochs):

        train_loss = train(model, train_data, optimizer, criterion, scheduler=scheduler,
                           clip_threshold=args.clip_threshold)

        val_loss = test(model, val_data, criterion)

        if (epoch % 10 == 1) or args.is_verbose:
            logger.info('Epoch: {:07d}, Training Loss: {:.4f}, Validation Loss: {:4f}'.format(
                epoch, train_loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end = time.time()
    logger.info('Training time elapsed: {}'.format(end - start))
    logger.info('Best epoch is: {}'.format(best_epoch))

    torch.save(model.state_dict(), get_path(os.path.join(result_folder, 'model.pth')))

    # Start inferring
    start = time.time()
    test_loss, prediction, label, in_embed, out_embed, general_embed = test(best_model, test_data, criterion,
                                                                            return_embed=True,
                                                                            return_prediction=True)
    end = time.time()
    logger.info('Inferring time elapsed: {}'.format(end - start))

    logger.info('L2 Loss: {}'.format(F.mse_loss(prediction, label).cpu().detach().numpy()))
    logger.info('RMSE: {}'.format(np.sqrt(F.mse_loss(prediction, label).cpu().detach().numpy())))
    logger.info('L1 Loss: {}'.format(F.l1_loss(prediction, label).cpu().detach().numpy()))
    logger.info('PCC: {}'.format(np.corrcoef(prediction.squeeze().cpu().detach().numpy(),
                                             label.squeeze().cpu().detach().numpy())[0][1]))

    g.convert_edge_index_with_edge_value_to_df(test_data.edge_label_index, [prediction, label]).to_csv(
        os.path.join(result_folder, 'test_prediction.csv'), header=False, index=False)

    torch.save(in_embed, get_path(os.path.join(result_folder, 'in_embed.pth')))
    torch.save(out_embed, get_path(os.path.join(result_folder, 'out_embed.pth')))
    torch.save(general_embed, get_path(os.path.join(result_folder, 'general_embed.pth')))

    logger.info('Final: {:4f}'.format(test_loss))
    logger.info('Best:{:4f}'.format(best_loss))


def main():
    result_folder = utils.create_output_dir(args.save_folder, args.experiment_name)
    # set output logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(get_path(os.path.join(result_folder, 'log.txt')))
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    targets = console_handler, file_handler
    logger.handlers = targets

    model_homo(result_folder)


if __name__ == '__main__':
    main()
