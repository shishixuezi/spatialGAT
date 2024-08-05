import os
import torch
import utils
import pandas as pd
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from args import args
from torch_geometric.explain import CaptumExplainer, Explainer
from graph import HomogeneousODGraph
from model import SpatialGAT
from preprocessing import preprocessing
from utils import gaussian_normalize


def visualize_explanation(sensitivity):
    color = ['black'] * 1 + ['red'] * 40 + ['blue'] * 24 + ['green'] * 1
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16, 8))
    custom_lines = [Line2D([0], [0], color='black', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    ax.set_title('Aoi Ward, Shizuoka City')
    ax.set_ylabel('Feature Importance')
    ax.set_xticks([])
    ax.legend(custom_lines, ['Night Population', 'POIs', 'Road Density', 'Railway Users'])
    ax.bar(range(sensitivity.shape[1]), sensitivity.iloc[0], color=color)

    plt.savefig(os.path.join(args.explain_path, 'feature_analysis.pdf'), bbox_inches='tight')


def generate_explanation():
    name_list = ['静岡市葵区', '静岡市駿河区', '浜松市中区', '富士市', '沼津市', '裾野市']

    dict_df_data = preprocessing(city_name=name_list[args.city])
    g = HomogeneousODGraph(dict_df_data['xs'], dict_df_data['edges']['od'])
    data = g.graph

    data.x = gaussian_normalize(data.x)
    data.edge_weight = gaussian_normalize(data.edge_weight)

    transform = T.Compose([T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=False,
                                             add_negative_train_samples=False, neg_sampling_ratio=0.0)])

    train_data, val_data, test_data = transform(data)

    model = SpatialGAT(in_channels=data.x.shape[1],
                       hidden_channels=args.hidden_channels,
                       out_channels=args.embedding_size,
                       heads=args.heads,
                       dropout=args.dropout,
                       layer_type=args.layer_type,
                       analyze_mode=True).to('cpu')

    model.load_state_dict(torch.load(os.path.join(args.explain_path, 'model.pth'),
                                     map_location=torch.device('cpu'), weights_only=False))

    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        model_config=dict(
            mode='regression',
            task_level='edge',
            return_type='raw'),
        node_mask_type='attributes',
        edge_mask_type='object',
        threshold_config=dict(threshold_type='topk',
                              value=200))

    node_index = 0
    explanation = explainer(test_data.x,
                            test_data.edge_index,
                            edge_weight=test_data.edge_weight,
                            edge_label_index=test_data.edge_label_index,
                            index=node_index)

    assert 'node_mask' in explanation.available_explanations

    score = explanation.node_mask.squeeze(0).abs().sum(dim=0)
    # normalize the score
    score /= score.max()

    df = pd.DataFrame(score.cpu().detach().numpy())
    visualize_explanation(df.T)


if __name__ == '__main__':
    utils.seed_everything(args.seed)
    generate_explanation()
