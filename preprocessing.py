import os
import warnings
import pandas as pd

from args import args
from feature import get_feature


def get_od(path, **kwargs):
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings('ignore', message="^Columns.*")

    df = pd.read_csv(path, header=0,
                     dtype={'origin_key': str, 'dest_key': str, 'num': float, 'city': str})

    if 'city_name' in kwargs:
        mask = (df['city'] == kwargs['city_name'])
        df = df.loc[mask]
        assert not df.empty

    df = df.drop(['city'], axis=1)

    return df


def get_node_feature(df_edges, feature):
    df_unique_nodes = pd.Series(pd.concat([df_edges['origin_key'],
                                           df_edges['dest_key']]).unique(), name='KEY_CODE').to_frame()
    return df_unique_nodes.merge(feature, on='KEY_CODE', how='left')


def preprocessing(**kwargs):
    dict_path = {'od': os.path.join(args.data_path, 'od.csv')}
    all_feature = get_feature(is_thousand=args.is_thousand).astype({'KEY_CODE': str})

    assert 'city_name' in kwargs
    edges = {'od': get_od(dict_path['od'], city_name=kwargs['city_name'])}
    xs = {'mesh': (get_node_feature(edges['od'], all_feature))}

    return {'edges': edges,
            'xs': xs}
