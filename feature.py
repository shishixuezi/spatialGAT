import os
import warnings
import pandas as pd

from args import args

"""
Extract the unified feature vector for meshes
"""


def population():
    path = os.path.join(args.data_path, 'meshFeature', 'night_population_shizuoka.csv')
    dataset = pd.read_csv(path, header=0, names=['KEY_CODE', 'population'])

    return dataset


def poi():
    path = os.path.join(args.data_path, 'meshFeature', 'poi')
    frame = []
    for filename in os.listdir(path):
        dataset = pd.read_csv(os.path.join(path, filename), encoding='shift-jis')
        dataset.drop(index=0, inplace=True)
        frame.append(dataset)
    dataset = pd.concat(frame).reset_index().drop(['index'], axis=1)
    dataset = dataset.astype('float')

    dataset['KEY_CODE'] = dataset['KEY_CODE'].astype('long')
    return dataset


def road_density(compact=True):
    path = os.path.join(args.data_path, 'meshFeature', 'road_density_shizuoka.csv')
    road = pd.read_csv(path)
    if compact:
        road['cross_1'] = road['N04_003'] + road['N04_004'] + road['N04_005'] + road['N04_006']
        road.drop(['N04_003', 'N04_004', 'N04_005', 'N04_006'], inplace=True, axis=1)
        road['cross_2'] = road['N04_009'] + road['N04_010'] + road['N04_011'] + road['N04_012']
        road.drop(['N04_009', 'N04_010', 'N04_011', 'N04_012'], inplace=True, axis=1)
        road['cross_3'] = road['N04_015'] + road['N04_016'] + road['N04_017'] + road['N04_018']
        road.drop(['N04_015', 'N04_016', 'N04_017', 'N04_018'], inplace=True, axis=1)
        road['cross_4'] = road['N04_021'] + road['N04_022'] + road['N04_023'] + road['N04_024']
        road.drop(['N04_021', 'N04_022', 'N04_023', 'N04_024'], inplace=True, axis=1)
        road['cross_5'] = road['N04_027'] + road['N04_028'] + road['N04_029'] + road['N04_030']
        road.drop(['N04_027', 'N04_028', 'N04_029', 'N04_030'], inplace=True, axis=1)
        road['cross_6'] = road['N04_033'] + road['N04_034'] + road['N04_035'] + road['N04_036']
        road.drop(['N04_033', 'N04_034', 'N04_035', 'N04_036'], inplace=True, axis=1)
        road['cross_7'] = road['N04_039'] + road['N04_040'] + road['N04_041'] + road['N04_042']
        road.drop(['N04_039', 'N04_040', 'N04_041', 'N04_042'], inplace=True, axis=1)
        road['cross_8'] = road['N04_051'] + road['N04_052'] + road['N04_053'] + road['N04_054']
        road.drop(['N04_051', 'N04_052', 'N04_053', 'N04_054'], inplace=True, axis=1)
        # width undetermined
        road.drop(['N04_045', 'N04_046', 'N04_047', 'N04_048', 'N04_049', 'N04_050'], inplace=True, axis=1)
        return road
    else:
        return road


def railway_passenger():
    rp = pd.read_csv(os.path.join(args.data_path, 'meshFeature', 'railway_passenger_shizuoka.csv'))
    rp.drop(columns=['S12_009', 'S12_013', 'S12_017', 'S12_021', 'S12_025', 'S12_029', 'S12_033', 'S12_037'],
            inplace=True, axis=1)

    return rp


def get_feature(is_thousand=False):
    """
    summarize feature
    :return:
    """
    warnings.filterwarnings('ignore', message="^Columns.*")
    h = population()
    p = poi()
    r = road_density()
    rp = railway_passenger()

    result = pd.merge(h, p, on='KEY_CODE', how='outer').fillna(0)
    result = pd.merge(result, r, on='KEY_CODE', how='outer').fillna(0)
    result = pd.merge(result, rp, on='KEY_CODE', how='outer').fillna(0)

    if is_thousand:
        result['KEY_CODE'] = result['KEY_CODE'] // 10
        result = result.groupby(['KEY_CODE']).sum().reset_index()

    return result


if __name__ == '__main__':
    all_feature = get_feature(is_thousand=False)
    all_feature.to_csv(os.path.join('result', 'input', 'all_feature.csv'), index=False)
