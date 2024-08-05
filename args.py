import argparse

parser = argparse.ArgumentParser()

# ---------------------------File--------------------------- #
parser.add_argument('--data_path',          default='./data')
parser.add_argument('--explain_path',       default='./explain')


# ---------------------------Train--------------------------- #
parser.add_argument('--experiment_name',                        default=None)
parser.add_argument('--device',                                 default=None)
parser.add_argument('--seed',                      type=int,    default=42)
parser.add_argument('--hidden_channels',           type=int,    default=256)
parser.add_argument('--embedding_size',            type=int,    default=128)
parser.add_argument('--lr',                        type=float,  default=0.0001)
parser.add_argument('--lr_weight_decay',           type=float,  default=0.001)
parser.add_argument('--epochs',                    type=int,    default=20000)
parser.add_argument('--dropout',                   type=float,  default=0.3)
parser.add_argument('--heads',                     type=int,    default=4)
parser.add_argument('--layer_type',                             default='gat')
parser.add_argument('--num_layer',                 type=int,    default=3)
parser.add_argument('--is_scheduled',                           default=True)
parser.add_argument('--scheduler_step',            type=int,    default=5000)
parser.add_argument('--scheduler_gamma',           type=float,  default=0.95)
parser.add_argument('--early_stopper_patience',    type=int,    default=1000)
parser.add_argument('--early_stopper_delta',       type=float,  default=500.)
parser.add_argument('--clip_threshold',            type=float,  default=None)
parser.add_argument('--is_verbose',                             default=False)
parser.add_argument('--evaluate_epoch_interval',   type=int,    default=50)
parser.add_argument('--is_inductive',                           default=False)


# ---------------------------Feature------------------------- #
parser.add_argument('--target',                                 default='shizuoka')
parser.add_argument('--city',                      type=int,    default=1,
                    help='0 for Aoi, 1 for Suruga, 2 for Naka, 3 for Numazu, 4 for Fuji or 5 for Susono')


# ---------------------------Model--------------------------- #
parser.add_argument('--loss_type',                             default='weighted_focal_l1')
parser.add_argument('--is_thousand',                           default=False)


# ---------------------------Explain------------------------- #
parser.add_argument('--explain_threshold',      type=int,       default=200)

# ---------------------------Save--------------------------- #
parser.add_argument('--save_folder', default='./result')

args = parser.parse_args()
