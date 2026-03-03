import argparse
import os
import json


def _strtobool(val):
    """Replacement for deprecated distutils.util.strtobool."""
    val = str(val).lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise argparse.ArgumentTypeError(f"invalid truth value {val!r}")


SUPPORTED_BACKBONES = [
    # Classic CNN (via fast-MPN-COV)
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet161', 'densenet201',
    # Modern backbones (torchvision direct)
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
    'maxvit_t',
]


def common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir', default="/mydata")
    parser.add_argument('--img_dir', default="classification_data")
    parser.add_argument('--supress_warnings', action='store_true')

    # Optimizer & scheduler
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'momentum', 'adamdelta'])
    parser.add_argument("--init_lr", type=float, default=1e-4, help='initial learning rate')
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout rate in the final layer')
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_factor", type=float, default=0.5, help='factor of decrease in the learning rate')
    parser.add_argument("--weight_decay", type=float, default=0.0, help='weight decay for optimizer')
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--add_persp_aug', default='1', type=_strtobool, help='enhanced augmentation with perspective')

    # Model architecture
    parser.add_argument('--appearance_network', default='resnet50', choices=SUPPORTED_BACKBONES)
    parser.add_argument('--pooling', default='GAvP', choices=['MPNCOV', 'CBP', 'BCNN', 'GAvP'],
                        help='pooling layer (only used for classic CNN backbones)')

    # Metric learning
    parser.add_argument("--metric_margin", type=float, default=1.0, help='margin for contrastive/triplet loss')
    parser.add_argument("--metric_embedding_dim", type=int, default=2048, help='embedding dimensionality')
    parser.add_argument('--train_with_side_labels', default='1', type=_strtobool,
                        help='treat front/back as different classes during training')
    parser.add_argument('--metric_simul_sidepairs_eval', default='1', type=_strtobool,
                        help='simulate per-pill side image pairs during evaluation')
    parser.add_argument('--sidepairs_agg', type=str, default="post_mean",
                        choices=['post_mean', 'post_max'],
                        help="aggregation method for front/back embeddings")
    parser.add_argument('--metric_evaluator_type', type=str, default="cosine",
                        choices=['euclidean', 'cosine', 'ann', 'faiss'],
                        help="distance/similarity computation method for evaluation")

    # Loss weights
    parser.add_argument("--ce_w", default=1.0, type=float, help='cross-entropy loss weight')
    parser.add_argument("--arcface_w", default=0.1, type=float, help='ArcFace loss weight')
    parser.add_argument("--contrastive_w", default=1.0, type=float, help='contrastive loss weight')
    parser.add_argument("--triplet_w", default=1.0, type=float, help='triplet loss weight')
    parser.add_argument("--focal_w", default=0.0, type=float, help='focal loss weight')
    parser.add_argument("--focal_gamma", type=float, default=0.0, help='gamma for focal loss')
    parser.add_argument("--circle_w", default=0.0, type=float, help='circle loss weight')
    parser.add_argument("--circle_m", default=0.25, type=float, help='circle loss relaxation margin')
    parser.add_argument("--circle_gamma", default=256.0, type=float, help='circle loss scale factor')

    # Augmentation
    parser.add_argument('--aug_backend', default='imgaug', choices=['imgaug', 'albumentations'],
                        help='augmentation library to use')

    parser.add_argument('--load_mod')
    parser.add_argument('--results_dir', default="classification_results")
    parser.add_argument('--load_config', help='load a pre-defined config from a json file')

    return parser


def nocv_parser():
    parser = common_parser()

    parser.add_argument('--all_imgs_csv',
                        default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    parser.add_argument('--val_imgs_csv',
                        default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    parser.add_argument('--test_imgs_csv',
                        default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv")
    parser.add_argument('--label_encoder',
                        default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/label_encoder_pytorch131.pickle")

    return parser


def cv_parser():
    parser = common_parser()

    parser.add_argument('--folds_csv_dir',
                        default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/",
                        help='folder that contains data-splits csv files and encoder')
    parser.add_argument('--all_img_src', default="all",
                        help='can be all pill id images or synth_date_crop')

    return parser


def load_config(args):
    print(f"Loading the predefined config: {args.load_config}")
    print("Warning: the command arguments will be overwritten by the predefined config.")

    with open(args.load_config, 'r', encoding='utf-8') as f:
        params = json.load(f)
    for k, v in params.items():
        setattr(args, k, v)
