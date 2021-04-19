import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR


def main(config, args):
    Mission = TextSR(config, args)

    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tsrn', choices=['tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn', 'san', 'pcan'])
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='../dataset/lmdb/str/TextZoom/test/medium/', help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--edge', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')

    # san
    parser.add_argument('--n_resgroups', type=int, default=20, help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--n_resblocks', type=int, default=10, help='number of residual blocks')
    parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--scale', default=2, help='super resolution scale')

    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)
