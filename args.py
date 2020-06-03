from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument("--mat_path", type=str)
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--patch_width", default=12, type=int)
    parser.add_argument("--patch_hight", default=12, type=int)
    parser.add_argument("--basis_func_num", default=100, type=int)
    parser.add_argument("--e_step_iter_num", default=3, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--iter_num", default=20000, type=int)
    return parser.parse_args()
