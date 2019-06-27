from argparse import ArgumentParser

import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

from mat_images import MatImages
from sparse_net_learning import SparseNetLearning


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--mat_path", type=str)
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--patch_width", default=12, type=int)
    parser.add_argument("--patch_hight", default=12, type=int)
    parser.add_argument("--basis_func_num", default=100, type=int)
    parser.add_argument("--e_step_iter_num", default=3, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--iter_num", default=20000, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    mat_images = MatImages(args.mat_path)
    patch_size = (args.patch_width, args.patch_hight)
    snl = SparseNetLearning(
        mat_images,
        patch_size,
        args.basis_func_num,
        args.e_step_iter_num,
        args.lr,
        args.iter_num
        )
    logger.info('Train start')
    snl.train()
    logger.info('Image save start')
    basis_func_list = snl.get_basis_func_list()
    for i, func in tqdm(
        enumerate(basis_func_list),
        total=len(basis_func_list),
        ncols=50
            ):
        plt.imsave(
            args.img_dir+'/patch'+str(i).zfill(3)+'.png',
            func,
            cmap='binary'
            )


if __name__ == "__main__":
    main()
