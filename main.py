import torch
from loguru import logger

from mat_images import MatImages
from sparse_net_learning import SparseNetLearning
from basis_func_list import BasisFuncList
from args import get_args


def main():
    args = get_args()
    mat_images = MatImages(args.mat_path)
    patch_size = (args.patch_width, args.patch_hight)
    with torch.no_grad():
        snl = SparseNetLearning(
            mat_images,
            patch_size,
            args.basis_func_num,
            args.e_step_iter_num,
            args.lr,
            args.iter_num,
            args.gpu
            )
        logger.info('Train start')
        snl.train()
    logger.info('Image save start')
    basis_func_list = BasisFuncList(snl.get_basis_func_list())
    basis_func_list.save(args.img_dir)


if __name__ == "__main__":
    main()
