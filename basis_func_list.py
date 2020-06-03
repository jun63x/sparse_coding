import matplotlib.pyplot as plt
from tqdm import tqdm


class BasisFuncList:
    def __init__(self, basis_func_list):
        self.basis_func_list = basis_func_list

    def save(self, img_dir):
        for i, func in tqdm(
            enumerate(self.basis_func_list),
            total=len(self.basis_func_list),
            ncols=50
                ):
            plt.imsave(
                img_dir+'/patch'+str(i).zfill(3)+'.png',
                func,
                cmap='binary'
                )
