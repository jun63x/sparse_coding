import torch
import scipy.stats
from tqdm import trange


class SparseNetLearning:
    def __init__(
        self,
        images,
        patch_size,
        basis_func_num,
        e_step_iter_num,
        lr,
        iter_num
            ):
        self.images = images
        self.patch_size = patch_size
        self.basis_func_num = basis_func_num
        self.vec_lambda = torch.ones(self.basis_func_num)
        self.e_step_iter_num = e_step_iter_num
        self.lr = lr
        self.iter_num = iter_num
        self.mat_phi = torch.randn(
            self.patch_size[0]*self.patch_size[1],
            self.basis_func_num,
            )

    def sample_patch(self):
        sampled_patch = self.images.sample_patch(self.patch_size).flatten()
        return torch.Tensor(scipy.stats.zscore(sampled_patch))

    def exe_e_step(self, y):
        x = torch.zeros(self.basis_func_num)
        for _ in range(self.e_step_iter_num):
            deriv1 = torch.matmul(
                torch.transpose(self.mat_phi, 0, 1),
                y-torch.matmul(self.mat_phi, x)
                ) - self.vec_lambda*x
            deriv2 = - torch.matmul(
                torch.transpose(self.mat_phi, 0, 1),
                self.mat_phi
                ) - torch.diag(self.vec_lambda)
            x = x - torch.matmul(torch.inverse(deriv2), deriv1)
            mat_w = torch.inverse(-deriv2)
            self.vec_lambda = 1 / torch.diag(mat_w + torch.ger(x, x))
        return mat_w, x

    def exe_m_step(self, y, mat_w, x):
        deriv = torch.ger(y, x) - \
            torch.matmul(self.mat_phi, mat_w+torch.ger(x, x))
        self.mat_phi = self.mat_phi + self.lr*deriv

    def norm_bases(self):
        self.mat_phi = self.mat_phi / self.mat_phi.norm(p=2)

    def train(self):
        for i in trange(self.iter_num, ncols=50):
            y = self.sample_patch()
            self.exe_m_step(y, *self.exe_e_step(y))
            self.norm_bases()

    def get_basis_func_list(self):
        return [
            vec.reshape(
                self.patch_size
                ).numpy() for vec in torch.transpose(self.mat_phi, 0, 1)
            ]