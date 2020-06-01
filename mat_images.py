import random

import scipy.io


class MatImages:
    def __init__(self, mat_path):
        mat_data = scipy.io.loadmat(mat_path)
        self.image_list = mat_data['IMAGES']

    def _sample_image(self):
        img_num = random.randint(0, self.image_list.shape[2]-1)
        return self.image_list[:, :, img_num]

    def sample_patch(self, patch_size):
        sampled_image = self._sample_image()
        width_start = random.randint(0, self.image_list.shape[0]-patch_size[0])
        hight_start = random.randint(0, self.image_list.shape[1]-patch_size[1])
        return sampled_image[
            width_start:width_start+patch_size[0],
            hight_start:hight_start+patch_size[1]
            ]
