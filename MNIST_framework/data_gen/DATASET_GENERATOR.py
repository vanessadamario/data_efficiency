import numpy as np
from PIL import Image
from retina.retina import warp_image


class DatasetGenerator(object):
    def __init__(self, data, output_dim=28, scenario=1, noise_var=None, common_dim=200):
        """ DatasetGenerator initialization.
        :param data: original dataset, MNIST
        :param output_dim: the dimensionality for the first transformation
        :param scenario: one of the paradigm proposed [1, 2, 4]
        :param noise_var: useful in paradigm 1, 4
        :param common_dim: dimensionality of output for scenario 4
        """
        self.data = data
        self.output_dim = output_dim
        self.scenario = scenario

        if noise_var is None:
            noise_var = 2e-1
        self.noise_var = noise_var

        n_samples, dim1, dim2 = self.data.shape
        # here we want to split

        self.n_samples = n_samples
        self.dim1 = dim1
        self.dim2 = dim2

        self.common_dim = common_dim  # we upscale and then add noise

        self.edge = int((self.output_dim - self.dim1) / 2)
        if self.scenario == 4:
            self.edge = int((self.common_dim - self.output_dim) / 2)

        self.output = None

    def add_noise_and_std(self):
        """ Add noise to the original image and standardize the entire image.
        The pixels for this image are between values [0,1].
        We generate the larger image, where the noise is such to be positive.
        We then standardize every image, so that its pixels distribution become Gaussian.
        """
        out = self.noise_var * np.abs(np.random.randn(self.n_samples,
                                                      2 * self.edge + self.dim1,
                                                      2 * self.edge + self.dim2))
        out[:, self.edge:self.edge+self.dim1, self.edge:self.edge+self.dim2] = self.data

        out_std = np.zeros_like(out)
        mean_ = np.mean(out, axis=(1, 2))
        std_ = np.std(out, axis=(1, 2))
        for k_, (m_, s_) in enumerate(zip(mean_, std_)):
            out_std[k_] = (out[k_] - m_) / s_
        self.output = out_std
        return self

    def upscale_std(self):
        """
        Automatic PIL upscale of the image with standardization.
        """
        new_x = np.zeros((self.n_samples, self.output_dim, self.output_dim))

        for n_, old_image_ in enumerate(self.data):
            image = Image.fromarray(old_image_)
            tmp_x = image.resize(size=(self.output_dim, self.output_dim))
            tmp_std_x = (tmp_x - np.mean(tmp_x)) / np.std(tmp_x)
            new_x[n_] = tmp_std_x

        self.output = new_x

        return self

    def _upscale_no_std(self):
        """ Upscale for experiment 4 wo standardization
        """
        new_x = np.zeros((self.n_samples, self.output_dim, self.output_dim))
        for n_, old_image_ in enumerate(self.data):
            image = Image.fromarray(old_image_)
            new_x[n_] = image.resize(size=(self.output_dim, self.output_dim))
        self.dim1 = self.output_dim
        self.dim2 = self.output_dim
        return new_x

    def upscale_add_noise_std(self):
        upscaled_mnist = self._upscale_no_std()
        self.data = upscaled_mnist
        self.add_noise_and_std()

    def foveation(self):
        """ In the original implementation, the image is rescaled to a smaller dimension
        and then lifted to the original dimensions. We do not want to lose information.
        To prevent this we keep the scaling factor as it is, and we stick to the implementation:
            https://github.com/dicarlolab/retinawarp
        We assume here that the image has square dimension.
        :returns foveated_dataset: the output after foveation.
        """
        ret_img = np.zeros_like(self.data)
        for n_ in range(self.n_samples):
            ret_img[n_] = warp_image(self.data[n_], output_size=self.dim1, input_size=self.dim1)

        self.output = ret_img

    def run(self):
        if self.scenario == 1:
            self.add_noise_and_std()
        elif self.scenario == 2:
            self.upscale_std()
        elif self.scenario == 4:
            self.upscale_add_noise_std()
        else:
            raise ValueError('Nope')

