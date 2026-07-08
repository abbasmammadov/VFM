from abc import ABC, abstractmethod
from .resizer import Resizer
import torch.nn.functional as F
from .fastmri_utils import fft2c_new, ifft2c_new
from .motion_blur import Kernel

import torch
import torch.nn as nn
import scipy
import numpy as np
from functools import partial


# abstract base class for operators
class Operator(ABC):
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        pass
    
    # New method to refresh randomness
    def update(self, seed=None):
        pass

    def transpose(self, y):
        return y

    def measure(self, x):
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)


# identity or denoising --> class: 0
class Identity(Operator):
    def __init__(self, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device

    def __call__(self, x):
        return x.to(self.device)

    def transpose(self, y):
        return y.to(self.device)


# random inpainting --> class: 1
class InpaintingRandom(Operator):
    def __init__(self, resolution=256, prob_range=(0.3, 0.7), device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.resolution = resolution
        self.prob_range = prob_range
        self.device = device
        self.update()

    def update(self, seed=None):
        self.mask = self.retrieve_random(seed=seed).to(self.device)

    def __call__(self, x):
        return x.to(self.device) * self.mask

    def transpose(self, y):
        return y.to(self.device)
    
    def retrieve_random(self, seed=None):
        rng = np.random.default_rng(seed)
        if isinstance(self.prob_range, float):
            prob = self.prob_range
        else:
            l, h = self.prob_range
            prob = rng.uniform(l, h)
        total = self.resolution ** 2
        mask_vec = torch.ones([1, self.resolution * self.resolution])
        samples = rng.choice(self.resolution * self.resolution, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask = mask_vec.view(1, 1, self.resolution, self.resolution).repeat(1, 3, 1, 1)
        return mask.to(self.device)
    

# box inpainting --> class: 2
class InpaintingBox(Operator):
    def __init__(self, resolution=256, box_range=(32, 128), device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.resolution = resolution
        self.box_range = box_range
        self.device = device
        self.update()

    def update(self, seed=None):
        self.mask = self.retrieve_box(seed=seed).to(self.device)

    def __call__(self, x):
        return x.to(self.device) * self.mask

    def transpose(self, y):
        return y.to(self.device)
    
    def retrieve_box(self, seed=None):
        rng = np.random.default_rng(seed)

        if isinstance(self.box_range, int):
            mask_h, mask_w = self.box_range, self.box_range
        else:
            l, h = self.box_range
            l, h = int(l), int(h)
            mask_h = rng.integers(l, h)
            mask_w = rng.integers(l, h)

        margin = (16, 16)
        t = rng.integers(margin[0], self.resolution - margin[0] - mask_h)
        l = rng.integers(margin[1], self.resolution - margin[1] - mask_w)

        mask = torch.ones([1, 3, self.resolution, self.resolution])
        mask[..., t:t + mask_h, l:l + mask_w] = 0

        return mask.to(self.device)


# super-resolution --> class: 3
class SuperResolution(Operator):
    def __init__(self, resolution=256, scale_factor=4, device='cuda', sigma=0.05):
        super().__init__(sigma)
        in_shape = [1, 3, resolution, resolution]
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.device = device

    def __call__(self, x):
        return self.down_sample(x).to(self.device)
    
    def transpose(self, y):
        return self.up_sample(y).to(self.device)


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=61, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2), nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3))
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            pass
        else:
            raise Exception("Only Gaussian and Motion blur is implemented")

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        k = k.to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


# gaussian_blur --> class: 4
class GaussianBlur(Operator):
    def __init__(self, kernel_size=61, intensity=3.0, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian', kernel_size=kernel_size, std=intensity, device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, x):
        return self.conv(x).to(self.device)


# motion blur --> class: 5
class MotionBlur(Operator):
    def __init__(self, kernel_size=61, intensity=0.5, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.intensity = intensity

        self.conv = Blurkernel(blur_type='motion', kernel_size=kernel_size, std=intensity, device=device).to(device)
        self.conv.requires_grad_(False)
        self.update()

    def update(self, seed=None):
        kernel_obj = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.intensity, seed=seed)
        k_tensor = torch.from_numpy(kernel_obj.kernelMatrix).type(torch.float32)
        self.conv.update_weights(k_tensor)

    def __call__(self, x):
        return self.conv(x).to(self.device)
    

# HDR imaging --> class: 6
class HighDynamicRange(Operator):
    def __init__(self, device='cuda', scale=2, sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.scale = scale

    def __call__(self, data):
        return torch.clip((data * self.scale), -1, 1).to(self.device)


# phase retrieval --> class: 7
class PhaseRetrieval(Operator):
    def __init__(self, device='cuda',oversample=2.0, resolution=256, sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.pad = int((oversample / 8.0) * resolution)

    def __call__(self, x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
        amplitude = fft2_m.abs()
        # amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
        return amplitude.to(self.device)
    
    def transpose(self, y):
        y_complex = y.type(torch.complex64).to(self.device)
        y_real_view = torch.view_as_real(y_complex)
        ifft_real_view = ifft2c_new(y_real_view)
        ifft_complex = torch.view_as_complex(ifft_real_view)
        ifft_real = ifft_complex.real
        x_hat_cropped = ifft_real[..., self.pad:-self.pad, self.pad:-self.pad]
        x_out = (x_hat_cropped - 0.5) * 2.0
        # x_out = torch.clamp(x_out, -1.0, 1.0)
        return x_out.to(self.device)
    # def transpose(self, y):
    #     return y[..., self.pad:-self.pad, self.pad:-self.pad]
    
