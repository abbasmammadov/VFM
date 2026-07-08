from PIL import Image
import numpy as np
from .measurements import Identity, InpaintingRandom, InpaintingBox, GaussianBlur, PhaseRetrieval, HighDynamicRange, SuperResolution, MotionBlur
import torch
from torch.nn.functional import interpolate


test_image = Image.open('./inverse_problems/test_imgs/test.png').convert('RGB')
test_image = test_image.resize((256, 256))
test_image = np.array(test_image).astype(np.float32) / 127.5 - 1.0  # scale to [-1, 1]
print(test_image.shape)
print(test_image.min(), test_image.max())
test_image = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, H, W)
print(test_image.shape, test_image.min(), test_image.max())

def test_operators():
    identity = Identity()
    inpaint_random = InpaintingRandom(prob_range=0.7, resolution=256, device='cpu')
    super_res = SuperResolution(scale_factor=4, resolution=256, device='cpu')
    inpaint_box = InpaintingBox(box_range=128, resolution=256, device='cpu')
    gaussian_blur = GaussianBlur(kernel_size=61, intensity=3.0, device='cpu')
    phase_retrieval = PhaseRetrieval(oversample=2.0, resolution=256, device='cpu')
    motion_blur = MotionBlur(kernel_size=61, intensity=0.5, device='cpu')
    hdr = HighDynamicRange(device='cpu', scale=2)

    operators = [identity, inpaint_random, inpaint_box, gaussian_blur, phase_retrieval, motion_blur, hdr, super_res]
    operator_names = ['Identity', 'InpaintingRandom', 'InpaintingBox', 'GaussianBlur', 'PhaseRetrieval', 'MotionBlur', 'HighDynamicRange', 'SuperResolution']
    for op, name in zip(operators, operator_names):
        print(f'Testing operator: {name}')
        output = op(test_image)
        output = output + 0.05 * torch.randn_like(output)  # add slight noise
        # output = op.transpose(output)
        if name == "PhaseRetrieval":
            output_tmp = interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
            tmp = (output_tmp - output_tmp.mean()) / output_tmp.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            output = tmp * 2 - 1

        print(f'Output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}')
        # also save
        output_image = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # if name == 'PhaseRetrieval':
        #     output_image = output_image.clip(0,1) * 2 - 1

        output_image = (output_image + 1.0) / 2.0 * 255.0
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        Image.fromarray(output_image).save(f'./inverse_problems/test_imgs/test_{name}.png')  

    # check reproducibility of box random
    motion_blur = MotionBlur(kernel_size=61, intensity=0.5, device='cpu')
    motion_blur.update(seed=32)
    output1 = motion_blur(test_image)
    output2 = motion_blur(test_image)
    motion_blur.update(seed=31)
    output3 = motion_blur(test_image)
    diff = torch.abs(output1 - output2).max().item()
    diff_2 = torch.abs(output1 - output3).max().item()
    print(f'Reproducibility test for  with seed 32, max difference: {diff}')
    print(f'Difference test for  with different seeds 32 and 31, max difference: {diff_2}')
    # let's save these outputs too
    for i, out in enumerate([output1, output2, output3]):
        output_image = out.squeeze(0).permute(1, 2, 0).detach().numpy()
        output_image = (output_image + 1.0) / 2.0 * 255.0
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        Image.fromarray(output_image).save(f'test_imgs/test_seed_{i}.png')

if __name__ == '__main__':
    test_operators()