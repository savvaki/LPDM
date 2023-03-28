# Must run from within the 'scripts' directory
import os, sys
import PIL
import glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
import torchvision.transforms as T
import argparse
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath(os.path.join('..', 'external', 'taming-transformers')))
sys.path.append(os.path.abspath(os.path.join('..', 'external', 'clip')))
from ldm.util import instantiate_from_config

def tensor_to_pil(t, mode='RGB'):
    if t.ndim == 4:
        assert t.shape[0] == 1
        t = t[0]
    t = torch.clamp((t + 1.0) / 2.0, min=0.0, max=1.0)
    img = 255. * rearrange(t.detach().cpu().numpy(), 'c h w -> h w c')
    return Image.fromarray(img.astype(np.uint8), mode=mode).convert('RGB')
    
def pil_to_tensor_in_range(t):
    return T.ToTensor()(t) * 2.0 - 1.0

def clamp_ldm_range(t):
    return torch.clamp(t, -1.0, 1.0)

def ldm_range_to_rgb_range(t):
    t = clamp_ldm_range(t)
    return (t + 1.0) / 2.0

def get_diff_loss_prediction(diffloss_model, z_m, z_x):
    t_ = torch.full((z_x.shape[0],), 0).long()
    return diffloss_model.diffusion_model.apply_model(torch.cat([z_m, z_x],dim=1), t_, cond=None).detach()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def load_model_given_name(name, device = torch.device('cpu')):
    config_path = os.path.join('../checkpoints',  f'{name}.yaml')
    config = OmegaConf.load(config_path)
    if 'first_stage_config' in config.model.params and not config.model.params.first_stage_config.params.ckpt_path.startswith('..') :
        config.model.params.first_stage_config.params.ckpt_path = os.path.join('..', config.model.params.first_stage_config.params.ckpt_path)
    model = load_model_from_config(config, f'{config_path[:-5]}.ckpt') # Assume the config is the same name
    model = model.to(device)
    _ = model.eval()
    return model

seed_everything(0)

parser = argparse.ArgumentParser(prog = 'Denoise Config', description = 'Denoise folders of images based on a configuration file.')

parser.add_argument('-b', '--base_path', type=str, default='../configs/test/denoise.yaml', help='a string path to the test configuration file') 
parser.add_argument('-s', '--skip_large', action='store_true', help='a flag indicating whether to skip large images which are >2000 pixels in width or height') 
parser.add_argument('-c', '--cut_large', action='store_true', help='a flag indicating whether to cut large images which are >2000 pixels in width or height. Has no effect if skip_large is set.') 
parser.add_argument('-d', '--device', type=str, help='device to move the model and tensors to. Either cuda or cpu.', default='cpu') 
parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing images rather than skipping the denoising process.') 
args = parser.parse_args()

config_path = args.base_path    
config = OmegaConf.load(config_path)
subdir = os.path.join(config.write_path, config.ddpm_name, f'phi{config.phi}_s{config.s}')
os.makedirs(subdir, exist_ok=True)

print(f'Checking {config_path}...')
# Config has not been denoised yet
write_path = os.path.join(config.write_path, config.ddpm_name, f'phi{config.phi}_s{config.s}')

pred_paths, cond_paths = sorted(glob.glob(config.pred_path)), sorted(glob.glob(config.cond_path))
assert len(pred_paths) == len(cond_paths), f"Number of images in folders do not match: {len(pred_paths)} != {len(cond_paths)}"

for p, c in zip(pred_paths, cond_paths):
    assert os.path.splitext(os.path.basename(p))[0] == os.path.splitext(os.path.basename(c))[0], f'{os.path.basename(p)} != {os.path.basename(c)}'

print(f'Denoising {config_path}...')

device = torch.device(args.device)
ddpm = load_model_given_name(config.ddpm_name).to(device)

os.makedirs(write_path, exist_ok=True)

def pad_to_multiple(im, mul=16):
    h, w = im.shape[2], im.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    return torch.nn.functional.pad(im, (0, padw, 0, padh), mode='reflect')


with torch.no_grad():
    for p_path, c_path  in tqdm(zip(pred_paths, cond_paths), total=len(pred_paths)):
        save_path = os.path.join(write_path, f'{os.path.splitext(os.path.basename(p_path))[0]}.png')
        if os.path.exists(save_path) and not args.overwrite:
            print(f'File {save_path} already exists, skipping...')
            continue

        p, c = pil_to_tensor_in_range(Image.open(p_path)).unsqueeze(0), pil_to_tensor_in_range(Image.open(c_path)).unsqueeze(0)
        if  p.shape[-1] >= 2000 or p.shape[-2] >=2000:
                if args.skip_large:
                    print(f'Skipping large image of size: {p.shape}')
                    continue
                elif args.cut_large:
                    print(f'Cutting image of size: {p.shape}')
                    # Split large tensors in half:
                    split_point = p.shape[-1] // 2 # split on width
                    left_p = p[..., :split_point]
                    right_p = p[..., split_point:]
                    left_c = c[..., :split_point]
                    right_c = c[..., split_point:]
                    denoised_pieces = []
                    for slice_p, slice_c in [(left_p, left_c), (right_p, right_c)]:
                        t = torch.tensor([config.phi], dtype=torch.long).to(device)
                        try:
                            noise_pred = ddpm.model(torch.cat([slice_p, slice_c], dim=1), t).detach()
                            x0 = ddpm.predict_start_from_noise(slice_p, torch.tensor([config.s], device=device).long(), noise_pred).detach()
                        except:
                            h, w = slice_p.shape[-2], slice_p.shape[-1]
                            noise_pred = ddpm.model(torch.cat([pad_to_multiple(slice_p), pad_to_multiple(slice_c)], dim=1), t).detach()
                            x0 = ddpm.predict_start_from_noise(pad_to_multiple(slice_p), torch.tensor([config.s], device=device).long(), noise_pred).detach()
                            x0 = x0[..., :h, :w]
                        denoised_pieces.append(x0)
                    x0 = torch.cat(denoised_pieces, dim=-1) # Concat on width
                    x0 = clamp_ldm_range(x0)
                    tensor_to_pil(x0).save(save_path)
                    continue
        
        t = torch.tensor([config.phi], dtype=torch.long, device=device)
        try:
            noise_pred = ddpm.model(torch.cat([p, c], dim=1), t).detach()
            x0 = ddpm.predict_start_from_noise(p, torch.tensor([config.s], device=device).long(), noise_pred).detach()
        except:
            h, w = p.shape[-2], p.shape[-1]
            noise_pred = ddpm.model(torch.cat([pad_to_multiple(p), pad_to_multiple(c)], dim=1), t).detach()
            x0 = ddpm.predict_start_from_noise(pad_to_multiple(p), torch.tensor([config.s], device=device).long(), noise_pred).detach()
            x0 = x0[..., :h, :w]

        x0 = clamp_ldm_range(x0)
        tensor_to_pil(x0).save(save_path)

print("Denoising Complete!")