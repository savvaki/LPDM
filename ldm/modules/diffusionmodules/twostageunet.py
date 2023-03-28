from ldm.modules.diffusionmodules.openaimodel import timestep_embedding
from ldm.util import instantiate_from_config
import torch
import torch.nn as nn


class TwoStageUNet(nn.Module):
    def __init__(self, unet_diff_config, unet_recon_config, **kwargs):
        print("Instantiating TwoStageUnet")
        super().__init__()
        self.unet_diff = instantiate_from_config(unet_diff_config)
        # Extract conditioning again for the second stage 
        self.latent_dim = unet_recon_config['params']['out_channels']
        # self.slice_from = None
        if unet_recon_config['params']['out_channels'] + self.latent_dim  != unet_recon_config['params']['in_channels']:
            raise NotImplementedError("Additional conditioning needs to be implemented ")

        self.unet_recon = instantiate_from_config(unet_recon_config)
        if kwargs:
            raise NotImplementedError(f"Unexpected kwargs found for TwoStageUNet: {kwargs}")

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...], [N x C x ...]: noise_pred, mask_pred
        """

        x, x_clean = x.chunk(2, dim=1)
        noise_pred = self.unet_diff(x, timesteps=timesteps, context=context, y=y, **kwargs)
        # Recondition the second phase with same conditioning.
        stage_two_input = torch.cat([noise_pred, x_clean], dim=1)
        mask_pred = self.unet_recon(stage_two_input, timesteps=timesteps, context=None, y=None, **kwargs)
        return noise_pred, mask_pred


class TwoStageUNetConditional(nn.Module):
    def __init__(self, unet_diff_config, unet_recon_config, **kwargs):
        print("Instantiating TwoStageUNetConditional")
        super().__init__()
        self.unet_diff = instantiate_from_config(unet_diff_config)
        # Extract conditioning again for the second stage 
        self.latent_dim = unet_diff_config['params']['out_channels']
        # self.slice_from = None
        if unet_recon_config['params']['out_channels'] + self.latent_dim  != unet_recon_config['params']['in_channels']:
            raise NotImplementedError("Additional conditioning needs to be implemented ")

        self.unet_recon = instantiate_from_config(unet_recon_config)
        if kwargs:
            raise NotImplementedError(f"Unexpected kwargs found for TwoStageUNet: {kwargs}")

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...], [N x C x ...]: noise_pred, mask_pred
        """

        stage_two_cond = x[:, :self.latent_dim, :, :]
        edge_pred = self.unet_diff(x, timesteps=timesteps, context=context, y=y, **kwargs)
        # Recondition the second phase with same conditioning.
        stage_two_input = torch.cat([edge_pred, stage_two_cond], dim=1)
        mask_pred = self.unet_recon(stage_two_input, timesteps=timesteps, context=context, y=y, **kwargs)
        return edge_pred, mask_pred


class NoDiffUNet(nn.Module):
    def __init__(self, unet_diff_config, unet_recon_config, **kwargs):
        print("Instantiating NoDiffTwoStageUnet")
        super().__init__()
        self.unet_diff = instantiate_from_config(unet_diff_config)
        self.unet_recon = instantiate_from_config(unet_recon_config)
        if kwargs:
            raise NotImplementedError(f"Unexpected kwargs found for TwoStageUNet: {kwargs}")

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...], [N x C x ...]: None, mask_pred
        """

        stage_one_out = self.unet_diff(x, timesteps=timesteps, context=context, y=y, **kwargs)
        # Recondition the second phase with same conditioning.
        mask_pred = self.unet_recon(stage_one_out, timesteps=timesteps, context=None, y=None, **kwargs)
        return None, mask_pred


class NoDiffUNetOneStage(nn.Module):
    def __init__(self, unet_recon_config, ckpt_path=None, **kwargs):
        print("Instantiating NoDiffUNetOneStage. This model has a single unet from z_x to z_m.")
        super().__init__()
        self.unet_recon = instantiate_from_config(unet_recon_config)
        if kwargs:
            raise NotImplementedError(f"Unexpected kwargs found for TwoStageUNet: {kwargs}")

        if ckpt_path:
            self.custom_init_from_ckpt(ckpt_path)

    def custom_init_from_ckpt(self, path, only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        prefix_to_remove = 'model.diffusion_model.'
        new_sd = {k:v for k,v in sd.items() if 'model.diffusion_model.' in k}       
        sd = { s[len(prefix_to_remove):] if s.startswith(prefix_to_remove) else k:v for s, v in new_sd.items()}
        print("Warning, custom-extracted state_dict keys for NoDiffUnetStage")

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...], [N x C x ...]: None, mask_pred
        """

        stage_one_out = self.unet_recon(x, timesteps=timesteps, context=context, y=y, **kwargs)
        return None, stage_one_out