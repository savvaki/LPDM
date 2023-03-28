import torch
import torch.nn as nn
from ldm.util import instantiate_from_config

class DiffusionPerceptualLoss(nn.Module):
    def __init__(self, model_config, mode='l1'):
        # This loads an LDM, but only the q sampling will be used as well as the internal model, we can ignore the first stage encoder here
        super().__init__()
        assert 'ckpt_path' in model_config, "ckpt_path is required for diff perceptual loss"
        assert 'max_loss_timestep' in model_config or 'timesteps' in model_config.params 
        
        model_config_trimmed = model_config.copy()
        ckpt_path = model_config_trimmed.pop('ckpt_path')
        self.max_loss_timestep = model_config_trimmed.pop('max_loss_timestep', model_config_trimmed.params.timesteps)
        self.diffusion_model = instantiate_from_config(model_config_trimmed)

        if mode == 'l1':
            self.criterion = nn.L1Loss()
        elif mode =='mse':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Unknown criterion: {mode}")

        self.custom_init_from_ckpt(ckpt_path)

        print("Freezing DiffusionPerceptualLoss unet")
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

    def custom_init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_sd = { k:v for k,v in sd.items() if k.startswith('model.diffusion_model')} # Ignore everything else that is not the diffusion model

        print("Warning, loading only the DiffusionWrapper model for DiffusionPerceptualLoss")
        missing, unexpected = self.diffusion_model.load_state_dict(new_sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, pred, target, t=None, cond=None):

        if t is None:
            t = torch.randint(0, self.max_loss_timestep, (pred.shape[0],), device=pred.device).long()

        eps =  torch.randn_like(pred, device=pred.device)

        # Use the same noise
        noised_pred = self.diffusion_model.q_sample(pred, t, noise=eps) # x_start, t, noise
        noised_target = self.diffusion_model.q_sample(target, t, noise=eps) # x_start, t, noise

        if cond is not None:
            noised_pred = torch.cat([noised_pred, cond], dim=1)
            noised_target = torch.cat([noised_target, cond], dim=1)

        eps1 = self.diffusion_model.apply_model(noised_pred, t, cond=None)
        eps2 = self.diffusion_model.apply_model(noised_target, t, cond=None)

        return self.criterion(eps1, eps2)


class DiffusionPerceptualRandNoiseNoTimestep(DiffusionPerceptualLoss):
    """Apply on a diffusion model that does not consider timesteps when denoising i.e. t = 0"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):

        if t is None:
            t = torch.randint(0, self.max_loss_timestep, (pred.shape[0],), device=pred.device).long()

        eps =  torch.randn_like(pred, device=pred.device)

        # Use the same noise
        noised_pred = self.diffusion_model.q_sample(pred, t, noise=eps) # x_start, t, noise
        noised_target = self.diffusion_model.q_sample(target, t, noise=eps) # x_start, t, noise

        if cond is not None:
            noised_pred = torch.cat([noised_pred, cond], dim=1)
            noised_target = torch.cat([noised_target, cond], dim=1)

        
        t_zero = torch.full((pred.shape[0],), 0, device=pred.device, dtype=torch.long)

        eps1 = self.diffusion_model.apply_model(noised_pred, t_zero, cond=None)
        eps2 = self.diffusion_model.apply_model(noised_target, t_zero, cond=None)

        return self.criterion(eps1, eps2)


class DiffusionAttention(DiffusionPerceptualRandNoiseNoTimestep):
    """Apply a diffusion model to pred_z_m with no noise added and return the result """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):
        model_in = torch.cat([pred, cond], dim=1) if cond is not None else pred
        t_zero = torch.full((pred.shape[0],), 0, device=pred.device, dtype=torch.long)
        pred_eps = self.diffusion_model.apply_model(model_in, t_zero, cond=None)
        return pred_eps


class DiffusionVariance(DiffusionPerceptualRandNoiseNoTimestep):
    """Apply a diffusion model to pred_z_m with no noise added and return the variance of the result"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):
        model_in = torch.cat([pred, cond], dim=1) if cond is not None else pred
        t_zero = torch.full((pred.shape[0],), 0, device=pred.device, dtype=torch.long)
        pred_eps = self.diffusion_model.apply_model(model_in, t_zero, cond=None)
        return pred_eps.var(dim=[2, 3]).mean() # Variance over (H, W) and then mean the result


class DiffusionPerceptualNoNoiseAdded(DiffusionPerceptualLoss):
    """Apply a diffusion model to pred_z_m as well as gt with no noise added and return criterion of the result"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):

        model_in_pred = torch.cat([pred, cond], dim=1) if cond is not None else pred
        model_in_target = torch.cat([target, cond], dim=1) if cond is not None else target

        t_zero = torch.full((pred.shape[0],), 0, device=pred.device, dtype=torch.long)

        pred_eps1 = self.diffusion_model.apply_model(model_in_pred, t_zero, cond=None)
        pred_eps2 = self.diffusion_model.apply_model(model_in_target, t_zero, cond=None)

        return self.criterion(pred_eps1, pred_eps2)

class DiffusionPerceptualNoGroundTruth(DiffusionPerceptualLoss):
    """Only compare to ground truth noise, ie. how well can a fixed diffusion model predict noise on the backbone prediction"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):

        if t is None:
            t = torch.randint(0, self.max_loss_timestep, (pred.shape[0],), device=pred.device).long()

        eps_gt =  torch.randn_like(pred, device=pred.device)

        # Use the same noise
        noised_pred = self.diffusion_model.q_sample(pred, t, noise=eps_gt) # x_start, t, noise

        if cond is not None:
            noised_pred = torch.cat([noised_pred, cond], dim=1)

        eps1 = self.diffusion_model.apply_model(noised_pred, t, cond=None)

        return self.criterion(eps1, eps_gt)

class DiffusionPerceptualMultiTimestepPredStart(DiffusionPerceptualLoss):
    """Apply the diffusion model for given values of t, predict the x0 and then supervise that result."""
    def __init__(self, *args, timesteps, **kwargs):
        self.timesteps = timesteps
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):
        losses = []
        for t_ in self.timesteps:
            model_in_pred = torch.cat([pred, cond], dim=1) if cond is not None else pred
            t_tensor = torch.full((pred.shape[0],), t_, device=pred.device, dtype=torch.long)
            pred_eps = self.diffusion_model.apply_model(model_in_pred, t_tensor, cond=None)
            denoised = self.diffusion_model.predict_start_from_noise(pred, t_tensor, pred_eps)
            losses.append(self.criterion(denoised, pred))
        return losses
    

class DiffusionDDPMPerceptualMultiTimestepPredStart(DiffusionPerceptualMultiTimestepPredStart):
    """Apply the diffusion model as a DDPM for given values of t, predict the x0 and then supervise that result."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):
        losses = []
        for t_ in self.timesteps:
            model_in_pred = torch.cat([pred, cond], dim=1) if cond is not None else pred
            t_tensor = torch.full((pred.shape[0],), t_, device=pred.device, dtype=torch.long)
            pred_eps = self.diffusion_model.model(model_in_pred, t_tensor) # Note a different API here
            denoised = self.diffusion_model.predict_start_from_noise(pred, t_tensor, pred_eps)
            losses.append(self.criterion(denoised, pred))
        return losses
    
class DiffusionDDPMPerceptualMultiTimestepPredStartSubtractCustomT(DiffusionPerceptualMultiTimestepPredStart):
    """Apply the diffusion model as a DDPM for given values of t, predict the x0 and then supervise that result."""
    def __init__(self, *args, pred_x0_t, **kwargs):
        self.pred_x0_t = pred_x0_t
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, t=None, cond=None):
        losses = []
        for t_ in self.timesteps:
            model_in_pred = torch.cat([pred, cond], dim=1) if cond is not None else pred
            t_tensor = torch.full((pred.shape[0],), t_, device=pred.device, dtype=torch.long)
            t_tensor_x0 = torch.full((pred.shape[0],), self.pred_x0_t, device=pred.device, dtype=torch.long)
            pred_eps = self.diffusion_model.model(model_in_pred, t_tensor) # Note a different API here
            denoised = self.diffusion_model.predict_start_from_noise(pred, t_tensor_x0, pred_eps) # Note that the amount of noise removed is not the same as what is detected 
            losses.append(self.criterion(pred, denoised))
        return losses

class DiffusionPerceptualTimestepCeiling(DiffusionPerceptualLoss):
    """Apply the diffusion model for all values of t up to a max value, predict the x0 and then supervise that result back the the original pred."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_loss_timestep is not None

    def forward(self, pred, target, t=None, cond=None):
        model_in_pred = torch.cat([pred, cond], dim=1) if cond is not None else pred
        t_tensor = torch.randint(0, self.max_loss_timestep, (pred.shape[0],), device=pred.device).long()
        pred_eps = self.diffusion_model.apply_model(model_in_pred, t_tensor, cond=None)
        denoised = self.diffusion_model.predict_start_from_noise(pred, t_tensor, pred_eps)
        return self.criterion(denoised, pred)
    
class DiffusionPerceptualTimestepCeilingCompareToTarget(DiffusionPerceptualLoss):
    """Apply the diffusion model for all values of t up to a max value (don't add noise), predict the x0 for both pred and target and then supervise that result."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_loss_timestep is not None

    def forward(self, pred, target, t=None, cond=None):
        t_tensor = torch.randint(0, self.max_loss_timestep, (pred.shape[0],), device=pred.device).long()

        model_in_pred = torch.cat([pred, cond], dim=1) if cond is not None else pred
        model_in_target = torch.cat([target, cond], dim=1) if cond is not None else target
        
        pred_eps = self.diffusion_model.apply_model(model_in_pred, t_tensor, cond=None)
        target_eps = self.diffusion_model.apply_model(model_in_target, t_tensor, cond=None)

        denoised_pred = self.diffusion_model.predict_start_from_noise(pred, t_tensor, pred_eps)
        denoised_target = self.diffusion_model.predict_start_from_noise(target, t_tensor, target_eps)

        return self.criterion(denoised_pred, denoised_target)