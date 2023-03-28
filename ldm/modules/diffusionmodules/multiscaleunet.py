import torch
from torch import nn
from ldm.modules.diffusionmodules.openaimodel import UNetModel, Downsample
from einops import repeat

def resize(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear')

def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)
    x_scaled = torch.nn.functional.interpolate(x, size=y_size, mode='bilinear')
    return x_scaled

class Resize(nn.Module):
    def __init__(self, size) -> None: 
        super().__init__()  
        if isinstance(size, int):
            size = (size, size)
        self.to_size = size

    def forward(self, img):
        return  nn.functional.interpolate(img, size=self.to_size, mode='bilinear')

def create_preprocessing_block(scale, destination_z_size, in_channels=4, reduction_dim=256, dilation_rates=[2, 4, 6]):
    modules = [
        AtrousSpatialPyramidPoolingModule(in_channels, reduction_dim, dilation_rates),
        nn.Conv2d(5 * reduction_dim, reduction_dim, kernel_size=1, bias=False), # Project dims back to reduction_dim
    ]
    if scale == 2.0:
        modules.append(Downsample(256, use_conv=True))             
    elif scale != 1.0:
        modules.append(Resize(destination_z_size))
    return nn.Sequential(*modules)

class MultiScaleUnet(UNetModel):

    def __init__(self, dilation_rates, z_channels=4, cond_scales=[0.5, 1.0, 2.0], aspp_reduction_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = super().forward # Use the forward of the parent
        self.scales = sorted(cond_scales)

        self.aspp_preprocess = nn.ModuleList()
        self.reduce_to_z_x = nn.ModuleList()
        for scale in self.scales:
            self.aspp_preprocess.append(create_preprocessing_block(scale=scale, destination_z_size=kwargs['image_size'], in_channels=z_channels, reduction_dim=aspp_reduction_dim, dilation_rates=dilation_rates))
            self.reduce_to_z_x.append(nn.Sequential(
                nn.Conv2d(aspp_reduction_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256), # Note: Batch norm not a good idea with noise
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, z_channels, kernel_size=1, bias=False))
            )

        self.emb_scale = nn.Embedding(len(cond_scales), self.time_embed[0].out_features)

        # Scale-attention prediction head
        num_scales = len(self.scales)
        self.scale_attn = nn.Sequential(
            nn.Conv2d(num_scales * 256, 256, kernel_size=3,
                        padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_scales, kernel_size=1, bias=False))

        

        self.initialize_weights(self.scale_attn)
        for m in self.reduce_to_z_x:
            self.initialize_weights(m)
        for m in self.aspp_preprocess:
            self.initialize_weights(m)

    def initialize_weights(self, *models):
        """
        Initialize Model Weights
        """
        for model in models:
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    

    def forward(self, z_m, timesteps=None, context=None, return_attn = False):
        assert context is not None
        assert 1.0 in context.keys(), 'expected one of scales to be 1.0'

        eps = {}
        preprocess_concat_feats = []

        for scale_index, scale in enumerate(context.keys()):
            z_x = context[scale]
            z_x = self.aspp_preprocess[scale_index](z_x)
            preprocess_concat_feats.append(z_x)
            z_x = self.reduce_to_z_x[scale_index](z_x)
            # Run through backbone with scale embedding
            in_concat = torch.cat([z_m, z_x], dim=1)
            curr_scale = repeat(torch.tensor([scale_index]), '1 -> b', b=z_m.shape[0]).to(z_m.device)
            eps[scale] = self.backbone(in_concat, timesteps, context=None, current_scale=curr_scale) # Note no context here

        preprocess_concat_feats = torch.cat(preprocess_concat_feats, 1)
        attn_tensor = self.scale_attn(preprocess_concat_feats) # Attention is predicted jointly off of muli-scale features and ultimately used to weigh the noise

        output = None
        for idx, scale in enumerate(self.scales):
            attn = attn_tensor[:, idx:idx+1, :, :]
            if output is None:
                output = eps[scale] * attn # broadcast the 2D attention map across 3D block
            else:
                output += eps[scale] * attn

        if return_attn:
            return output, attn_tensor

        return output


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, rates=(2, 4, 6)):
        super().__init__()

        # if output_stride == 8:
        #     rates = [2 * r for r in rates]
        # elif output_stride == 16:
        #     pass
        # else:
        #     raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,
                                    bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = nn.functional.interpolate(img_features, size=x_size[2:], mode='nearest')
        out = img_features
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


