from ldm.modules.diffusionmodules.openaimodel import UNetModel, timestep_embedding
import torch
import torch.nn as nn


class EdgeUNet(UNetModel):
    def __init__(self, *args, **kwargs):
        print("Instantiating EdgeUNet")
        super().__init__(*args, **kwargs)

        self.feature_channels = [self.model_channels * m for  m in self.channel_mult]
        # self.feature_channels = [320, 640, 1280, 1280]
        self.feature_channels_indices = [2, 5, 8, 11]
        assert len(self.feature_channels) == len(self.feature_channels_indices)
        assert len(self.feature_channels) == len(self.channel_mult)
        assert not self.predict_codebook_ids

        self.edge_upsample = EdgeUpsample(channels_in=self.feature_channels, feature_indices=self.feature_channels_indices, latent_channels=self.out_channels)
        self.proj_edge = nn.Conv2d(self.out_channels, self.feature_channels[0], kernel_size=1, bias=False) # Batch norm?
        self.norm_reduction = nn.GroupNorm(num_groups=self.feature_channels[0] // 16, num_channels=self.feature_channels[0])
        self.act_edges = nn.SiLU(inplace=True)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        if self.num_scales is not None:
            emb = emb + self.scale_emb(kwargs['current_scale'])

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        # Now use hs to predict shape and concat shape for the max scale instead of original
        edge_prediction = self.edge_upsample(hs) # The same shape as z_me
        hs[0] = self.act_edges(self.norm_reduction(self.proj_edge(edge_prediction))) # The final concat is modified to the prediction

        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)

        return self.out(h), edge_prediction # Note: always returns the edge prediction





class EdgeUpsample(nn.Module):
    def __init__(self, channels_in, feature_indices, latent_channels) -> None:
        """
        Args:
            channels_in (List[int]): The number of channels of each feature block in he heirarchy in ascending order
            feature_indices (List[int]): The indices in the concat_feats list of the forward function where the channels_in feature blocks can be found.
        """
        super().__init__()

        channels_in = list(reversed(channels_in)) # Upsample from high level to low level e.g. 1280, 640, 320
        feature_indices = list(reversed(feature_indices))
        
        self.feature_indices = feature_indices

        self.gau_blocks = nn.ModuleList()
        for i in range(len(channels_in)):
            if i == (len(channels_in) - 1):
                self.gau_blocks.append(GAU(channels_high=channels_in[i], channels_low=channels_in[i], upsample=False)) # Don't upsample last block
                continue
            self.gau_blocks.append(GAU(channels_high=channels_in[i], channels_low=channels_in[i+1], upsample=True))


        self.proj_edge = nn.Conv2d(channels_in[-1], latent_channels, kernel_size=1, bias=False) 
        self.norm_reduction = nn.InstanceNorm2d(latent_channels) # Don't normalise accross all of the latent feature maps do each separately
        self.activation = nn.SiLU(inplace=True)
                
    def forward(self, concat_feats):
        """
        Args:
            concat_feats (List): A List of unet tensors, these will be selected at certain indices and plugged into the GAUs
        """
        
        block_index = 0
        for i in range(1, len(self.feature_indices)):
            if i == 1:
                fm_high = concat_feats[self.feature_indices[0]]
            fm_high = self.gau_blocks[block_index](fm_high, concat_feats[self.feature_indices[i]])
            block_index += 1
        fm_high = self.gau_blocks[block_index](fm_high, concat_feats[self.feature_indices[-1]])

        fm_high = self.activation(self.norm_reduction(self.proj_edge(fm_high))) # Project to same number of channels as z_me
        return fm_high


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, fms_high, fms_low):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :return: fms_att_upsample
        """

        fms_high_gp = self.global_pool(fms_high)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out