import torch
import torch.nn as nn

class SegmentationAELoss(nn.Module):
    def __init__(self, kl_weight=1.0, ignore_index=-100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.kl_weight = kl_weight

    def forward(self, target, reconstructions, posteriors, split="train"):
        ce_loss = self.ce_loss(reconstructions, target)
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = ce_loss + self.kl_weight * kl_loss 
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
               "{}/kl_loss".format(split): kl_loss.detach().mean(),
               "{}/ce_loss".format(split): ce_loss.detach().mean(),
                }
        return loss, log
