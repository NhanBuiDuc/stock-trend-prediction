import torch.nn as nn
import torch
import torch.nn.functional as F


class Unified_Adversarial_Loss(nn.Module):
    def __init__(self, bce_weight=3, mse_weight=1, hard_negative_weight=2):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.hard_negative_weight = hard_negative_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.threshold = 0.5

    def forward(self, output, target):
        batch = target.shape[0]
        # Calculate the binary cross-entropy loss for the classification part
        bce_loss = self.bce_loss(output[:, :1], target[:, :1])

        # Calculate the mean squared error loss for the regression part
        mse_loss = self.mse_loss(output[:, 1:], target[:, 1:])

        # Calculate the hard negative loss for the classification part
        # target_cls_int = (target[:, :1] > self.threshold)  # Convert the one-hot encoding to integers
        output_cls_int = (output[:, :1] >= self.threshold)  # Convert the one-hot encoding to integers
        hard_negative_mask = (target[:, :1] != output_cls_int)  # Create a mask for hard negative examples
        hard_negative_mask = hard_negative_mask.float()  # Convert the mask to a float tensor

        # hard_negative_mask[hard_negative_mask == 1] = 2
        # hard_negative_mask[hard_negative_mask == 0] = 1
        # bce_loss = self.hard_negative_weight * hard_negative_mask * bce_loss

        # Combine the three losses

        loss = (self.bce_weight * bce_loss + self.mse_weight * mse_loss)
        # loss = loss.mean()
        # print("loss", loss)
        # print("bce", bce_loss)
        # print("mse_loss", mse_loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)
