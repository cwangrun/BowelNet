import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def dice_loss_PL(prob, target, epsilon=1e-7):
    """Computes the Sørensen–Dice loss.
      Args:
          target: a tensor of shape [B, C, D, H, W].
          prob: a tensor of shape [B, C, D, H, W]. Corresponds to
              the prob output of the model.
          epsilon: added to the denominator for numerical stability.
      Returns:
          dice: the Sørensen–Dice loss.
      """
    assert prob.size() == target.size(),   "the size of predict and target must be equal."

    intersection = torch.sum(prob * target, dim=[0, 2, 3, 4])
    union = torch.sum(prob + target, dim=[0, 2, 3, 4])
    dice = (2. * intersection / (union + epsilon)).mean()   # average over classes

    return 1 - dice


def dice_similarity(output, target, smooth=1e-7):
    """Computes the Dice similarity"""

    output = output.float()
    target = target.float()

    seg_channel = output.view(output.size(0), -1)     # (batch, D*H*W)
    target_channel = target.view(target.size(0), -1)

    intersection = (seg_channel * target_channel).sum(-1)
    union = (seg_channel + target_channel).sum(-1)
    dice = (2. * intersection) / (union + smooth)

    return torch.mean(dice)


def dice_score_partial(output, target, data_type):
    """Computes the Dice scores, given foreground classes"""

    assert data_type in ['fully_labeled', 'smallbowel', 'colon_sigmoid']

    result = torch.argmax(output, dim=1, keepdim=True)
    if data_type == 'fully_labeled':
        valid_class = [1, 2, 3, 4, 5]
    if data_type == 'smallbowel':
        valid_class = [4]
    if data_type == 'colon_sigmoid':
        valid_class = [2, 3]

    total_dice = []
    for c in valid_class:
        target_c = (target == c).long()
        output_c = (result == c).long()
        dice_c = dice_similarity(output_c, target_c)
        total_dice.append(dice_c.item())
    return total_dice


