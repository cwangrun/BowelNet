import torch
import torch.nn.functional as F


def dice_loss(prob, target, epsilon=1e-7):
    """Computes the Sørensen–Dice loss.
      Args:
          target: a tensor of shape [B, 1, D, H, W].
          prob: a tensor of shape [B, C, D, H, W]. Corresponds to
              the prob output of the model.
          epsilon: added to the denominator for numerical stability.
      Returns:
          dice: the Sørensen–Dice loss.
      """
    # target_ = torch.cat((1 - target, target), dim=1)
    num_classes = prob.shape[1]
    target = F.one_hot(target.squeeze(1).long(), num_classes).permute(0, 4, 1, 2, 3)   # [B, c, D, H, W]
    target = target.type(prob.type())

    assert prob.size() == target.size(),   "the size of predict and target must be equal."

    intersection = torch.sum(prob * target, dim=[0, 2, 3, 4])
    union = torch.sum(prob + target, dim=[0, 2, 3, 4])
    dice = (2. * intersection / (union + epsilon)).mean()   # average over classes

    return 1 - dice


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


def dice_score_metric(input, target, epsilon=1e-7):
    result = torch.argmax(input, dim=1).unsqueeze(1)
    intersection = torch.sum(result * target)
    union = torch.sum(result) + torch.sum(target)
    dice = 2. * intersection / (union + epsilon)
    return dice
