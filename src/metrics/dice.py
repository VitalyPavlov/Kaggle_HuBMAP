import numpy as np
from collections import namedtuple


def get_metrics_custom(pred, true):
    dice_pos, dice_neg = [], []
    p = pred.reshape(-1,)
    t = true.reshape(-1,)

    if t.max() == 1:
        dice_pos.append((2 * (p * t).sum()) / (p.sum() + t.sum()))
        dice_neg.append(np.nan)
    else:
        dice_pos.append(np.nan)
        dice_neg.append(0 if p.max() == 1 else 1)

    return dice_pos, dice_neg


def get_metrics(pred, true):
    p = pred.reshape(-1,)
    t = true.reshape(-1,)

    t_sum = t.sum()
    p_sum = p.sum()
    if t_sum == 0 and p_sum == 0:
        return [2], [0]
    dice = (2 * (p * t).sum()) / (p_sum + t_sum + 1e-4)

    return [2 * dice], [0]