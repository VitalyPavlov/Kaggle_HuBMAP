import torch
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses import DiceLoss, BinaryFocalLoss


class BCE_LDICE(_Loss):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode="binary", log_loss=True)

    def forward(self, *input):
        return self.bce_loss(*input) + self.dice_loss(*input)


class BCE_DICE(_Loss):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode="binary", log_loss=False)

    def forward(self, *input):
        return self.bce_loss(*input) + self.dice_loss(*input)


class BCE_FOCAL(_Loss):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.focal_loss = BinaryFocalLoss(alpha=0.1)

    def forward(self, *input):
        return self.bce_loss(*input) + self.focal_loss(*input)


class BCE_FOCAL_02(_Loss):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.focal_loss = BinaryFocalLoss(alpha=0.2)

    def forward(self, *input):
        return self.bce_loss(*input) + self.focal_loss(*input)


class BCE_FOCAL_03(_Loss):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.focal_loss = BinaryFocalLoss(alpha=0.3)

    def forward(self, *input):
        return self.bce_loss(*input) + self.focal_loss(*input)