"""Vision Transformer via timm (pretrained or from scratch). Image-only; `meta` is ignored."""
import torch.nn as nn


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=7, name="vit_small_patch16_224", pretrained=True, drop_path=0.1):
        super().__init__()
        import timm  # imported lazily so the SVM/CNN paths do not need it

        self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path)

    def forward(self, img, meta=None):
        return self.net(img)
