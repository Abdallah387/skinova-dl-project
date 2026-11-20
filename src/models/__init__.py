from src.models.cnn import CNNClassifier
from src.models.fusion import FusionNet
from src.models.vit import ViTClassifier


def build_model(cfg, meta_dim):
    m = dict(cfg["model"])
    kind = m.pop("type")
    if kind == "cnn":
        return CNNClassifier(dropout=m.get("dropout", 0.3))
    if kind == "fusion":
        return FusionNet(meta_dim, dropout=m.get("dropout", 0.3))
    if kind == "vit":
        return ViTClassifier(name=m.get("name", "vit_small_patch16_224"), pretrained=m.get("pretrained", True),
                             drop_path=m.get("drop_path", 0.1))
    raise ValueError(f"unknown model type {kind}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
