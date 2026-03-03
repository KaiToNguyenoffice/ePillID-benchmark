import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

from . import fast_MPN_COV_wrapper

logger = logging.getLogger(__name__)

MODERN_BACKBONES = {
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
    'maxvit_t',
}


def _build_modern_backbone(network: str, pretrained: bool, out_dim: int):
    """Build a modern torchvision backbone and return (backbone, feature_dim).

    The classifier head is replaced by an identity so the model outputs
    raw feature vectors.
    """
    weights = 'DEFAULT' if pretrained else None

    factory = getattr(models, network, None)
    if factory is None:
        raise ValueError(f"Unknown modern backbone: {network}")

    model = factory(weights=weights)

    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            feat_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
        else:
            feat_dim = model.classifier.in_features
            model.classifier = nn.Identity()
    elif hasattr(model, 'heads'):
        if isinstance(model.heads, nn.Sequential):
            feat_dim = model.heads[-1].in_features
            model.heads[-1] = nn.Identity()
        else:
            feat_dim = model.heads.head.in_features
            model.heads.head = nn.Identity()
    elif hasattr(model, 'head'):
        feat_dim = model.head.in_features
        model.head = nn.Identity()
    elif hasattr(model, 'fc'):
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Cannot determine classifier head for {network}")

    projection = nn.Linear(feat_dim, out_dim) if feat_dim != out_dim else nn.Identity()

    backbone = nn.Sequential(model, projection)
    logger.info("Modern backbone %s: feat_dim=%d -> out_dim=%d", network, feat_dim, out_dim)
    return backbone


class EmbeddingModel(nn.Module):
    """Unified embedding model supporting both classic CNN + bilinear pooling
    (via fast-MPN-COV) and modern backbones (EfficientNet, ConvNeXt, ViT, Swin).

    For classic backbones (resnet*, densenet*), the original fast-MPN-COV
    pipeline with optional bilinear pooling is used.
    For modern backbones, the torchvision pretrained model is used directly
    with a linear projection layer.
    """

    def __init__(self, network='resnet18', pooling='CBP', dropout_p=0.5,
                 cont_dims=2048, pretrained=True, middle=1000, skip_emb=False):
        super(EmbeddingModel, self).__init__()

        self.network = network
        self.is_modern = network in MODERN_BACKBONES
        self.out_features = cont_dims

        if self.is_modern:
            self.base_model = _build_modern_backbone(network, pretrained, cont_dims)
        else:
            self.base_model = fast_MPN_COV_wrapper.get_model(
                arch=network, repr_agg=pooling,
                num_classes=cont_dims, pretrained=pretrained)

        self.dropout = nn.Dropout(p=dropout_p)
        if skip_emb:
            self.emb = None
        else:
            self.emb = nn.Sequential(
                nn.Linear(cont_dims, middle),
                nn.BatchNorm1d(middle, affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(middle, cont_dims),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        if self.emb is not None:
            x = self.emb(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


if __name__ == '__main__':
    print("=== Classic backbone ===")
    model = EmbeddingModel(network='resnet18', cont_dims=2048)
    inp = torch.randn(4, 3, 224, 224)
    out = model(inp)
    print(f"resnet18 output: {out.shape}")

    print("\n=== Modern backbone (EfficientNet V2 S) ===")
    model2 = EmbeddingModel(network='efficientnet_v2_s', cont_dims=2048)
    out2 = model2(inp)
    print(f"efficientnet_v2_s output: {out2.shape}")

    print("\n=== Modern backbone (Swin-T) ===")
    model3 = EmbeddingModel(network='swin_t', cont_dims=2048)
    out3 = model3(inp)
    print(f"swin_t output: {out3.shape}")
