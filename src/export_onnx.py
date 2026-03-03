"""
ONNX Export Utility for ePillID models.

Exports the embedding model (backbone + embedding MLP) to ONNX format
for deployment.  The classification heads (BinaryHead, MarginHead) are
**not** exported because inference is embedding-distance based.

Usage:
    python export_onnx.py --model_path model.pth --appearance_network resnet50 \
        --output embedding_model.onnx
"""

import argparse
import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_embedding_model(model_path, appearance_network, pooling, embedding_dim,
                           n_classes, output_path, img_size=224,
                           train_with_side_labels=True, opset_version=14):
    """Export the embedding sub-model to ONNX.

    Only the ``EmbeddingModel`` (backbone -> pooling -> MLP) is exported so
    the resulting ONNX graph takes an image tensor and returns a fixed-size
    embedding vector.
    """
    from models.embedding_model import EmbeddingModel
    from models.multihead_model import MultiheadModel

    emb_model = EmbeddingModel(
        network=appearance_network, pooling=pooling,
        dropout_p=0.0, cont_dims=embedding_dim, pretrained=False)

    full_model = MultiheadModel(
        emb_model, n_classes, train_with_side_labels=train_with_side_labels)

    state_dict = torch.load(model_path, map_location='cpu')
    full_model.load_state_dict(state_dict)
    full_model.eval()

    embedding_only = full_model.embedding_model
    embedding_only.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    logger.info("Exporting embedding model to %s (opset %d)", output_path, opset_version)
    torch.onnx.export(
        embedding_only,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Export complete: %s (%.1f MB)", output_path, file_size_mb)

    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        logger.info("ONNX model validation passed")
    except ImportError:
        logger.warning("onnx package not installed – skipping validation")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export ePillID embedding model to ONNX')
    parser.add_argument('--model_path', required=True, help='Path to .pth state dict')
    parser.add_argument('--appearance_network', default='resnet50')
    parser.add_argument('--pooling', default='GAvP')
    parser.add_argument('--embedding_dim', type=int, default=2048)
    parser.add_argument('--n_classes', type=int, required=True,
                        help='Number of pill type classes (before side-label doubling)')
    parser.add_argument('--train_with_side_labels', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--output', default='embedding_model.onnx')
    parser.add_argument('--opset', type=int, default=14)
    args = parser.parse_args()

    export_embedding_model(
        model_path=args.model_path,
        appearance_network=args.appearance_network,
        pooling=args.pooling,
        embedding_dim=args.embedding_dim,
        n_classes=args.n_classes,
        output_path=args.output,
        img_size=args.img_size,
        train_with_side_labels=bool(args.train_with_side_labels),
        opset_version=args.opset,
    )
