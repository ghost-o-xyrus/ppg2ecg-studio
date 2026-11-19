"""CLEP-GAN: bench.models.trained.ClepGAN (unchanged from Track 1) with the network run by ONNX Runtime.

setup() exports the network from the authors' PyTorch code and our weights, and checks the export.
"""
from studio.engine import trained

ID = "clep_gan"


def build(ctx):
    return trained.onnx_adapter(ID, ctx)


def setup(ctx, say):
    trained.export(ID, ctx, say)
