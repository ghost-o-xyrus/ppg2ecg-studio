"""P2E-WGAN: bench.models.trained.P2EWGAN (unchanged from Track 1) with the network run by ONNX Runtime.

setup() exports the network from the authors' PyTorch code and our weights, and checks the export.
"""
from studio.engine import trained

ID = "p2e_wgan"


def build(ctx):
    return trained.onnx_adapter(ID, ctx)


def setup(ctx, say):
    trained.export(ID, ctx, say)
