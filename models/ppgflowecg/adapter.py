"""PPGFlowECG: bench.models.ppgflowecg (unchanged from Track 1), PyTorch on the CPU, fixed seed.

setup() keeps only what inference reads from the authors' checkpoints (the aligned ECG encoder and decoder
of the 3.8 GB VAE file, the EMA weights of the flow) and deletes the originals.
"""
import os

from bench.models.ppgflowecg import PPGFlowECG
from studio.engine.isolate import repo_imports

ID = "ppgflowecg"
CLASH = ("model", "utils", "engine")                       # module names other repositories use too


class PPGFlowECGIsolated(PPGFlowECG):
    def load(self):
        with repo_imports(self.repo_dir, CLASH):
            super().load()


def build(ctx):
    os.environ.setdefault("TQDM_DISABLE", "1")          # the sampler's progress bar is noise here
    p = ctx.spec.params
    m = PPGFlowECGIsolated(repo_dir=str(ctx.weights_dir / "repo"), vae_ckpt=str(ctx.weights_dir / "vae_inference.pth"),
                   flow_ckpt=str(ctx.weights_dir / "flow_inference.pt"), sampling_steps=int(p.get("sampling_steps", 10)),
                   device="cpu", seed=0, batch_size=int(p.get("batch_size", 64)))
    ctx.info["device"] = "CPU"
    return m


def setup(ctx, say):
    import torch
    d = ctx.weights_dir
    say("    keeping only the inference parts of the checkpoints ...")
    vae = torch.load(d / "VAE-iter-40000.pth", map_location="cpu", weights_only=True, mmap=True)
    torch.save({k: vae[k] for k in ("encoder_ecg", "decoder_ecg")}, d / "vae_inference.pth")
    del vae
    st = torch.load(d / "checkpoint-10.pt", map_location="cpu", weights_only=True, mmap=True)
    torch.save({"ema": {k: v for k, v in st["ema"].items() if k.startswith("ema_model.")}}, d / "flow_inference.pt")
    del st
    m = build(ctx)
    m.load()                                           # the reduced files must load exactly like the originals
    for f in ("VAE-iter-40000.pth", "checkpoint-10.pt"):
        (d / f).unlink()
