"""Template baseline adapter (bench.models.template_baseline, unchanged from Track 1)."""
import json

from bench.models.template_baseline import TemplateBaseline


def build(ctx):
    pat = float(ctx.spec.params.get("pat", 0.3828125))
    f = ctx.weights_dir / "template.json"          # written by tools/fit_baselines.py (refit on CapnoBase)
    if f.exists():
        pat = float(json.loads(f.read_text())["pat"])
    return TemplateBaseline(pat=pat)
