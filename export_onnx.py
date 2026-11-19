"""
export_onnx.py - run ONCE. Converts CLIP's image half into models/clip_vision.onnx,
which ONNX Runtime runs about 2x faster than PyTorch on the Pi 5
(~120 ms per crop instead of ~270 ms). Same model, same answers.

    python export_onnx.py

live_boxes_v2.py uses the file automatically if it exists (and falls back to
plain PyTorch if it doesn't). Needs: pip install onnx onnxruntime  (in the venv).
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers import CLIPModel
from live_boxes_v2 import CLIP_NAME, ONNX_FILE

model = CLIPModel.from_pretrained(CLIP_NAME).eval()


class ImageHalf(torch.nn.Module):
    """picture (n,3,224,224) -> embedding (n,512)"""
    def __init__(self):
        super().__init__()
        self.vision, self.proj = model.vision_model, model.visual_projection

    def forward(self, x):
        return self.proj(self.vision(pixel_values=x).pooler_output)


net = ImageHalf().eval()
x = torch.randn(2, 3, 224, 224)
ONNX_FILE.parent.mkdir(exist_ok=True)
torch.onnx.export(net, (x,), str(ONNX_FILE), input_names=["x"], output_names=["f"],
                  dynamic_axes={"x": {0: "n"}, "f": {0: "n"}}, opset_version=17, dynamo=False)

# check the exported file gives the same numbers as PyTorch
import onnxruntime as ort
got = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
with torch.no_grad():
    want = net(x).numpy()
print(f"Saved {ONNX_FILE} ({ONNX_FILE.stat().st_size / 1e6:.0f} MB), "
      f"max difference vs PyTorch: {np.abs(got - want).max():.6f} (should be < 0.001)")
