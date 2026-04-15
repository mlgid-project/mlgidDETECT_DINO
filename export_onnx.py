import argparse
from pathlib import Path

import torch
from models.registry import MODULE_BUILD_FUNCS


DEFAULT_CHECKPOINT = "/path-to-checkpoint/checkpoint.pth"


def build_model_main(args):
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

# Required because raw DINO/DETR models return dict outputs which are not stable for ONNX export. This wrapper exposes only the tensor outputs.
class DINOOnnxWrapper(torch.nn.Module):
    """Expose only tensor outputs for stable ONNX export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        out = self.model(images)
        return out["pred_logits"], out["pred_boxes"]


def export_model_to_onnx(model, input_shape, output_path, device, opset=16, use_dynamic_axes=False):
    model.eval()
    # ONNX export must run on CPU regardless of the training device.
    # Custom CUDA kernels (e.g. MSDeformAttnFunction) can segfault when the
    # ONNX tracer executes a forward pass on the GPU.  The exported .onnx file
    # is device-agnostic and runs on CUDA at inference time without any change.
    export_device = torch.device("cpu")
    wrapper = DINOOnnxWrapper(model).to(export_device).eval()
    dummy_input = torch.randn(input_shape, device=export_device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = {
        "input_names": ["images"],
        "output_names": ["pred_logits", "pred_boxes"],
        "opset_version": opset,
        "do_constant_folding": True,
        "verbose": False,
    }
    if use_dynamic_axes:
        export_kwargs["dynamic_axes"] = {
            "images": {0: "batch_size"},
            "pred_logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        }

    with torch.no_grad():
        # Keep this call unwrapped to preserve the exact exporter traceback.
        torch.onnx.export(wrapper, dummy_input, str(output_path), **export_kwargs)
    print(f"Model exported to {output_path}")


def _to_namespace(maybe_args):
    if isinstance(maybe_args, dict):
        return argparse.Namespace(**maybe_args)
    return maybe_args


def main():
    parser = argparse.ArgumentParser(description="Export DINO model checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint.pth")
    parser.add_argument("--output", default="", help="Output ONNX file path")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", default="cpu", help="Device to load the model on for export (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch axis")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_args = _to_namespace(checkpoint["args"])

    model, _, _ = build_model_main(model_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    input_shape = (args.batch, args.channels, args.height, args.width)
    export_model_to_onnx(
        model,
        input_shape=input_shape,
        output_path=Path(args.output),
        device=device,
        opset=args.opset,
        use_dynamic_axes=args.dynamic,
    )


if __name__ == "__main__":
    main()