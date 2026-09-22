"""
Purpose : Make an int8 version of the exported ONNX model (weights stored as 8-bit integers instead
          of 32-bit floats, ~4x smaller) so we can measure what quantisation costs in accuracy and
          what it buys in file size / phone speed. This is ablation D of Task 9.
Depends : onnxruntime (quantization tools), onnx.
Question you should be able to answer: what does quantisation trade for size and speed?
          Weights lose precision (256 levels per tensor instead of a float range), so accuracy can
          drop slightly; in exchange the file is ~4x smaller to download and integer maths is often
          faster on a CPU/WASM backend, which is what the phone actually uses.
Usage   : training/.venv/bin/python training/scripts/quantize.py public/models/best.onnx public/models/best_int8.onnx
"""
import sys
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        # Shape inference + constant folding first: dynamic quantisation is much better behaved
        # on a graph whose tensor shapes are known.
        quant_pre_process(str(src), str(pre))
        quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QUInt8)
    mb = lambda p: p.stat().st_size / 1e6
    print(f"wrote {dst}  {mb(dst):.1f} MB  (fp32 {mb(src):.1f} MB, {mb(src) / mb(dst):.1f}x smaller)")


if __name__ == "__main__":
    main()
