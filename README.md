## Description

TODO (one line)

## Setup

1. Install Python dependencies

   ```bash
   # with uv
   uv sync
   source .venv/bin/activate

   # or with pip
   pip install -r requirements.txt
   ```

2. Download the [pretrained transformer model] using [bs-roformer-infer]

   ```bash
   bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou
   ```

3. Download [RawStems] to the `rawstems` directory

## Commands to reproduce the results

TODO

## Expected runtime and hardware

TODO

[RawStems]: https://huggingface.co/datasets/yongyizang/RawStems
[bs-roformer-infer]: https://github.com/openmirlab/bs-roformer-infer
[pretrained transformer model]: https://huggingface.co/jarredou/BS-ROFO-SW-Fixed
