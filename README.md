## Description

Music stem separation based on a [pretrained transformer model],
with additional training and testing using the [RawStems] corpus.

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

1. Follow the last section to set up the dependencies.

2. Run [newmodel_training.ipynb](notebooks/newmodel_training.ipynb) on Google Colab.
   Make sure the required files are mounted at the correct path in your Google Drive.

## Expected runtime and hardware

Training is expected to take around 11 hours on a high-RAM A100 GPU.

[RawStems]: https://huggingface.co/datasets/yongyizang/RawStems
[bs-roformer-infer]: https://github.com/openmirlab/bs-roformer-infer
[pretrained transformer model]: https://huggingface.co/jarredou/BS-ROFO-SW-Fixed
