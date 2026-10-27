# ASR

ASR model (e.g., Whisper) fine-tuning and evaluation.

## Setup

```bash
pip install -r models/asr/requirements.txt
```

`ffmpeg` must be available on `PATH`.

## Data

```text
models/asr/datasets/
├── train_set/
│   ├── wav/
│   └── transcript/
├── val_set/
│   ├── wav/
│   └── transcript/
└── test_set/
    ├── wav/
    ├── transcript/
    └── translate/    # optional; required for translation evaluation
```

Audio files are matched to labels by filename stem; common formats are detected automatically.

## Train

```bash
python -m asr.main.trainer --config-path models/asr/configs/tiny.yaml
bash models/asr/scripts/train.sh tiny
```

Training is transcription-only by default. Add `--multitask` to the CLI or set `MULTITASK=true` for the script to train transcription and translation.

## Evaluate

```bash
bash models/asr/scripts/evaluate.sh tiny
```

The script uses default dataset and checkpoint paths. Set `TASK=translate` to
evaluate translations; metrics default to CER/EMR for transcription and
BERTScore for translation.
