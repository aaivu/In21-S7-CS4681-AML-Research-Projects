# WavLM CTC experiments

This folder contains the training, evaluation, and decoding scripts used for the project.

## Files
- `config.py` — default model and path constants
- `dataset_loader.py` — utilities to load Librispeech via HF datasets or manifest CSVs
- `preprocess.py` — preprocess and save a processed dataset (optional)
- `train_ctc.py` — main training script (uses Hugging Face Trainer)
- `evaluate.py` — evaluate a saved checkpoint and print WER/CER
- `decode_with_kenlm.py` — decode logits using KenLM via pyctcdecode
- `utils.py` — data collator, metric helpers, saving utilities

## Quick start / Terminal commands

1. Install requirements:
```bash
pip install -r ../requirements.txt
```

2. (Optional) Preprocess and save processed dataset:
```bash
python preprocess.py \
  --model_name microsoft/wavlm-large \
  --split train.clean.100 \
  --output_path ../data/processed/train_clean_100 \
  --batch 8
```

3. Train (example — Experiment 2 config):
```bash
python train_ctc.py \
  --model_name microsoft/wavlm-large \
  --processor_name facebook/wav2vec2-base-960h \
  --output_dir ../results/experiment_2 \
  --train_split train.clean.100 \
  --eval_split validation.clean \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 4 \
  --lr 2e-4 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --dropout 0.2

```

Training logs and final metrics are printed to the terminal and saved under ../results/experiment_2.

4. Evaluate:
```bash
python eval.py --model_dir ../results/experiment_2
```

5. Decode with KenLM (make sure kenlm model at ../models/4gram.arpa exists):
```bash
python decode_with_kenlm.py \
  --model_dir ../results/experiment_3 \
  --kenlm_path ../models/4gram.arpa \
  --split test.clean \
  --alpha 0.5 \
  --beta 0.1 
```

## Notes
- These scripts print progress and final WER/CER metrics to terminal.
- Modify --model_name to other Hugging Face compatible models if desired.
- For KenLM: build the .arpa -> .bin optional pre-compile for faster loads. See KenLM docs.
- Important KenLM note: pip install kenlm may fail on some systems. Preferred approach:
```bash
# on Linux
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake ..
make -j4
# then from python, pip install https://github.com/kpu/kenlm/archive/master.zip
```

