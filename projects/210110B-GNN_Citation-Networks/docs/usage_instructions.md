# Student Usage Instructions

## Project Workflow at a Glance
1. Set up a clean Python environment and install the required packages.
2. Download and preprocess the citation network data.
3. Review and adjust the experiment configuration in `src/configs/h_unimp.yaml`.
4. Train the heterogeneous UniMP model and monitor validation metrics.
5. Export predictions, capture logs, and archive results under `results/`.

## 1. Prerequisites
- Python 3.9 or 3.10 (PaddlePaddle 2.6.1 CPU build works reliably on macOS/Ubuntu).
- `pip` >= 21.0 and a C/C++ toolchain (Xcode Command Line Tools on macOS, `build-essential` on Ubuntu).
- Optional: `tensorboard` for log inspection and `virtualenvwrapper` if you prefer managed environments.

> **Paddle on Apple Silicon:** Use the CPU wheel (`pip install paddlepaddle==2.6.1`) unless you already have a working GPU-enabled install. GPU runtime variables are disabled in the training script to avoid kernel crashes.

## 2. Environment Setup
```bash
cd projects/210110B-GNN_Citation-Networks
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Sanity check
python -c "import paddle; import pgl; print('Paddle', paddle.__version__)"
```

Keep the virtual environment activated whenever you run project scripts. If you install extra packages, pin the versions in `requirements.txt` so the experiment is reproducible.

## 3. Data Preparation
1. Open `data/dataset-link.txt` and download the archive. Extract it to retrieve `outputacm.txt`.
2. Place `outputacm.txt` inside `src/dataset-helper/`.
3. From the project root run:
   ```bash
   python src/dataset-helper/parse_citationnetwork.py
   python src/dataset-helper/build_graphs.py
   ```
   These scripts create a `processed/` directory containing features, labels, graphs, and metadata.
4. The training config expects the processed data under `src/dataset/processed/`. Either move the folder or update `data_dir` in the config (see next section). A minimal move looks like:
   ```bash
   mkdir -p src/dataset
   mv src/dataset-helper/processed src/dataset/
   ```
5. Verify the setup:
   ```bash
   ls src/dataset/processed
   # expected files: metadata.pkl, paper_features.npy, paper_labels.npy,
   # paper_citation_graph, author_paper_graph_src, author_paper_graph_dst, ...
   ```

If you regenerate the dataset, remove any stale `processed/` directory first to avoid mixing old and new metadata.

## 4. Configuration
The default experiment config lives in `src/configs/h_unimp.yaml`. Key fields:

| Field | Purpose |
|-------|---------|
| `data_dir` | Folder (relative to `src/`) that contains `processed/`. Adjust if you keep the data elsewhere. |
| `samples` | Fan-out per hop (string). `"15-10"` means 15 first-hop and 10 second-hop neighbors. |
| `batch_size`, `epochs`, `max_steps` | Core training loop controls - tune for resource limits. |
| `model.*` | Architecture knobs (relation-aware projections, hidden size, number of heads). |
| `output_path` | Where checkpoints, logs, and predictions are written. Defaults to `src/output/citationnetwork_runimp`. |
| `device`, `num_workers` | Runtime settings. Leave on CPU/0 workers for macOS unless you have tested multiprocessing. |

After editing the YAML, keep it under version control (`experiments/` directory contains notes for each iteration).

## 5. Training & Validation
Run the heterogeneous UniMP training loop from the project root:
```bash
python src/h_unimp_train.py --conf src/configs/h_unimp.yaml
```

What happens:
- The script forces a CPU-friendly environment, prepares the dataset, and builds the model specified by `config.model.name` (ensure it matches a module inside `src/models`).
- Checkpoints are saved to `<output_path>/model_epoch_*.pdparams`; the latest checkpoint is auto-loaded on restart.
- Scalar metrics (loss, accuracy) are logged to TensorBoard under `<output_path>/log`.

To run a quick validation-only pass on the latest checkpoint:
```bash
python src/h_unimp_train.py --conf src/configs/h_unimp.yaml --do_eval
```
This skips optimization, loads the newest checkpoint, and reports validation metrics to both the console and TensorBoard.

## 6. Prediction / Inference
- Using the training script wrapper:
  ```bash
  python src/h_unimp_train.py --conf src/configs/h_unimp.yaml --do_predict
  ```
  Saves `test_predictions.npy` under `<output_path>/predictions/`.

- Dedicated inference entry point (handy if you want a different `test_name`):
  ```bash
  python src/h_unimp_infer.py --conf src/configs/h_unimp.yaml
  ```
  Produces `<output_path>/predictions/{test_name}_predictions.npy` and the sampled indices.

Make sure the checkpoint you wish to use is present in `output_path`; otherwise the script starts from random weights.

## 7. Monitoring & Result Tracking
- Launch TensorBoard to follow training curves:
  ```bash
  tensorboard --logdir src/output
  ```
- Summaries for each experiment iteration live in `experiments/`. Keep that directory updated whenever you change hyperparameters.
- Archive key metrics and qualitative findings in `results/mid-evaluation-results.md` and `results/final-evaluation-results.md`.

## 8. Troubleshooting & Tips
- **Missing module errors:** Confirm `config.model.name` has a matching module inside `src/models/`. Import errors mean you need to expose the class in `src/models/__init__.py` before training.
- **Dataset not found:** Double-check the `processed/` folder path and that `metadata.pkl` exists. If the loader prints `Metadata file exists: False`, rebuild the dataset.
- **Slow or unstable training:** Reduce `batch_size`, shorten neighbor fan-out (`samples`), or lower `epochs`. Keep `num_workers=0` on macOS to avoid multiprocessing hangs.
- **Resume from a checkpoint:** Leave `output_path` untouched - `load_model` pulls the highest epoch automatically. Delete old checkpoints only if you really want to restart from scratch.

With the environment configured, data processed, and config tuned, you can iterate on model ideas quickly. Update the experiment notes and commit both code and documentation changes regularly so supervisors can follow your progress.
