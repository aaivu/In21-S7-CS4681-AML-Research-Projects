# Dataset Generation

The dataset is generated using **convex optimization** for orbital rendezvous trajectories.  
Each trajectory consists of:
- **ROE states (6D)**  
- **RTN states (6D)**  
- **Actions (Δv in RTN frame)**  
- **Auxiliary parameters:** time, orbital elements, reward-to-go (RTG), constraint-to-go (CTG)

## Run Dataset Generator
```bash
python src/dataset-generation/generate_data_art_physicsaware.py
```

## Output
- Directory: `dataset-seed-{seed}/`
  - `dataset-rpod-cvx.npz`
  - `dataset-rpod-cvx-param.npz`

*Tip:* Modify parameters like `TARGET_SAMPLES`, `RNG_SEED`, or `HORIZON_GRID` at the top of the generator code file to customize dataset size and range.