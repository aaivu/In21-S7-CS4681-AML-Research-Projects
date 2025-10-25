import os
import pandas as pd
from ogb.graphproppred import GraphPropPredDataset

ROOT = os.path.join("datasets", "molecule_datasets")
os.makedirs(os.path.join(ROOT, "hiv", "raw"), exist_ok=True)

# Trigger OGB download (small)
ds = GraphPropPredDataset("ogbg-molhiv", root=ROOT)

# Get SMILES from mapping
map_csv = os.path.join(ROOT, "ogbg_molhiv", "mapping", "mol.csv.gz")
smiles = pd.read_csv(map_csv, compression="infer")["smiles"].tolist()

# Get labels in dataset order
labels = []
for i in range(len(ds)):
    _, y = ds[i]              # y is shape (1,)
    labels.append(int(y.item()))

out = pd.DataFrame({"smiles": smiles, "HIV_active": labels})
out_path = os.path.join(ROOT, "hiv", "raw", "HIV.csv")
out.to_csv(out_path, index=False)
print(f"✅ Wrote {out_path} with {len(out)} rows.")
