import os
import gzip
import pandas as pd
from pathlib import Path
from ogb.graphproppred import GraphPropPredDataset

ROOT = Path("./datasets/molecule_datasets").resolve()

ALIASES = {
    "ogbg-moltox21": "tox21",
    "ogbg-molhiv": "HIV",
    "ogbg-molbbbp": "BBBP",
}

def find_mapping_file(ds_root: Path) -> Path:
    """Return the path to the OGB mapping file that contains SMILES."""
    mapping_dir = ds_root / "mapping"
    if mapping_dir.is_dir():
        # Common name across OGB mol datasets
        cand = mapping_dir / "mol.csv.gz"
        if cand.exists():
            return cand
        # Fallback: any csv in mapping/
        for p in mapping_dir.glob("*.csv*"):
            return p
    # Rare fallback: search the whole dataset folder
    for p in ds_root.rglob("*.csv*"):
        if "mapping" in str(p.parent).lower():
            return p
    raise FileNotFoundError(f"Could not find mapping CSV under {ds_root}")

def extract_smiles_to_processed(name: str, alias: str):
    print(f"\n=== {name} → {alias} ===")
    # 1) Trigger download (populates ROOT / ogbg_molxxx)
    _ = GraphPropPredDataset(name=name, root=str(ROOT))

    ds_dir = ROOT / name.replace("-", "_")          # e.g., ogbg_molhiv
    out_dir = ROOT / alias / "processed"            # e.g., HIV/processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "smiles.csv"

    # 2) Locate mapping file & read SMILES
    mapping_csv = find_mapping_file(ds_dir)
    print(f"• Using mapping file: {mapping_csv}")

    # pandas can read .gz directly
    df = pd.read_csv(mapping_csv, compression="infer")
    col = None
    for c in df.columns:
        if c.lower() == "smiles":
            col = c
            break
    if col is None:
        raise KeyError(f"No 'smiles' column found in {mapping_csv}. Columns: {list(df.columns)}")

    # 3) Write GraphMVP-style smiles.csv
    df[[col]].rename(columns={col: "smiles"}).to_csv(out_csv, index=False)
    print(f"✅ Wrote {len(df)} SMILES → {out_csv}")

if __name__ == "__main__":
    for ogb_name, alias in ALIASES.items():
        extract_smiles_to_processed(ogb_name, alias)
    print("\n🎉 Done. You can now run fine-tuning (you’re skipping GEOM).")
