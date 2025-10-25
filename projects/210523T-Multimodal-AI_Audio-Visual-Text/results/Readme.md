The project experiments ar ein the [experiments](../experiments)

## Splitting and Restoring Checkpoints

### Reason for Splitting `best_pt` Files
To manage large model checkpoints efficiently and avoid storing huge single files, the `best_pt` checkpoints for experiments are split into multiple smaller parts. This ensures smoother version control, faster file transfers, and easier reproducibility.

### How to Restore the Full Checkpoint
To restore the original checkpoint from the split parts,  can concatenate the files in the correct order. For example, in Python:

```python
import shutil

# List of split files in order
parts = [
    "best_part_aa",
    "best_part_ab",
    # add more parts if present
]

# Destination file
full_model = "best_model.pt"

with open(full_model, 'wb') as outfile:
    for part in parts:
        with open(part, 'rb') as infile:
            outfile.write(infile.read())
