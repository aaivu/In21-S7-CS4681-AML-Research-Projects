# icews_loader_and_train_local.py
import argparse
import os
import torch
from torch.utils.data import Dataset
from torchdrug import data as td_data, core

# 🔑 use memory-based model & task
from src.memory_temporal_nbfnet import NeuralBellmanFordNetworkTemporal
from src.memory_temporal_tasks import TemporalKnowledgeGraphCompletionMemory


# ---------------------------
# Local Dataset Loader
# ---------------------------


def load_local_icews(data_dir="data/WIKI", valid_ratio=0.01, seed=42):
    """
    Load ICEWS18 from local txt files and split train into train/valid.
    Expected format per line: head<TAB>relation<TAB>tail<TAB>timestamp
    Files: train.txt, test.txt
    """
    def read_file(fname, after, before):
        triples = []
        with open(fname, "r") as f:
            count=0
            for line in f:
                count += 1
                # if count == 100:
                #     break
                parts = line.strip().split("\t")
                # print(parts)
                # if len(parts) != 4:
                

                # # Skip empty/malformed lines
                #     continue
                h, r, t, ts = parts[0], parts[1], parts[2], parts[3]
                if int(ts) >= after  and int(ts) < before:
                    # continue 3090
                    triples.append((int(h), int(t), int(r), int(ts)))
        return torch.tensor(triples, dtype=torch.long)
    full_train = read_file(os.path.join(data_dir, "train.txt") ,0, 210)
    valid = read_file(os.path.join(data_dir, "valid.txt"),211, 221)
    test = read_file(os.path.join(data_dir, "test.txt"),222, 231)


    # Split train into train/valid by time
    # Sort by timestamp (column 3)
    sorted_train, indices = torch.sort(full_train[:, 3])
    full_train = full_train[indices]
    split_idx = int(len(full_train) )
    train = torch.utils.data.Subset(full_train, range(0, split_idx))
    # valid = torch.utils.data.Subset(full_train, range(split_idx, len(full_train)))
    # test = torch.utils.data.Subset(test, range(0, len(test)))


    # Compute relation count directly
    print(f"Full train size: {full_train.size()}, Test size: { len(test)}")

    all_relations = torch.cat([full_train[:,2],valid[:,2], test[:,2]])
    num_relation = int(torch.max(all_relations)) + 1  
    print(f"num_relation {num_relation}")

    ds = {
     "train": full_train[train.indices],
"valid": valid,
        "test": test
    }
    return ds , num_relation


class TemporalKGBenchmark(Dataset):
    """
    Temporal KG dataset for training, validation, and testing.
    For validation/test, the graph includes all historical events up to current timestamp.
    """
    def __init__(self, ds, split="train"):
        """
        ds: dict with keys 'train', 'valid', 'test' containing torch.LongTensor of shape (num_triples, 4)
        split: 'train', 'valid', or 'test'
        """
        self.split = split

        if split == "train":
            # For training: use only train triples
            self.data = ds["train"]
            self.graph_triples = self.data
        elif split == "valid":
            # For validation: include all training triples + validation triples
            self.data = ds["valid"]
            # self.graph_triples = torch.cat([ds["train"], self.data], dim=0)
            self.graph_triples = torch.cat([ds["train"]], dim=0)
        elif split == "test":
            # For test: include all training + validation triples
            self.data = torch.utils.data.Subset(ds["test"], range(0, len(ds["test"])))
            self.graph_triples = torch.cat([ds["train"], ds["valid"], ds["test"]], dim=0)
        else:
            raise ValueError(f"Unknown split {split}")

        # Compute number of entities and relations
        overall_graph =   torch.cat([ds["train"], ds["valid"] , ds["test"]], dim=0)
        all_entities = torch.cat([overall_graph[:,0], overall_graph[:,1]])
        all_relations = overall_graph[:,2]
        self.num_entity = int(torch.max(all_entities)) + 1
        self.num_relation = int(torch.max(all_relations)) + 1 
        print(F"loader edges : {self.num_relation}")

        # Build graph with all historical edges
        h = self.graph_triples[:,0]
        t = self.graph_triples[:,1]
        r = self.graph_triples[:,2]
        edge_list = torch.stack([h, t, r], dim=-1)
        edge_weight = torch.ones(edge_list.size(0))

        self.graph = td_data.Graph(
            edge_list=edge_list,
            edge_weight=edge_weight,
            num_node=self.num_entity,
            num_relation=self.num_relation
        )

        # Assign edge timestamps
        with self.graph.edge():
            self.graph.edge_time = self.graph_triples[:,3].clone()
        if split == "train":
            self.data = self.data[-10000:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]


def collate_quadruples(batch):
    return torch.stack(batch, dim=0)


# ---------------------------
# Train / Eval
# ---------------------------

def main(
    data_dir="data/WIKI",
    hidden_dims=(256, 256),
    message_func="rotate",
    time_encode_dim=32,
    time_decay="exp",
    time_half_life=300,
    time_window=None,
    batch_size=8, # 16
    lr=5e-4,
    max_epoch=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    evaluate_only=True,
    checkpoint_path="/home/sajeenthiranp/KG/fork/NBFNet/t-nbfnet/data/WIKI/model_epoch_2.pt",
    #  checkpoint_path=None,

):

    ds, num_relation = load_local_icews(data_dir)
    train_dataset = TemporalKGBenchmark(ds, split="train")
    valid_dataset = TemporalKGBenchmark(ds, split="valid")
    test_dataset = TemporalKGBenchmark(ds, split="test")

    print(
        f"[DEBUG] Train set size: {len(train_dataset)}, Valid set size: {len(valid_dataset)}, Test set size: {len(test_dataset)}"
    )
    print("[DEBUG] Initializing model...")

    model = NeuralBellmanFordNetworkTemporal(
        input_dim=16,
        hidden_dims=[16, 16, 16, 16, 16, 32],
        num_relation=num_relation,
        message_func=message_func,
        aggregate_func="sum",
        short_cut=True,
        layer_norm=True,
        activation="relu",
        concat_hidden=False,
        num_mlp_layer=12,
        dependent=True,
        memory_time_dim=32,
        remove_one_hop=False,
        time_encode_dim=time_encode_dim,
        time_decay=time_decay,
        time_half_life=time_half_life,
        time_window=time_window,
    ).to(device)

    print(
        f"[DEBUG] Initialized model with {sum(p.numel() for p in model.parameters())} parameters."
    )

    print("[DEBUG] Creating task and moving to device...")
    task = TemporalKnowledgeGraphCompletionMemory(model).to(device)

    print("[DEBUG] Creating engine...")
    optimizer = torch.optim.Adam(task.parameters(), lr=lr, weight_decay=1e-5)
    engine = core.Engine(
        task,
        train_set=train_dataset,
        valid_set=valid_dataset,
        test_set=test_dataset,
        batch_size=batch_size,
        gpus=[0] if device.startswith("cuda") else [],
        optimizer=optimizer,
    )

    # load checkpoint if needed
    if checkpoint_path and os.path.exists(checkpoint_path):
    
        if evaluate_only:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[DEBUG] Loaded checkpoint from {checkpoint_path}, epoch {checkpoint['epoch']}")
            # print("[DEBUG] Running evaluation only...")
            # valid_results = engine.evaluate("valid")
            # print(f"[DEBUG] Validation results: {valid_results}")
            test_results = engine.evaluate("test")
            print(f"[DEBUG] Test results: {test_results}")
            return

    print("[DEBUG] Starting training loop...")
    for epoch in range(max_epoch):
        print(f"[DEBUG] Epoch {epoch} begin")
        engine.train()
        print(f"[DEBUG] Epoch {epoch} training finished, starting evaluation...")
        results = engine.evaluate("valid")
        print(f"[DEBUG] Epoch {epoch}: Validation results: {results}")

        checkpoint_file = os.path.join(data_dir, f"model_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "results": results,
            },
            checkpoint_file,
        )
        print(f"[DEBUG] Saved checkpoint: {checkpoint_file}")

    print("[DEBUG] Training finished. Evaluating on test set...")
    test_results = engine.evaluate("test")
    print(f"[DEBUG] Test results: {test_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate WIKI temporal model")
    parser.add_argument("--data-dir", type=str, default="data/WIKI",
                        help="Path to dataset directory")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train: train model; test: load checkpoint and run evaluation")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training/evaluation")
    parser.add_argument("--max-epoch", type=int, default=5,
                        help="Maximum number of epochs")
    parser.add_argument("--checkpoint", type=str, default="data/WIKI/model_epoch_2.pt",
                        help="Path to checkpoint to load (optional)")
    args = parser.parse_args()

    # map mode to evaluate_only flag
    print(f"Mode: {args.mode}")
    evaluate_only = args.mode == "test"
    print(f"Evaluate only: {evaluate_only}")

    main(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        evaluate_only=evaluate_only,
        checkpoint_path=args.checkpoint,
    )