from datasets import load_dataset
import json

# # Load the dataset
# dataset = load_dataset("linxy/ICEWS14", "all")

# # Access the splits
# train_data = dataset["train"]
# valid_data = dataset["validation"]
# test_data = dataset["test"]

# # Optionally, save the splits to disk
# train_data.to_json("train.json")
# valid_data.to_json("valid.json")
# test_data.to_json("test.json")



def convert_jsonl_to_txt(jsonl_file, txt_file):
    with open(jsonl_file, "r") as f_in, open(txt_file, "w") as f_out:
        for line in f_in:
            entry = json.loads(line)
            # Take first tail as gold
            h, r, tstamp = entry["query"]
            t = entry["answer"][0]
            f_out.write(f"{h}\t{r}\t{t}\t{tstamp}\n")

# Convert each split
convert_jsonl_to_txt("train.json", "train.txt")
convert_jsonl_to_txt("valid.json", "valid.txt")
convert_jsonl_to_txt("test.json", "test.txt")