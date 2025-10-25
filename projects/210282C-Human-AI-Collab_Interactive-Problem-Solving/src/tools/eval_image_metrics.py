import argparse
import json
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from pathlib import Path


def compute_clip_score(model, processor, image: Image.Image, prompt: str, device: str):
    with torch.no_grad():
        inputs = processor(images=image, text=[prompt], return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        sim = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
        return float(sim.item())


def retrieval_rank(model, processor, image: Image.Image, prompt: str, negatives: list, device: str):
    texts = [prompt] + negatives
    with torch.no_grad():
        inputs = processor(images=image, text=texts, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        image_embed = outputs.image_embeds[0].unsqueeze(0)
        text_embeds = outputs.text_embeds
        sims = torch.nn.functional.cosine_similarity(image_embed, text_embeds)
        sorted_idxs = torch.argsort(sims, descending=True)
        rank = (sorted_idxs == 0).nonzero(as_tuple=True)[0].item() + 1
        return 1.0 / float(rank)


def robustness_score(model, processor, image: Image.Image, prompt: str, device: str, n_augs: int = 5):
    base = compute_clip_score(model, processor, image, prompt, device)
    if base == 0:
        return 0.0
    aug_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)], p=0.5),
        transforms.RandomAdjustSharpness(0.5, p=0.5),
        transforms.RandomRotation(5),
    ])
    scores = []
    for i in range(n_augs):
        aug = aug_transforms(image)
        s = compute_clip_score(model, processor, aug, prompt, device)
        scores.append(s)
    mean_aug = float(np.mean(scores)) if scores else 0.0
    return mean_aug / base


def analyze_components(model, processor, image: Image.Image, prompt: str, device: str):
    return {
        'objects': compute_clip_score(model, processor, image, f"objects: {prompt}", device),
        'style': compute_clip_score(model, processor, image, f"artistic style of {prompt}", device),
        'composition': compute_clip_score(model, processor, image, f"composition of {prompt}", device),
        'lighting': compute_clip_score(model, processor, image, f"lighting in {prompt}", device),
        'detail': compute_clip_score(model, processor, image, f"details in {prompt}", device),
    }


def aggregate_score(metrics: dict):
    # metrics expected to contain component keys plus retrieval_rank and robustness
    component_scores = [v for k, v in metrics.items() if k not in ("retrieval_rank", "robustness", "global_clip")]
    component_mean = float(np.mean(component_scores)) if component_scores else 0.0
    w_comp = 0.7
    w_retr = 0.2
    w_rob = 0.1
    retrieval = metrics.get('retrieval_rank', 0.0)
    robustness = metrics.get('robustness', 0.0)
    return w_comp * component_mean + w_retr * retrieval + w_rob * robustness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negatives-file', type=str, default=None)
    parser.add_argument('--output', type=str, default='metrics_output.json')
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_name = 'openai/clip-vit-large-patch14'
    print('Loading CLIP model (this may download weights if not cached)...')
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path).convert('RGB')

    metrics = analyze_components(model, processor, img, args.prompt, device)
    negatives = [
        "a photo of a cat",
        "a painting of a landscape",
        "an illustration of a food item",
        "a sketch of an object",
    ]
    if args.negatives_file:
        nf = Path(args.negatives_file)
        if nf.exists():
            negatives = [l.strip() for l in nf.read_text().splitlines() if l.strip()]

    metrics['retrieval_rank'] = retrieval_rank(model, processor, img, args.prompt, negatives, device)
    metrics['robustness'] = robustness_score(model, processor, img, args.prompt, device)
    metrics['global_clip'] = compute_clip_score(model, processor, img, args.prompt, device)

    metrics['aggregate_coherence'] = aggregate_score(metrics)

    out = {
        'image': str(img_path),
        'prompt': args.prompt,
        'metrics': metrics
    }

    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)

    print('Metrics written to', args.output)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
