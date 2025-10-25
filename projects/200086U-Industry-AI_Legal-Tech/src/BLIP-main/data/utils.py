# import re
# import json
# import os

# import torch
# import torch.distributed as dist

# import utils

# def pre_caption(caption,max_words=50):
#     caption = re.sub(
#         r"([.!\"()*#:;~])",       
#         ' ',
#         caption.lower(),
#     )
#     caption = re.sub(
#         r"\s{2,}",
#         ' ',
#         caption,
#     )
#     caption = caption.rstrip('\n') 
#     caption = caption.strip(' ')

#     #truncate caption
#     caption_words = caption.split(' ')
#     if len(caption_words)>max_words:
#         caption = ' '.join(caption_words[:max_words])
            
#     return caption

# def pre_question(question,max_ques_words=50):
#     question = re.sub(
#         r"([.!\"()*#:;~])",
#         '',
#         question.lower(),
#     ) 
#     question = question.rstrip(' ')
    
#     #truncate question
#     question_words = question.split(' ')
#     if len(question_words)>max_ques_words:
#         question = ' '.join(question_words[:max_ques_words])
            
#     return question


# def save_result(result, result_dir, filename, remove_duplicate=''):
#     result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
#     final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
#     json.dump(result,open(result_file,'w'))

#     dist.barrier()

#     if utils.is_main_process():   
#         # combine results from all processes
#         result = []

#         for rank in range(utils.get_world_size()):
#             result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
#             res = json.load(open(result_file,'r'))
#             result += res

#         if remove_duplicate:
#             result_new = []
#             id_list = []    
#             for res in result:
#                 if res[remove_duplicate] not in id_list:
#                     id_list.append(res[remove_duplicate])
#                     result_new.append(res)
#             result = result_new             
                
#         json.dump(result,open(final_result_file,'w'))            
#         print('result file saved to %s'%final_result_file)

#     return final_result_file



# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
# from torchvision.datasets.utils import download_url

# def coco_caption_eval(coco_gt_root, results_file, split):
#     urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
#             'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
#     filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
#     download_url(urls[split],coco_gt_root)
#     annotation_file = os.path.join(coco_gt_root,filenames[split])
    
#     # create coco object and coco_result object
#     coco = COCO(annotation_file)
#     coco_result = coco.loadRes(results_file)

#     # create coco_eval object by taking coco and coco_result
#     coco_eval = COCOEvalCap(coco, coco_result)

#     # evaluate on a subset of images by setting
#     # coco_eval.params['image_id'] = coco_result.getImgIds()
#     # please remove this line when evaluating the full validation set
#     # coco_eval.params['image_id'] = coco_result.getImgIds()

#     # evaluate results
#     # SPICE will take a few minutes the first time, but speeds up due to caching
#     coco_eval.evaluate()

#     # print output evaluation scores
#     for metric, score in coco_eval.eval.items():
#         print(f'{metric}: {score:.3f}')
    
#     return coco_eval

import os
import re
import json
import torch
import torch.distributed as dist
from torchvision.datasets.utils import download_url

# ==========================================================
# Environment setup (Java discovery for pycocoevalcap)
# ==========================================================
import shutil
java_path = shutil.which("java")
if java_path:
    java_home = os.path.dirname(os.path.dirname(java_path))
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = java_home + "/bin:" + os.environ["PATH"]
    print(f"✅ Java detected at {java_path}")
else:
    print("⚠️ Java not found in PATH — fallback tokenizer will be used if needed.")

# ==========================================================
# Text preprocessing
# ==========================================================

def pre_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.strip()
    words = caption.split(' ')
    if len(words) > max_words:
        caption = ' '.join(words[:max_words])
    return caption


def pre_question(question, max_ques_words=50):
    question = re.sub(r"([.!\"()*#:;~])", '', question.lower())
    question = question.strip()
    words = question.split(' ')
    if len(words) > max_ques_words:
        question = ' '.join(words[:max_ques_words])
    return question


# ==========================================================
# Result saving
# ==========================================================

def save_result(result, result_dir, filename, remove_duplicate=''):
    os.makedirs(result_dir, exist_ok=True)
    rank = get_rank()
    world_size = get_world_size()

    result_file = os.path.join(result_dir, f'{filename}_rank{rank}.json')
    final_result_file = os.path.join(result_dir, f'{filename}.json')

    json.dump(result, open(result_file, 'w'))
    dist.barrier()

    if is_main_process():
        combined = []
        for r in range(world_size):
            part = os.path.join(result_dir, f'{filename}_rank{r}.json')
            if os.path.exists(part):
                combined.extend(json.load(open(part, 'r')))

        if remove_duplicate:
            seen = set()
            deduped = []
            for res in combined:
                key = res.get(remove_duplicate)
                if key not in seen:
                    seen.add(key)
                    deduped.append(res)
            combined = deduped

        json.dump(combined, open(final_result_file, 'w'))
        print(f"✅ result file saved to {final_result_file}")

    return final_result_file


# ==========================================================
# COCO Caption Evaluation (patched version)
# ==========================================================

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# Simple fallback tokenizer (no Java)
class SimpleTokenizer:
    def tokenize(self, captions_for_image):
        out = {}
        for img_id, caps in captions_for_image.items():
            processed = []
            for d in caps:
                s = d['caption'].lower().strip()
                s = re.sub(r"([?!,.;:])", r" \1 ", s)
                s = re.sub(r"\s+", " ", s).strip()
                processed.append({'caption': s})
            out[img_id] = processed
        return out


def _ensure_coco_gt_full(annotation_file: str) -> str:
    """
    Ensure the ground truth JSON has 'info', 'licenses', 'images', 'annotations'.
    If not, create a fixed copy with proper structure.
    """
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Already proper COCO format
    if isinstance(data, dict) and all(k in data for k in ["info", "images", "annotations"]):
        return annotation_file

    # Otherwise fix
    anns = data.get("annotations", data)
    img_ids = sorted({a["image_id"] for a in anns})
    images = [{"id": i, "file_name": f"COCO_train2014_{i:012d}.jpg"} for i in img_ids]

    fixed = {
        "info": {
            "description": "COCO Caption Karpathy Split (Auto-Fixed)",
            "version": "1.0",
            "year": 2014,
            "contributor": "Auto-fix",
        },
        "licenses": [
            {"id": 1, "name": "Unknown", "url": "https://cocodataset.org"}
        ],
        "images": images,
        "annotations": anns,
    }

    fixed_path = os.path.splitext(annotation_file)[0] + "_fixed.json"
    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2)
    print(f"🧩 Fixed GT written to {fixed_path}")
    return fixed_path


def coco_caption_eval(coco_gt_root, results_file, split):
    """
    Evaluate COCO-style caption results using pycocoevalcap.
    Fixes missing 'info' key and ensures tokenizer works.
    """
    urls = {
        'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
        'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'
    }
    filenames = {
        'val': 'coco_karpathy_val_gt.json',
        'test': 'coco_karpathy_test_gt.json'
    }

    os.makedirs(coco_gt_root, exist_ok=True)
    annotation_file = os.path.join(coco_gt_root, filenames[split])
    if not os.path.exists(annotation_file):
        download_url(urls[split], coco_gt_root)

    # Ensure COCO-compliant file
    annotation_file = _ensure_coco_gt_full(annotation_file)

    # Try Java tokenizer first
    from pycocoevalcap import tokenizer as tok_module
    try:
        if not shutil.which("java"):
            raise EnvironmentError("Java missing")
        print("🟢 Using PTBTokenizer (Java-based)")
    except Exception:
        print("⚠️ Using SimpleTokenizer (Java not found or failed)")
        tok_module.ptbtokenizer.PTBTokenizer = SimpleTokenizer

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    print(f"\n===== Caption Evaluation Results ({split}) =====")
    for metric, score in coco_eval.eval.items():
        print(f'{metric:10s}: {score:.3f}')

    return coco_eval


# ==========================================================
# Distributed helpers
# ==========================================================

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def is_main_process():
    return get_rank() == 0