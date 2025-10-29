# save_load.py
import torch
import os

def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix from keys if present (DataParallel)."""
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def _partial_update_model(model, ckpt_sd):
    model_sd = model.state_dict()
    # intersect keys
    intersect = {k: v for k, v in ckpt_sd.items() if k in model_sd}
    if not intersect:
        return False, 0
    model_sd.update(intersect)
    model.load_state_dict(model_sd, strict=False)
    return True, len(intersect)

def _map_and_load(model, ckpt_sd):
    """
    Try some heuristic mappings when keys don't match:
      - q1_net.* -> net.*  (use q1 weights for single-net models)
      - net.* -> q1_net.* and net.* -> q2_net.* (duplicate single-net into twin critic)
    Returns (loaded: bool, desc: str)
    """
    model_sd = model.state_dict()
    keys_ckpt = set(ckpt_sd.keys())
    keys_model = set(model_sd.keys())

    # case: ckpt has q1_net.* and model expects net.*
    if any(k.startswith("q1_net.") for k in keys_ckpt) and any(k.startswith("net.") for k in keys_model):
        mapped = {}
        for k, v in ckpt_sd.items():
            if k.startswith("q1_net."):
                mapped_key = k.replace("q1_net.", "net.")
                if mapped_key in model_sd:
                    mapped[mapped_key] = v
        if mapped:
            model_sd.update(mapped)
            model.load_state_dict(model_sd, strict=False)
            return True, f"Mapped q1_net.* -> net.* and loaded {len(mapped)} keys."

    # case: ckpt has net.* and model expects q1_net.* and q2_net.* (single->twin)
    if any(k.startswith("net.") for k in keys_ckpt) and (any(k.startswith("q1_net.") for k in keys_model) or any(k.startswith("q2_net.") for k in keys_model)):
        mapped = {}
        for k, v in ckpt_sd.items():
            if k.startswith("net."):
                k_q1 = k.replace("net.", "q1_net.")
                k_q2 = k.replace("net.", "q2_net.")
                if k_q1 in model_sd:
                    mapped[k_q1] = v
                if k_q2 in model_sd:
                    mapped[k_q2] = v
        if mapped:
            model_sd.update(mapped)
            model.load_state_dict(model_sd, strict=False)
            return True, f"Duplicated net.* -> q1_net.* & q2_net.* and loaded {len(mapped)} keys."

    return False, "No heuristic mapping found."

def _safe_load_into_model(model, path, device="cpu", model_name="model"):
    if not os.path.exists(path):
        print(f"[save_load] {model_name}: file not found at {path}, skipping.")
        return

    ckpt = torch.load(path, map_location=device)
    # If the checkpoint is a dict with other metadata, try to find state_dict
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
        ckpt_sd = ckpt[key]
    else:
        ckpt_sd = ckpt

    ckpt_sd = _strip_module_prefix(ckpt_sd)
    model_sd = model.state_dict()

    # 1) Exact match
    if set(ckpt_sd.keys()) == set(model_sd.keys()):
        model.load_state_dict(ckpt_sd, strict=False)
        print(f"[save_load] {model_name}: exact state_dict keys match — loaded successfully from {path}.")
        return

    # 2) Partial intersection (safe)
    loaded, count = _partial_update_model(model, ckpt_sd)
    if loaded:
        print(f"[save_load] {model_name}: partially loaded {count}/{len(model_sd)} keys from {path}.")
        return

    # 3) Heuristic mapping attempts
    mapped, desc = _map_and_load(model, ckpt_sd)
    if mapped:
        print(f"[save_load] {model_name}: {desc}")
        return

    # 4) Nothing sensible
    print(f"[save_load] WARNING: could not load compatible keys for {model_name} from {path}.")
    print(f"[save_load] ckpt keys sample: {list(ckpt_sd.keys())[:10]}")
    print(f"[save_load] model keys sample: {list(model_sd.keys())[:10]}")

def save_all(actor, critic, reward_model, buffer, cfg):
    # ensure dir exists
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    torch.save(actor.state_dict(), cfg.ACTOR_PATH)
    torch.save(critic.state_dict(), cfg.CRITIC_PATH)
    if reward_model is not None:
        torch.save(reward_model.state_dict(), cfg.REWARD_MODEL_PATH)
    try:
        buffer.save(cfg.BUFFER_PATH)
    except Exception as e:
        print(f"[save_load] Warning: could not save buffer: {e}")
    print("[save_load] Saved actor, critic, reward model and buffer to", cfg.SAVE_DIR)

def load_all(actor, critic, reward_model, buffer, cfg, device="cpu"):
    # Be defensive and informative
    _safe_load_into_model(actor, cfg.ACTOR_PATH, device=device, model_name="Actor")
    _safe_load_into_model(critic, cfg.CRITIC_PATH, device=device, model_name="Critic/TwinCritic")
    if reward_model is not None:
        _safe_load_into_model(reward_model, cfg.REWARD_MODEL_PATH, device=device, model_name="RewardModel/Ensemble")
    if os.path.exists(cfg.BUFFER_PATH):
        try:
            buffer.load(cfg.BUFFER_PATH)
            print("[save_load] Buffer loaded.")
        except Exception as e:
            print("[save_load] Warning: failed to load buffer:", e)
    print("[save_load] load_all finished (loaded available compatible artifacts).")
