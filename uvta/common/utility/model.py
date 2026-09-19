import os

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass


def load_hydra_config(cfg_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    cfg = OmegaConf.create(cfg)
    return cfg


def load_config(model_path):
    cfg = omegaconf.OmegaConf.load(os.path.join(model_path, ".hydra/config.yaml"))
    cfg = OmegaConf.create(cfg)
    return cfg


def load_model(model_path, ckpt):
    cfg = load_config(model_path)
    model = hydra.utils.instantiate(cfg.model)
    loadpath = os.path.join(model_path, f"epoch={ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location="cuda:0")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda()
    model.eval()
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def enable_gradient(model):
    for param in model.parameters():
        param.requires_grad = True


def infer_unet_dims_from_state_dict(state_dict, diffusion_step_embed_dim=256):
    """Infer ``(global_cond_dim, action_dim)`` from a ConditionalUnet1D ckpt.

    Training computes ``global_cond_dim`` from the real input shapes via a
    one-batch dry-run (vision_emb * obs_horizon + proprio + fsr) and only
    overrides the *in-memory* cfg; the value saved to ``.hydra/config.yaml``
    stays at the yaml fallback (e.g. 384).  At deploy time we therefore must
    NOT trust the yaml -- instead we read the true dims back from the weights:

    * any ``*.cond_encoder.1.weight`` has shape ``[out*2, cond_dim]`` where
      ``cond_dim = diffusion_step_embed_dim + global_cond_dim``.
    * ``final_conv.1.weight`` has shape ``[action_dim, start_dim, 1]``.

    Returns ``(global_cond_dim, action_dim)`` with either entry ``None`` when
    the corresponding key is absent (so callers can fall back to the cfg).
    """
    prefix = "diffusion_policy_head."
    global_cond_dim = None
    action_dim = None
    for key, tensor in state_dict.items():
        if key.endswith("cond_encoder.1.weight") and global_cond_dim is None:
            cond_dim = int(tensor.shape[1])
            global_cond_dim = cond_dim - diffusion_step_embed_dim
        if key == f"{prefix}final_conv.1.weight":
            action_dim = int(tensor.shape[0])
    return global_cond_dim, action_dim


def load_diffusion_model(
    model_path, ckpt, load_pretrain_weight=True, use_ema=False, **kwargs
):
    model_cfg = load_config(model_path)
    if use_ema:
        print("Loading ema model checkpoints!")
        ckpt_path = os.path.join(model_path, "checkpoints", f"ema_epoch_{ckpt}.ckpt")
    else:
        ckpt_path = os.path.join(model_path, "checkpoints", f"epoch_{ckpt}.ckpt")

    # ``global_cond_dim`` / ``action_dim`` in the saved yaml are only yaml
    # fallbacks -- the trainer overrides them at runtime from the real input
    # dims but never writes them back to ``.hydra/config.yaml``.  Mirror that
    # "build the model from the actual dims" logic here by recovering the true
    # dims from the checkpoint weights before instantiating the model.
    state_dict = None
    if load_pretrain_weight:
        state_dict = torch.load(ckpt_path)
        global_cond_dim, action_dim = infer_unet_dims_from_state_dict(state_dict)
        head_cfg = model_cfg.model.diffusion_policy_head
        if global_cond_dim is not None and global_cond_dim != head_cfg.get(
            "global_cond_dim"
        ):
            print(
                f"[load_diffusion_model] overriding global_cond_dim "
                f"{head_cfg.get('global_cond_dim')} -> {global_cond_dim} "
                f"(inferred from checkpoint weights)"
            )
            OmegaConf.update(
                model_cfg,
                "model.diffusion_policy_head.global_cond_dim",
                int(global_cond_dim),
                merge=True,
            )
        if action_dim is not None and action_dim != head_cfg.get("input_dim"):
            print(
                f"[load_diffusion_model] overriding action input_dim "
                f"{head_cfg.get('input_dim')} -> {action_dim} "
                f"(inferred from checkpoint weights)"
            )
            OmegaConf.update(
                model_cfg,
                "model.diffusion_policy_head.input_dim",
                int(action_dim),
                merge=True,
            )

    model = hydra.utils.instantiate(model_cfg.model)
    if load_pretrain_weight:
        model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    noise_scheduler = hydra.utils.instantiate(model_cfg.noise_scheduler)

    return model, noise_scheduler
