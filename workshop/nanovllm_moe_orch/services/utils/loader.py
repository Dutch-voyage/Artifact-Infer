import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# HF MoE keys look like:  ....experts.{e}.{gate|up|down}_proj.{weight|bias}
# We collapse them into stacked tensors `....experts.w1` / `....experts.w2`
# and call the param's weight_loader(param, loaded_weight, expert_id, shard_id).
_MOE_EXPERT_RE = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$")
_MOE_SHARD_TO_TARGET = {
    "gate_proj": ("w1", "gate"),
    "up_proj":   ("w1", "up"),
    "down_proj": ("w2", None),
}


def _safe_get_parameter(model: nn.Module, param_name: str):
    """Returns the parameter or None if it doesn't exist on the model
    (e.g. when the model has fewer hidden layers than the checkpoint)."""
    try:
        return model.get_parameter(param_name)
    except AttributeError:
        return None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    n_loaded = n_skipped = 0
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 1) MoE stacked-expert path
                m = _MOE_EXPERT_RE.match(weight_name)
                if m is not None:
                    parent, expert_id_s, proj, suffix = m.groups()
                    target_attr, shard_id = _MOE_SHARD_TO_TARGET[proj]
                    param_name = f"{parent}.{target_attr}" + ("" if suffix == "weight" else f".{suffix}")
                    param = _safe_get_parameter(model, param_name)
                    if param is None:
                        n_skipped += 1
                        continue
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, f.get_tensor(weight_name), int(expert_id_s), shard_id)
                    n_loaded += 1
                    continue

                # 2) Existing packed-merge path (qkv_proj, gate_up_proj, ...)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = _safe_get_parameter(model, param_name)
                        if param is None:
                            n_skipped += 1
                            break
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        n_loaded += 1
                        break
                else:
                    # 3) Plain param
                    param = _safe_get_parameter(model, weight_name)
                    if param is None:
                        n_skipped += 1
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    n_loaded += 1
    print(f"[loader] loaded {n_loaded} tensors, skipped {n_skipped} (model has fewer layers than ckpt)")
