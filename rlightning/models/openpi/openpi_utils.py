import os

import openpi.shared.download as download
import openpi.transforms as transforms
import safetensors
import torch
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

from .openpi_action_model import OpenPi0Config, OpenPi0ForRLActionPrediction


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)


def get_openpi_model(cfg, **kwargs):
    device = kwargs.get("device", torch.device("cuda"))
    # config
    if getattr(cfg.openpi, "pi05", False):
        actor_train_config = _config.get_config("pi05_libero")
    else:
        actor_train_config = _config.get_config("pi0_libero")
    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_config_kwargs = cfg.openpi.to_dict()
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            actor_model_config.__dict__[key] = val
    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(actor_model_config)
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()
    safetensors.torch.load_model(model, weight_path, strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(actor_train_config.assets_dirs, actor_model_config)
    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )
    model = model.to(device=device)

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=[
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",
                    "out_proj",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            if cfg.model_name == "openpi":
                module_to_lora = model.paligemma_with_expert.paligemma
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.paligemma_with_expert.paligemma = module_to_lora
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
        model_dict = torch.load(cfg.ckpt_path)
        model.load_state_dict(model_dict)
    return model
