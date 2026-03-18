# Derived from code copied from RLinf/RLinf (Apache-2.0):
# https://github.com/RLinf/RLinf
# Original path: rlightning/models/openvla/openvla_model.py
# Modified in this repository.
# See THIRD_PARTY_NOTICES.md for details.

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
    LogitsProcessorList,
    TopKLogitsWarper,
)

from rlightning.models.model_utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)
from rlightning.utils.utils import torch_dtype_from_precision

from .openvla_utils import check_model_logic_mismatch, update_auto_map
from .value_head import ValueHead, VLALogitsProcessor


def convert_to_regular_types(value):
    if isinstance(value, list) or isinstance(value, tuple):
        return [convert_to_regular_types(v) for v in value]
    if hasattr(value, "to_container"):
        return value.to_container()
    return value


def register_openvla_model(vla_model_name, model_path):
    if vla_model_name == "openvla-oft":
        from third_party.openvla_oft.configuration_prismatic import OpenVLAConfig
        from third_party.openvla_oft.modeling_prismatic import (
            OpenVLAForActionPrediction,
        )
        from third_party.openvla_oft.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    elif vla_model_name == "openvla":
        from third_party.openvla.configuration_prismatic import OpenVLAConfig
        from third_party.openvla.modeling_prismatic import (
            OpenVLAForBatchActionPrediction,
        )
        from third_party.openvla.processing_prismatic import (
            PrismaticImageProcessorForBatch,
            PrismaticProcessorForBatch,
        )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessorForBatch)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessorForBatch)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForBatchActionPrediction)

        update_auto_map(model_path)
        check_model_logic_mismatch(model_path)


class OpenVLAModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.device = kwargs.get("device", torch.device("cuda"))

        self.unnorm_key = self.config.unnorm_key
        # self.policy_setup = self.config.policy_setup
        # self.adv_type = cfg.algorithm.adv_type
        self.max_prompt_length = self.config.max_prompt_length
        self.action_dim = self.config.action_dim
        self.num_action_chunks = self.config.num_action_chunks

        # init model
        self._init_model()
        # init logits processor
        self._init_logits_processor()

    def _init_logits_processor(self):
        self.logits_processors = LogitsProcessorList()
        self.logits_processors.append(VLALogitsProcessor(self.n_action_bins))

    def _init_model(self):
        # register_openvla_model(self.config.model_name)

        torch_dtype = torch_dtype_from_precision(self.config.precision)
        trust_remote_code = self.config.get("trust_remote_code", True)
        low_cpu_mem_usage = self.config.get("low_cpu_mem_usage", True)

        model_path = self.config.model_path
        from third_party.openvla.configuration_prismatic import OpenVLAConfig
        from third_party.openvla.modeling_prismatic import (
            OpenVLAForBatchActionPrediction,
        )
        from third_party.openvla.processing_prismatic import (
            PrismaticImageProcessorForBatch,
            PrismaticProcessorForBatch,
        )

        model_config = OpenVLAConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(model_config, "norm_stats", norm_stats)
        self.setup_config(model_config)

        self.vla = OpenVLAForBatchActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation=self.config.attn_implementation,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
        )
        if self.config.add_value_head:
            self.value_head = ValueHead(
                input_dim=self.config.hidden_size,
                hidden_sizes=(512, 128),
                output_dim=1,
                activation="gelu",
                bias_last=False,
            )
        self.value_head.to(device=self.device, dtype=torch_dtype)
        self.vla.to(device=self.device, dtype=torch_dtype)

        image_processor = PrismaticImageProcessorForBatch.from_pretrained(
            self.config.tokenizer_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path, trust_remote_code=True, padding_side="left"
        )
        self.input_processor = PrismaticProcessorForBatch.from_pretrained(
            self.config.tokenizer_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )
        self._init_lora()

        if hasattr(self.config, "ckpt_path") and self.config.ckpt_path is not None:
            model_dict = torch.load(self.config.ckpt_path)
            self.vla.load_state_dict(model_dict)

        # log_gpu_memory_usage("After init from HF AutoModel", logger=logger)

    def _init_lora(self):
        if self.config.is_lora:
            from peft import LoraConfig, PeftModel, get_peft_model

            if not hasattr(self.config, "lora_path") or self.config.lora_path is None:
                lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
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
                self.vla = get_peft_model(self.vla, lora_config)
            else:
                self.vla = PeftModel.from_pretrained(self.vla, self.config.lora_path, is_trainable=True)

    def _preprocess_obs(
        self,
        raw_obs,
    ):
        task_descriptions = [
            f"In: What action should the robot take to {t.lower()}?\nOut: " for t in raw_obs["task_descriptions"]
        ]
        image_tensor = raw_obs["images"]
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(1)
        assert image_tensor.ndim == 5

        max_length = self.max_prompt_length
        processed_obs = self.input_processor(
            text=task_descriptions,
            images=image_tensor,
            padding="max_length",
            max_length=max_length,
        )

        return processed_obs

    def _postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = action.cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = np.asarray([self.bin_centers[da] for da in discretized_actions])  # [B, dim]

        # Unnormalize actions
        mask = self.action_mask.reshape(1, -1).repeat(action.shape[0], axis=0)  # [B, dim]
        action_high, action_low = self.max_action, self.min_action
        actions_np = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions_np

    @torch.inference_mode()
    def get_action(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        env_obs=None,
        calulate_logprobs=True,
        calulate_values=True,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")

        if env_obs is not None:
            processed_obs = self._preprocess_obs(env_obs)
            device = next(self.parameters()).device
            precision = next(self.parameters()).dtype
            input_ids = processed_obs["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = processed_obs["attention_mask"].to(device=device, dtype=torch.bool)
            pixel_values = processed_obs["pixel_values"].to(device=device, dtype=precision)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        assert input_ids is not None and attention_mask is not None and pixel_values is not None

        # assert first token is 1
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        generated_results: GenerateDecoderOnlyOutput = self.vla.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_scores=True,
            output_logits=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=do_sample,
            logits_processor=self.logits_processors,
            **kwargs,
        )
        action_tokens = generated_results.sequences
        action_tokens = action_tokens[:, -self.action_dim :]

        # check valid action
        assert torch.all(action_tokens >= 32000 - 256) and torch.all(action_tokens < 32000)

        # postprocess action
        actions_np = self._postprocess_action(action_tokens)

        # logits
        token_logits = generated_results.scores  # ([B, vocab-size], ...), after logits processor and warper results
        token_logits_tensor = torch.stack(token_logits, dim=1)  # [B, action-dim, vocab-size]

        last_hidden_states = torch.stack(
            [token_hidden_states[-1][:, -1] for token_hidden_states in generated_results.hidden_states],
            dim=1,
        )  # [B, hidden_states] -> [B, action-dim, hidden_states]

        action_logits = token_logits_tensor.permute(0, 2, 1)  # [B, vocab-size, action-dim]
        action_logits[:, : self.vocab_size - self.n_action_bins] = -torch.inf
        action_logits[:, self.vocab_size :] = -torch.inf

        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=action_tokens)

        if hasattr(self, "value_head") and calulate_values:
            hidden_features = last_hidden_states[
                :, -self.action_dim * self.num_action_chunks
            ]  # [batch_size, hidden_dim]

            chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions_np = actions_np.reshape(-1, self.num_action_chunks, self.action_dim)
        chunk_action_tokens = action_tokens.reshape(-1, self.num_action_chunks, self.action_dim)

        forward_inputs["action_tokens"] = chunk_action_tokens

        results = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values.squeeze(-1),
            "forward_inputs": forward_inputs,
        }
        return chunk_actions_np, results

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        if forward_inputs is not None:
            forward_inputs = self._preprocess_for_train(forward_inputs)
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]

            action_tokens = forward_inputs["action_tokens"]

        if compute_values:
            output_hidden_states = True

        outputs = self.vla.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        if compute_logprobs:
            logits = outputs.logits[
                :, -self.action_dim * self.num_action_chunks - 1 : -1
            ]  # [B, action-dim, vocab-size]

            processed_logits_tensor = logits / kwargs["temperature"]
            top_k = min(kwargs["top_k"], processed_logits_tensor.size(-1))  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            action_logits = processed_logits_tensor.permute(0, 2, 1)  # [B, vocab-size, action-dim]
            action_logits[:, : self.vocab_size - self.n_action_bins] = -torch.inf
            action_logits[:, self.vocab_size :] = -torch.inf

            logprobs = compute_logprobs_from_logits(logits=action_logits, target=action_tokens)

            entropy = None
            if compute_entropy:
                entropy = compute_entropy_from_logits(logits=action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[
                :, -self.action_dim * self.num_action_chunks - 1
            ]  # [batch_size, hidden_dim]
            values = self.value_head(hidden_features)
        else:
            values = None

        result = {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

        return result

    def _check_unnorm_key(self, norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def _get_action_stats(self) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def _preprocess_for_train(self, data):
        input_ids = data["input_ids"]
        action_tokens = data["action_tokens"]
        attention_mask = data["attention_mask"]

        action_tokens = action_tokens.reshape(action_tokens.shape[0], self.action_dim)

        data["input_ids"] = torch.cat([input_ids, action_tokens], dim=-1)  # [B, seq-len+action-dim]
        data["attention_mask"] = torch.cat(
            [attention_mask, torch.ones_like(action_tokens).to(attention_mask.dtype)],
            dim=-1,
        )
        data["action_tokens"] = action_tokens
        return data

    def setup_config(self, model_config):
        self.vocab_size = model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        self.n_action_bins = model_config.n_action_bins
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.norm_stats = model_config.norm_stats
        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_mask = np.array(action_norm_stats["mask"])
        self.action_scale = 1.0

    def save(self, save_dir: str):
        path = os.path.join(save_dir)
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.processor.save_pretrained(path)
        self.vla.save_pretrained(path)
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.pt"))

    def load(self, load_dir: str):
        self.vla.from_pretrained(load_dir)
        self.value_head.load_state_dict(torch.load(os.path.join(load_dir, "value_head.pt")))
