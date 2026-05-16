#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel, LlamaTokenizerFast, \
    PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy

from llava.train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from ..prompt_llama3 import Llama3ChatPromptBuilder

IGNORE_INDEX = -100


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


## FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)
        # self.init_vlm2(config=config, *args, **kwargs)
        # self.history_mamba_hidden_states = None
        # self.norm_stats = norm_stats
        # self.action_tokenizer = action_tokenizer

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_tower", "llm", "mm_projector"]
        self.trainable_module_keys = []


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        history_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        same_seq: bool = False,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.get_llm()(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.get_llm()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run Visual Feature Extraction
        # with torch.set_grad_enabled(self.vision_backbone_requires_grad):
        with torch.set_grad_enabled(False):
            # patch_features = self.get_vision_tower()(images[multimodal_indices])

            multimodal_images = pixel_values[multimodal_indices]
            # 如果有多余的维度，移除它
            # if multimodal_images.dim() == 5 and multimodal_images.size(1) == 1:
            #     multimodal_images = multimodal_images.squeeze(1)
            patch_features = self.get_vision_tower()(multimodal_images)
            # if history_values.dim() == 5 and history_values.size(0) == 1:
            #     history_values = history_values.squeeze(0)
            # history_features = []
            # for history_value in history_values:
            #     history_value = history_value.unsqueeze(0)
            #     history_feature = self.get_vision_tower()(history_value)
            #     history_features.append(history_feature)
            # history_features = torch.cat(history_features, dim=1)


            # Mamba setting
            if history_values is not None:
                history_features = self.get_history_mamba()(history_values)
            #     # self.history_mamba_hidden_states = history_features
            else:
                history_features = self.get_history_mamba()(pixel_values)

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS

        # projected_patch_embedding = []
        # for idx, patch_feature in enumerate(patch_features):
        #     projected_patch_embedding.append(self.get_mm_projector()(patch_feature))

        # 确保数据类型和设备一致性
        patch_features = patch_features.to(dtype=torch.bfloat16)
        mm_projector = self.get_mm_projector()
        mm_projector = mm_projector.to(dtype=torch.bfloat16)  # 确保投影器参数也是float32
        projected_patch_embeddings = mm_projector(patch_features)
        # projected_history_embeddings = []
        # for history_feature in history_features:
        #     projected_history_embedding = mm_projector(history_feature)
        #     projected_history_embeddings.append(projected_history_embedding)
        # projected_history_embeddings = torch.cat(projected_history_embeddings, dim=1)

        # Mamba setting
        mb_projector = self.get_mb_projector()
        projected_history_embeddings = mb_projector(history_features)


        # projected_patch_embeddings = torch.cat(projected_patch_embedding[::-1], dim=1)
        projected_patch_attention_mask = None

        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1] + projected_history_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # # 临时调试代码，查看可用的嵌入方法
        # llm_model = self.get_llm()
        # print("Available embedding methods:")
        # print([attr for attr in dir(llm_model) if 'embed' in attr.lower()])
        # print([attr for attr in dir(llm_model.model) if 'embed' in attr.lower()])

        # 通常Llama模型使用：
        # 方法1: llm_model.model.embed_tokens(input_ids)
        # 方法2: llm_model.get_input_embeddings()(input_ids)
        # input_embeddings = self.get_llm().embed_input_ids(input_ids)
        input_embeddings = self.get_llm().model.embed_tokens(input_ids)


        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                projected_history_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1] + projected_history_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [
                    labels[multimodal_indices, :1],
                    projected_patch_labels,
                    labels[multimodal_indices, 1:]
                ],
                dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.get_llm()(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
                (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.llm._reorder_cache(*args, **kwargs)


    @torch.inference_mode()
    def predict_action(
            # self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
            self,
            image: Image,
            instruction: str,
            norm_stats,
            action_tokenizer=None,
            unnorm_key: Optional[str] = None,
            same_seq: bool = False,
            **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        # prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
        prompt_builder = Llama3ChatPromptBuilder("navila")
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        image_processor = self.get_vision_tower().image_processor
        tokenizer = self.tokenizer
        if not same_seq:
            self.get_history_mamba().reset_state()

        # image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer
        #
        # # Build VLA Prompt
        # prompt_builder = self.get_prompt_builder()
        # prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        # prompt_text = prompt_builder.get_prompt()
        #
        # # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, (LlamaTokenizerFast, PreTrainedTokenizerFast)):
            # 检查是否是 Llama3
            if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size == 128000:
                # if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
                #     tokenizer.bos_token = "<|begin_of_text|>"
                # if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None:
                #     tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
                #
                #     # 同样确保 EOS token 设置正确
                # if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
                #     tokenizer.eos_token = "<|eot_id|>"
                # if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
                #     tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                # Llama3 通常不需要额外添加特殊 token，因为对话格式已经包含了
                # 如果需要添加，使用 Llama3 的特殊 token
                # assistant_start_token_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
                # 或者直接使用 token ID
                # input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([assistant_start_token_id]).long(), dim=0).to(self.device)), dim=1)
                pass  # Llama3 通常不需要额外处理
            else:
                # 原来的 Llama2 处理逻辑
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")


        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        # autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda"):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,  # Shape: [1, seq]
                # pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, ...]
                pixel_values=pixel_values,
                max_new_tokens=self.get_action_dim(norm_stats, unnorm_key),
                # do_sample=True,
                do_sample=False,
                # temperature=0.7,
                bos_token_id=tokenizer.bos_token_id,
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(norm_stats, unnorm_key):]
        print(f"Predicted action: {predicted_action_token_ids}")
        normalized_actions = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(norm_stats, unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions


    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
                unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, norm_stats, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, norm_stats, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]



class LlavaLlamaModel2(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm2(config=config, *args, **kwargs)
        # self.history_mamba_hidden_states = None
        # self.norm_stats = norm_stats
        # self.action_tokenizer = action_tokenizer

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_tower", "llm", "mm_projector"]
        self.trainable_module_keys = []


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        history_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        same_seq: bool = False,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.get_llm()(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.get_llm()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run Visual Feature Extraction
        # with torch.set_grad_enabled(self.vision_backbone_requires_grad):
        with torch.set_grad_enabled(False):
            # patch_features = self.get_vision_tower()(images[multimodal_indices])

            multimodal_images = pixel_values[multimodal_indices]
            # 如果有多余的维度，移除它
            # if multimodal_images.dim() == 5 and multimodal_images.size(1) == 1:
            #     multimodal_images = multimodal_images.squeeze(1)
            patch_features = self.get_vision_tower()(multimodal_images)

            # if history_values is not None:
            #     history_features = self.get_history_mamba()(history_values)
                # self.history_mamba_hidden_states = history_features
            # else:
            #     history_features = self.get_history_mamba()(pixel_values)

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS

        # projected_patch_embedding = []
        # for idx, patch_feature in enumerate(patch_features):
        #     projected_patch_embedding.append(self.get_mm_projector()(patch_feature))

        # 确保数据类型和设备一致性
        patch_features = patch_features.to(dtype=torch.bfloat16)
        mm_projector = self.get_mm_projector()
        mm_projector = mm_projector.to(dtype=torch.bfloat16)  # 确保投影器参数也是float32
        projected_patch_embeddings = mm_projector(patch_features)


        # mb_projector = self.get_mb_projector()
        # projected_history_embeddings = mb_projector(history_features)


        # projected_patch_embeddings = torch.cat(projected_patch_embedding[::-1], dim=1)
        projected_patch_attention_mask = None

        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0],
                 # projected_patch_embeddings.shape[1] + projected_history_embeddings.shape[1]),
                 projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # # 临时调试代码，查看可用的嵌入方法
        # llm_model = self.get_llm()
        # print("Available embedding methods:")
        # print([attr for attr in dir(llm_model) if 'embed' in attr.lower()])
        # print([attr for attr in dir(llm_model.model) if 'embed' in attr.lower()])

        # 通常Llama模型使用：
        # 方法1: llm_model.model.embed_tokens(input_ids)
        # 方法2: llm_model.get_input_embeddings()(input_ids)
        # input_embeddings = self.get_llm().embed_input_ids(input_ids)
        input_embeddings = self.get_llm().model.embed_tokens(input_ids)


        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                # projected_history_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1]),
                 # projected_patch_embeddings.shape[1] + projected_history_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [
                    labels[multimodal_indices, :1],
                    projected_patch_labels,
                    labels[multimodal_indices, 1:]
                ],
                dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.get_llm()(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
                (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.llm._reorder_cache(*args, **kwargs)


    @torch.inference_mode()
    def predict_action(
            # self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
            self,
            image: Image,
            instruction: str,
            norm_stats,
            action_tokenizer=None,
            unnorm_key: Optional[str] = None,
            same_seq: bool = False,
            **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        # prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
        prompt_builder = Llama3ChatPromptBuilder("navila")
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        image_processor = self.get_vision_tower().image_processor
        tokenizer = self.tokenizer
        # if not same_seq:
        #     self.get_history_mamba().reset_state()

        # image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer
        #
        # # Build VLA Prompt
        # prompt_builder = self.get_prompt_builder()
        # prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        # prompt_text = prompt_builder.get_prompt()
        #
        # # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, (LlamaTokenizerFast, PreTrainedTokenizerFast)):
            # 检查是否是 Llama3
            if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size == 128000:
                # if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
                #     tokenizer.bos_token = "<|begin_of_text|>"
                # if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None:
                #     tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
                #
                #     # 同样确保 EOS token 设置正确
                # if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
                #     tokenizer.eos_token = "<|eot_id|>"
                # if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
                #     tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                # Llama3 通常不需要额外添加特殊 token，因为对话格式已经包含了
                # 如果需要添加，使用 Llama3 的特殊 token
                # assistant_start_token_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
                # 或者直接使用 token ID
                # input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([assistant_start_token_id]).long(), dim=0).to(self.device)), dim=1)
                pass  # Llama3 通常不需要额外处理
            else:
                # 原来的 Llama2 处理逻辑
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")


        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        # autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda"):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,  # Shape: [1, seq]
                # pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, ...]
                pixel_values=pixel_values,
                max_new_tokens=self.get_action_dim(norm_stats, unnorm_key),
                # do_sample=True,
                do_sample=False,
                # temperature=0.7,
                bos_token_id=tokenizer.bos_token_id,
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(norm_stats, unnorm_key):]
        print(f"Predicted action: {predicted_action_token_ids}")
        normalized_actions = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(norm_stats, unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions


    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
                unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, norm_stats, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, norm_stats, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]



class LlavaLlamaModel3(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        # self.init_vlm(config=config, *args, **kwargs)
        self.init_vlm2(config=config, *args, **kwargs)
        # self.history_mamba_hidden_states = None
        # self.norm_stats = norm_stats
        # self.action_tokenizer = action_tokenizer

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_tower", "llm", "mm_projector"]
        self.trainable_module_keys = []


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        history_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        same_seq: bool = False,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.get_llm()(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.get_llm()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run Visual Feature Extraction
        # with torch.set_grad_enabled(self.vision_backbone_requires_grad):
        with torch.set_grad_enabled(False):
            # patch_features = self.get_vision_tower()(images[multimodal_indices])

            multimodal_images = pixel_values[multimodal_indices]
            # 如果有多余的维度，移除它
            # if multimodal_images.dim() == 5 and multimodal_images.size(1) == 1:
            #     multimodal_images = multimodal_images.squeeze(1)
            patch_features = self.get_vision_tower()(multimodal_images)
            if history_values.dim() == 5 and history_values.size(0) == 1:
                history_values = history_values.squeeze(0)
            history_features = []
            for history_value in history_values:
                history_value = history_value.unsqueeze(0)
                history_feature = self.get_vision_tower()(history_value)
                history_features.append(history_feature)
            # history_features = torch.cat(history_features, dim=1)


            # Mamba setting
            # if history_values is not None:
            #     history_features = self.get_history_mamba()(history_values)
            #     # self.history_mamba_hidden_states = history_features
            # else:
            #     history_features = self.get_history_mamba()(pixel_values)

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS

        # projected_patch_embedding = []
        # for idx, patch_feature in enumerate(patch_features):
        #     projected_patch_embedding.append(self.get_mm_projector()(patch_feature))

        # 确保数据类型和设备一致性
        patch_features = patch_features.to(dtype=torch.bfloat16)
        mm_projector = self.get_mm_projector()
        mm_projector = mm_projector.to(dtype=torch.bfloat16)  # 确保投影器参数也是float32
        projected_patch_embeddings = mm_projector(patch_features)
        projected_history_embeddings = []
        for history_feature in history_features:
            projected_history_embedding = mm_projector(history_feature)
            projected_history_embeddings.append(projected_history_embedding)
        projected_history_embeddings = torch.cat(projected_history_embeddings, dim=1)

        ## VTM
        # projected_history_embeddings = projected_history_embeddings.mean(dim=1, keepdim=True)


        # Mamba setting
        # mb_projector = self.get_mb_projector()
        # projected_history_embeddings = mb_projector(history_features)


        # projected_patch_embeddings = torch.cat(projected_patch_embedding[::-1], dim=1)
        projected_patch_attention_mask = None

        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1] + projected_history_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # # 临时调试代码，查看可用的嵌入方法
        # llm_model = self.get_llm()
        # print("Available embedding methods:")
        # print([attr for attr in dir(llm_model) if 'embed' in attr.lower()])
        # print([attr for attr in dir(llm_model.model) if 'embed' in attr.lower()])

        # 通常Llama模型使用：
        # 方法1: llm_model.model.embed_tokens(input_ids)
        # 方法2: llm_model.get_input_embeddings()(input_ids)
        # input_embeddings = self.get_llm().embed_input_ids(input_ids)
        input_embeddings = self.get_llm().model.embed_tokens(input_ids)


        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                projected_history_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1] + projected_history_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [
                    labels[multimodal_indices, :1],
                    projected_patch_labels,
                    labels[multimodal_indices, 1:]
                ],
                dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.get_llm()(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
                (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.llm._reorder_cache(*args, **kwargs)


    @torch.inference_mode()
    def predict_action(
            # self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
            self,
            image: Image,
            instruction: str,
            norm_stats,
            action_tokenizer=None,
            unnorm_key: Optional[str] = None,
            same_seq: bool = False,
            **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        # prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
        prompt_builder = Llama3ChatPromptBuilder("navila")
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        image_processor = self.get_vision_tower().image_processor
        tokenizer = self.tokenizer
        # if not same_seq:
        #     self.get_history_mamba().reset_state()

        # image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer
        #
        # # Build VLA Prompt
        # prompt_builder = self.get_prompt_builder()
        # prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        # prompt_text = prompt_builder.get_prompt()
        #
        # # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, (LlamaTokenizerFast, PreTrainedTokenizerFast)):
            # 检查是否是 Llama3
            if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size == 128000:
                # if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
                #     tokenizer.bos_token = "<|begin_of_text|>"
                # if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None:
                #     tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
                #
                #     # 同样确保 EOS token 设置正确
                # if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
                #     tokenizer.eos_token = "<|eot_id|>"
                # if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
                #     tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                # Llama3 通常不需要额外添加特殊 token，因为对话格式已经包含了
                # 如果需要添加，使用 Llama3 的特殊 token
                # assistant_start_token_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
                # 或者直接使用 token ID
                # input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([assistant_start_token_id]).long(), dim=0).to(self.device)), dim=1)
                pass  # Llama3 通常不需要额外处理
            else:
                # 原来的 Llama2 处理逻辑
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")


        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        # autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda"):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,  # Shape: [1, seq]
                # pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, ...]
                pixel_values=pixel_values,
                max_new_tokens=self.get_action_dim(norm_stats, unnorm_key),
                # do_sample=True,
                do_sample=False,
                # temperature=0.7,
                bos_token_id=tokenizer.bos_token_id,
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(norm_stats, unnorm_key):]
        print(f"Predicted action: {predicted_action_token_ids}")
        normalized_actions = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(norm_stats, unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions


    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
                unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, norm_stats, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, norm_stats, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]





AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
