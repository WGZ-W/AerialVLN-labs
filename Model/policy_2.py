from typing import Any

from torch import nn as nn
import torch

from src.common.param import args
from Model.utils.common import CategoricalNet, CustomFixedCategorical


class ILPolicy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()

        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        step=0,
    ):
        if args.policy_type in ['seq2seq', 'cma']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
        else:
            raise NotImplementedError

        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_hidden_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        if args.policy_type in ['seq2seq', 'cma']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
        else:
            raise NotImplementedError

        return self.action_distribution(features)


import torch.nn as nn
import torch.distributions as D
from llava.model.language_model.llava_llama import LlavaLlamaModel2
from llava.model.configuration_llava import LlavaConfig


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


class LLaVAPolicy(nn.Module):
    def __init__(
            self,
            vla_path,
            action_space,
    ):
        super().__init__()

        config = LlavaLlamaConfig.from_pretrained(vla_path, resume=False)
        config.model_dtype = torch.bfloat16
        config.model_dtype = config.model_dtype.__str__()
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = vla_path

        self.vlm = LlavaLlamaModel2(
            config=config,
            # attn_implementation="flash_attention_2",
            attn_implementation="eager",
            model_max_length=4096,
        )

        # 冻结全部 VLM 参数
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.tokenizer = self.vlm.tokenizer
        self.image_processor = self.vlm.vision_tower.image_processor

        llm_hidden_size = self.vlm.llm.config.hidden_size
        self.action_head = nn.Linear(llm_hidden_size, action_space.n)

    def forward(self, observations, rnn_states, prev_actions, masks, **kwargs):
        pixel_values = observations['pixel_values']

        print(f"[DEBUG] pixel_values.shape: {pixel_values.shape}")  # 期望 [N, 3, H, W]
        print(f"[DEBUG] pixel_values.dtype: {pixel_values.dtype}")
        print(f"[DEBUG] pixel_values.device: {pixel_values.device}")

        # 调试：单独运行 vision tower
        with torch.no_grad():
            vision_tower = self.vlm.get_vision_tower()
            # 假设 vision_tower.vision_tower 是 SiglipModel
            vision_outputs = vision_tower(pixel_values)
            print(f"[DEBUG] last_hidden_state.shape: {vision_outputs.last_hidden_state.shape}")
            if hasattr(vision_outputs, 'pooler_output'):
                print(f"[DEBUG] pooler_output.shape: {vision_outputs.pooler_output.shape}")

        input_ids = observations['input_ids']

        history_values = observations.get('history_values', None)

        outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            history_values=history_values,
            attention_mask=observations.get('attention_mask'),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        last_hidden = last_hidden.float()
        action_logits = self.action_head(last_hidden)

        return action_logits, None

    def act(
            self,
            observations,
            rnn_states,
            prev_actions,
            masks,
            deterministic=False,
            step=0,
    ):
        logits, rnn_states = self.forward(observations, rnn_states, prev_actions, masks)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        actions = actions.unsqueeze(1)
        return actions, rnn_states

    def build_distribution(self, observations, rnn_states, prev_actions, masks):
        logits, _ = self.forward(observations, rnn_states, prev_actions, masks)
        return D.Categorical(logits=logits)