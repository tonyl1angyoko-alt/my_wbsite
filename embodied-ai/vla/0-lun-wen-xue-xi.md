---
description: 'π0: A Vision-Language-Action Flow Model for  General Robot Control'
---

# Π0：论文学习



1. produces continuous actions via flow matching。这里的flow matching是什么\
   A：相当于diffusion从噪音到降噪的过程。获得从噪声到真实动作的流场。主动加噪声，作为数据喂给模型让他学会去噪过程
2. 训练具体任务也是先训练general任务再进行微调效果更好，为什么
3. general模型需要训练数据达到一定规模存在一个必须的阈值
4. training recipe的优化始终是最重要的一步
5. VLA训练时选择了不同具身的数据的融合
6. 传统方法：交叉熵离散的动作。采取flow matching可以获得连续动作。\
   借鉴以往的分权重（也就是分模块），VLM+expert

<figure><img src="../../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

7. 他们并非简单地照搬 Transfusion 的混合训练模式，而是通过创建一个专门负责处理机器人本体状态和动作的“专家网络”（拥有独立的参数），对原有思想进行了优化，并用实验证明了这种“专业化分工”的设计能够带来实实在在的性能好处。
8.  模型学习根据【状态，摄像头，语义】学习给出的动作A，但是

    * 动作块 (Action Chunk)：模型一次性预测出未来 H 个时间步的完整动作序列 。
    * H = 50：在这篇论文的任务中，模型会一次性生成接下来连续50步的动作 。这使得机器人的动作非常连贯和流畅，而不是一系列断断续续的、卡顿的动作

    “翻译”过程：模型会用不同的“编码器 (encoders)”把这三种不同格式的数据，“翻译”成统一的数学语言（即“same embedding space”，相同的嵌入空间）



pi0torch：

```python
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
```

### 理解代码

在代码中，这个过程由 `embed_prefix` (嵌入前缀) 和 `embed_suffix` (嵌入后缀) 这两个函数完成。我们先看 `embed_prefix`。

#### 1. `embed_prefix`：理解“看”和“听”

这个函数的唯一目标：把图像（“看”）和语言（“听”）转换成一长串的嵌入向量（Embeddings）。这些向量就是模型“大脑”能理解的语言。

它同时还会生成对应的 `pad_masks` (填充掩码) 和 `att_masks` (注意力掩码)。

```
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer...
        """
        embs = []
        pad_masks = []
        att_masks = []
```

* 解释：初始化三个空列表，用来收集所有前缀（图像、语言）的信息。

**处理图像**

```
        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
```

* `for img, ... in zip(...)`：模型可以接收多个图像（比如3个不同摄像头的视角），这里它一个一个地处理。
* `self.paligemma_with_expert.embed_image(img)`：这就是调用`PaliGemma`的“视觉大脑”。它会把一张 `(224, 224, 3)` 的图像转换成一堆向量，比如 `(B, 576, D)` (B是批量大小, 576是图像块数量, D是嵌入维度)。
* `embs.append(img_emb)`：把图像嵌入向量（比如576个）加入列表。
* `pad_masks.append(...)`：为这576个向量添加 `True`，表示它们都是“真实的”token，不是填充。

```
            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs
```

* `att_masks += [0] * num_img_embs`：这是关键！还记得 `make_att_2d_masks` 里的 `cumsum` 吗？
* 通过在这里添加 `num_img_embs` 个 (比如576个) `0`，我们是在说：“所有这些图像token都属于同一个注意力块”。它们的“块ID”在 `cumsum` 之后都会是 `0`。

**处理语言**

```
        # Process language tokens
        def lang_embed_func(lang_tokens):
            ...
        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
```

* `self.paligemma_with_expert.embed_language_tokens(lang_tokens)`：调用`PaliGemma`的“语言大脑”，把语言指令（一串token ID）转换成嵌入向量。
* `embs.append(lang_emb)`：把语言嵌入向量（比如10个）加入列表。
* `pad_masks.append(lang_masks)`：语言指令有长有短，这里使用它自带的填充掩码。
* `att_masks += [0] * num_lang_embs`：再次关键！我们又为语言token添加了 `num_lang_embs` 个 (比如10个) `0`。

**组合**

```
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        ...
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks
```

* `torch.cat(...)`：把所有图像向量和语言向量拼接（concatenate）成一个超长的序列。
* `att_masks` 的结果：
  * 假设有1张图 (576个token) 和1条指令 (10个token)。
  * `att_masks` 列表会是：`[0, 0, ..., 0 (共576个), 0, 0, ..., 0 (共10个)]`。
  * 当 `make_att_2d_masks` 对它进行 `cumsum` 时，得到的“块ID”向量会是：`[0, 0, 0, ..., 0, 0, 0]`。
* 结论：所有前缀token（图像+语言）的“块ID”都是 `0`。根据 `make_att_2d_masks` 的逻辑 (`ID_k <= ID_j`)，`0 <= 0` 永远为 `True`。这意味着所有图像和语言token之间可以相互完全关注（Full Attention）。

总结：`embed_prefix` 把图像和语言“打包”成一个整体，并设置好了注意力规则，允许它们内部自由地交流信息。

***

好的，我们接着看 `embed_suffix`。

“学徒”看完了菜谱（`embed_prefix`），现在他要开始处理\*\*手头的“活儿”\*\*了。

#### 2. `embed_suffix`：理解“做”和“时间”

这个函数的唯一目标：把机器人状态（`state`）、带噪动作（`noisy_actions`）和当前时间（`timestep`）转换成另一组嵌入向量。

这里是 `Pi0` 和 `Pi0.5` 变体差异最大的地方。

```
    def embed_suffix(self, state, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []
```

* 解释：同样，初始化三个空列表，用来收集所有后缀（状态、动作）的信息。

**(A) Pi0 模式：处理“状态”**

```
        if not self.pi05:  # <-- 如果是 Pi0 (pi05=False)
            ...
            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
```

* `if not self.pi05`：这个代码块只有在 Pi0 模式下才会执行。Pi0.5 模式会完全跳过这一步（它不使用 `state` 作为token）。
* `self.state_proj(state)`：调用 `__init__` 中定义的那个线性层，把 `32` 维的 `state` 向量\*\*投影（放大）\*\*到和模型一样的 `width` 维度 (比如 `2048`)。
* `embs.append(state_emb[:, None, :])`：`state_emb` 形状是 `(B, D)`，`[:, None, :]` 把它变成 `(B, 1, D)`，即一个长度为1的“token序列”。
* `pad_masks.append(...)`：为这个“状态token”添加 `True`，表示它是真实的。

```
            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]
```

* `att_masks += [1]`：极其关键！我们在 `att_masks` 列表里加了第一个 `1`。
* 回忆 `embed_prefix`：前缀（图像+语言）的 `att_masks` 全是 `0`。
* `cumsum` 累积和：
  * `att_masks` 列表现在是 `[0, 0, ..., 0, 1]`。
  * 对应的“块ID”会是 `[0, 0, ..., 0, 1]`。
* 含义：
  * 前缀 (ID 0) -> 后缀 (ID 1)：`1 <= 0` 为 `False`。前缀（菜谱）不能“偷看”后缀（动作）。这符合逻辑。
  * 后缀 (ID 1) -> 前缀 (ID 0)：`0 <= 1` 为 `True`。后缀（动作）可以“回头看”前缀（菜谱）。这也符合逻辑。

**(B) 所有模式：处理“时间”和“动作”**

```
        # Embed timestep using sine-cosine positional encoding...
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, ...
        )
        time_emb = time_emb.type(dtype=timestep.dtype)
```

* 解释：调用我们之前详细分析过的 `create_sinusoidal_pos_embedding` 函数。
* 输入：`timestep` (一个 `(B,)` 的时间向量，比如 `[0.7, 0.7, 0.7, ...]`) 和 `self.action_in_proj.out_features` (即 `width`，比如 `2048`)。
* 输出：`time_emb` 是一个 `(B, 2048)` 的时间嵌入向量。

```
        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
```

* `self.action_in_proj(noisy_actions)`：调用输入投影层。
* 输入：`noisy_actions` (形状 `(B, 16, 32)`)，`B` 是批量，`16` 是动作序列长度（`action_horizon`），`32` 是动作维度。
* 输出：`action_emb` (形状 `(B, 16, 2048)`)。

**(C) Pi0 vs Pi0.5：融合方式**

现在，模型有了 `action_emb` (形状 `(B, 16, 2048)`) 和 `time_emb` (形状 `(B, 2048)`)。它如何将两者结合？

Python

```
        if not self.pi05:  # <-- Pi0 模式
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                ...
                return self.action_time_mlp_out(x)
            
            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
```

* Pi0 模式 (拼接融合)：
  1. `time_emb.expand_as(action_emb)`：把 `(B, 2048)` 的 `time_emb` 复制16次，变成 `(B, 16, 2048)`。
  2. `torch.cat(...)`：在最后一个维度（`dim=2`）上拼接。`action_emb` (2048维) 和 `time_emb` (2048维) 拼成了 `(B, 16, 4096)`。
  3. `self.action_time_mlp_in(action_time_emb)`：这个MLP层把 `4096` 维压缩回 `2048` 维。
  4. 结果：得到一个 `(B, 16, 2048)` 的 `action_time_emb` 向量，它同时包含了动作和时间信息。
  5. `adarms_cond = None`：Pi0 不使用 AdaRMS。

Python

```
        else:  # <-- Pi0.5 模式
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                ...
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb
```

* Pi0.5 模式 (AdaRMS 调节)：
  1. `time_mlp_func(time_emb)`：`time_emb` (形状 `(B, 2048)`) 被送入它自己的 `time_mlp`。
  2. `action_time_emb = action_emb`：注意！ `action_time_emb` 就是 `action_emb`！时间信息根本没有被拼接进去！
  3. `adarms_cond = time_emb`：处理过的 `time_emb` 被单独存放在 `adarms_cond` 变量里。
  4. 结果：`action_time_emb` (形状 `(B, 16, 2048)`) 只包含动作信息。`adarms_cond` (形状 `(B, 2048)`) 只包含时间信息。这个 `adarms_cond` 会在 `forward` 函数中被传递给 `PaliGemmaWithExpertModel`，用于动态调节 `ActionExpert` 模型内部的 `RMSNorm` 层。

**(D) 组合后缀**

Python

```
        # Add to input tokens
        embs.append(action_time_emb)
        ...
        pad_masks.append(action_time_mask)

        # Set attention masks ...
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))
```

* `embs.append(action_time_emb)`：把融合后的 `(B, 16, 2048)` 向量（`action_time_emb`）加入列表。
* `att_masks += [1] + ([0] * 15)`：这是第二个关键。
  * 我们在 `att_masks` 列表里加了一个 `1` 和 `15` 个 `0`。

**最终的 `att_masks` 结构 (Pi0 模式)**

我们来总结一下 `att_masks` 列表现在的样子 (假设1个图像=576, 语言=10, 状态=1, 动作=16)：

* 前缀 (Prefix): `[0, 0, ..., 0]` (共 576+10 = 586 个 `0`)
* 后缀 (Suffix):
  * 状态 (State): `[1]` (来自 Pi0 模式 `if not self.pi05`)
  * 动作 (Action): `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` (来自 `[1] + ([0] * 15)`)

att\_masks 完整列表：

\[0...0, 1, 1, 0...0]

(586个0, 接着1, 接着1, 接着15个0)

cumsum 后的“块ID”：

\[0...0, 1, 2, 2...2]

* 前缀 (图像+语言): 块ID = `0`
* 状态 (State): 块ID = `1`
* 动作 (Action):
  * 第1个动作token: 块ID = `2`
  * 第2-16个动作token: 块ID = `2`

这实现了什么注意力？

1. 块0 (前缀): 只能看块0 (ID `0 <= 0`)。
2. 块1 (状态): 能看块0和块1 (ID `0 <= 1`, `1 <= 1`)。
3. 块2 (动作): 能看块0、块1和块2 (ID `0 <= 2`, `1 <= 2`, `2 <= 2`)。

结论：`embed_prefix` 和 `embed_suffix` 巧妙地利用 `0` 和 `1`，与 `make_att_2d_masks` 配合，构建了一个完美的“前缀-LM（Prefix-LM）”注意力掩码，它强制模型：

> 动作（和状态）可以关注（attend to）图像和语言（菜谱），
>
> 但反过来，
>
> 图像和语言（菜谱）永远不能“偷看”未来的动作。

这就是这两个嵌入函数的核心功能。

好的，我们已经万事俱备了。

我们已经准备好了：

1. `prefix_embs` (前缀嵌入)：来自 `embed_prefix`，包含了（图像+语言）。
2. `suffix_embs` (后缀嵌入)：来自 `embed_suffix`，包含了（状态+带噪动作 $$ $x_t$ $$+时间 $$ $t$ $$）。
3. `pad_masks` (填充掩码)：一个长 `(B, N)` 向量，标记哪些是真实token。
4. `att_masks` (注意力掩码)：一个长 `(B, N)` 向量，用于定义“块ID”（比如 `[0,0,0, 1, 2,2,2]`）。

现在，我们进入训练的核心：`forward` 方法的主体部分。

#### 3. `forward` 方法：执行“看题-预测-打分”

`forward` 方法的目标是计算损失（loss）。

Python

```
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss..."""
```

* 解释：它接收 `observation`（观测）和 `actions`（标准答案）。

Python

```
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
```

* 解释：
  1. `_preprocess_observation`：只是个辅助函数，把数据整理成张量。
  2. `sample_noise`：生成一个和 `actions` 形状相同的随机高斯噪声 $$ $\epsilon$ $$。
  3. `sample_time`：调用我们分析过的 `sample_beta`，生成一个 $$ $\alpha=1.5, \beta=1$ $$ 的、偏向1.0的随机时间 $$ $t$ $$。

Python

```
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
```

* 解释：流匹配（Flow Matching）的核心！
  1. `time_expanded`：把 `(B,)` 的时间 $$t$$ 扩展成 `(B, 1, 1)`，以便和 `(B, 16, 32)` 的动作进行广播运算。
  2. `x_t = ...`：这就是在制作“练习题”。它在“完美动作” `actions` (在 $$ $t=0$ $$) 和“纯噪声” `noise` (在 $$ $t=1$ $$) 之间进行线性插值。
  3. `u_t = noise - actions`：这就是“标准答案”。$$ $u_t$ $$ 是模型需要预测的目标“速度向量”。

Python

```
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
```

* 解释：调用我们刚刚详细分析过的那两个函数。
  * `embed_prefix` 处理（图像+语言）。
  * `embed_suffix` 处理（状态, 带噪动作 $$ $x_t$ $$, 时间 $$ $t$ $$）。
  * 注意 `x_t` (练习题) 被传进去了，而不是 `actions` (标准答案)。
  * `adarms_cond`：在 Pi0.5 模式下，这里会接收到处理过的时间嵌入；在 Pi0 模式下，这里是 `None`。

Python

```
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
```

* 解释：
  1. `torch.cat`：把前缀和后缀的所有掩码拼接起来，形成一个完整的序列。
  2. `make_att_2d_masks(...)`：调用我们分析过的函数，根据拼接后的 `pad_masks` 和 `att_masks`（那个 `[0,0,1,2,2]`）来生成最终的 `(B, N, N)` 二维注意力矩阵。
  3. `position_ids`：计算每个token的（非填充）位置索引。

Python

```
        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
```

* 解释：`_prepare_attention_masks_4d` 是个辅助函数。它把 `True/False` 的掩码矩阵转换成 Transformer (Gemma) 期望的 `0.0` 和 负无穷（`-2.38e38`）的浮点数矩阵。

Python

```
        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out
```

* 解释：
  1. 这里定义了一个 `forward_func`，它封装了对模型核心 `self.paligemma_with_expert.forward` 的调用。
  2. `attention_mask=att_2d_masks_4d`：传入我们精心制作的前缀-LM 掩码。
  3. `inputs_embeds=[prefix_embs, suffix_embs]`：同时传入前缀和后缀的嵌入。模型内部（`PaliGemmaWithExpertModel`）会分别把它们交给 `PaliGemma`（视觉大脑）和 `ActionExpert`（动作大脑）处理。
  4. `adarms_cond=[None, adarms_cond]`：
     * 第一个 `None` 对应 `PaliGemma`（它不用 AdaRMS）。
     * 第二个 `adarms_cond` 对应 `ActionExpert`。在 Pi0.5 模式下，这里传入了时间嵌入；在 Pi0 模式下，这里传入了 `None`。
  5. `(_, suffix_out), _`：模型会返回 `(prefix_output, suffix_output)`。我们只关心 `suffix_out`（后缀的输出），因为这才是“动作大脑”的思考结果。

Python

```
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
```

* 解释：执行上面定义的 `forward_func`。

Python

```
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
```

* 解释：`suffix_out` 包含了（状态+动作）的输出。我们只关心最后 `action_horizon` 个（比如16个）token，因为它们对应的是动作。

Python

```
        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
```

* 解释：
  1. `self.action_out_proj`：调用 `__init__` 中定义的输出投影层。
  2. 它把 `suffix_out`（形状 `(B, 16, 2048)`）压缩回 `(B, 16, 32)`。
  3. `v_t`：这就是学徒（模型）对“修正方向”的最终预测。

Python

```
        return F.mse_loss(u_t, v_t, reduction="none")
```

* 解释：最后一步：打分！
* 计算标准答案 `u_t`（`noise - actions`）和模型预测 `v_t` 之间的均方误差（MSE）。
* 这个 `loss` 会被返回，PyTorch 的优化器会用它来更新模型的所有参数（反向传播）。

***

`forward` (训练) 的流程就是这样。

好的，现在我们来看 `sample_actions` (推理)。

这是学徒真正“上岗做菜”的过程。这个方法没有标准答案 `actions`，它的目标是从零（噪声）开始，生成动作。

这个过程要巧妙得多，因为它是一个迭代循环，并且用到了一个关键的性能优化：KV 缓存（KV Cache）。

#### 4. `sample_actions` (推理): “迭代做菜”

Python

```
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action ..."""
```

* `@torch.no_grad()`：这是一个 PyTorch 装饰器，它告诉 PyTorch：“在这个函数里，不要计算梯度”。这能极大地节省显存和计算资源，因为在推理时我们不需要反向传播。
* 输入：只接收 `observation`（观测）。`num_steps`（比如10）告诉模型要迭代多少步来去噪。

Python

```
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
```

* 解释：
  1. 获取批量大小 `bsize`。
  2. 如果外部没有提供 `noise`，它就自己生成一个。
  3. `noise` 在这里就是 $$x_t$$ 的初始值（在 $$t=1.0$$ 时）。这就是我们开始“做菜”的“纯面团”。

Python

```
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
```

* 解释：和 `forward` (训练) 一样，它首先预处理观测数据，并调用 `embed_prefix` 来获取\*\*前缀（图像+语言）\*\*的嵌入和掩码。

***

#### 🚀 关键优化：KV 缓存

接下来的几行是推理过程中最重要的优化。

Python

```
        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        ...

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # <-- 只传入前缀！
            use_cache=True,                     # <-- 告诉模型“请返回缓存”！
        )
```

* 思考一下：在 `num_steps=10` 的迭代过程中，“菜谱”（图像+语言）是永远不会改变的。
* 如果我们在每一步迭代（共10次）都让 `PaliGemma`（视觉大脑）重新看一遍图像和语言，那将是巨大的浪费。
* 解决方案：我们在这里（循环开始之前）只运行一次 `PaliGemma`。
  1. `inputs_embeds=[prefix_embs, None]`：我们\*\*只把 `prefix_embs`（前缀）\*\*传给模型。`suffix_embs`（后缀）部分是 `None`。
  2. `use_cache=True`：我们告诉模型：“请计算 `prefix_embs` 的\*\*键（Key）和值（Value）\*\*向量，并把它们作为 `past_key_values`（KV缓存）返回给我。”
* `past_key_values`：这就是学徒的那张“便签”。它包含了 `PaliGemma` 对“菜谱”的全部理解（所有层的K和V向量）。

***

#### 🌀 迭代去噪循环

现在我们有了“便签”（`past_key_values`）和“面团”（`noise`），开始循环。

Python

```
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise  # 我们的“面团”，初始是纯噪声
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2: # 循环直到 t 约等于 0
            expanded_time = time.expand(bsize)
```

* 解释：设置时间步长 `dt`（一个小的负数，比如 `-0.1`）。`x_t` 被初始化为 `noise`，`time` 被初始化为 `1.0`。

Python

```
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,  # <-- 传入“便签”
                x_t,              # <-- 传入当前的“面团”
                expanded_time,
            )
```

* 解释：这是循环的核心。它调用一个辅助函数 `denoise_step`。
* `denoise_step` 的任务：执行一次预测。它接收“便签”（`past_key_values`）、当前的“面团”（`x_t`）、`state` 和 `time`，然后返回对当前“修正方向” $$ $v_t$ $$ 的预测。
* (我们稍后会深入 `denoise_step` 函数。)

Python

```
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
```

* 解释：欧拉法（Euler method）积分。
  1. `x_t = x_t + dt * v_t`：“修正面团”。我们根据模型预测的方向 $$ $v_t$ $$ 和时间步长 $$ $dt$ $$，来更新我们的 $$ $x_t$ $$。
  2. `time += dt`：更新时间 $$ $t$ $$（比如从 `1.0` 变成 `0.9`）。
* 这个循环会重复 `num_steps` 次（比如10次）。

Python

```
        return x_t
```

* 解释：当 `time` 约等于 `0` 时，循环结束。此时的 `x_t`（即 $$ $x_0$ $$）已经从“纯噪声”被完全去噪成了“干净的、可执行的动作”。函数将其返回。

***

现在，这个推理流程应该很清楚了。`sample_actions` 负责设置（KV缓存）和循环（迭代），而真正的工作是在 `denoise_step` 里完成的。

好的，我们来看 `denoise_step` 函数。

这是 `sample_actions` (推理) 循环内部调用的核心辅助函数。

它的目标非常明确：给定 $$ $t$ $$ 时刻的“面团”（`x_t`）和“便签”（`past_key_values`），仅执行一步去噪，并返回预测的速度 $$ $v_t$ $$。

#### 5. `denoise_step` (推理): “查便签，修面团”

Python

```
    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,  # <-- “便签” (来自PaliGemma)
        x_t,              # <-- 当前的“面团” (带噪动作)
        timestep,         # <-- 当前的时间 t
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
```

* 解释：接收所有需要的当前状态。

Python

```
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
```

* 解释：
  1. 再次调用 `embed_suffix`。
  2. 但这一次，它嵌入的是推理时的 $$ $state$ $$、当前的 $$ $x_t$ $$ 和当前的 $$ $timestep$ $$。
  3. 返回 `suffix_embs` (后缀嵌入) 和 `adarms_cond` (Pi0.5 模式下的时间嵌入)。

Python

```
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
```

* 解释：这几行在动态地创建注意力掩码。
* 思考一下：在 `forward` (训练) 中，前缀和后缀是一起被送入模型的，所以我们可以一次性创建 `(N+M, N+M)` 的大掩码。
* 但在推理时：前缀（`prefix`）已经被“缓存”了。我们现在只把后缀（`suffix`）送入模型。
* 所以：`full_att_2d_masks` 是一个 `(B, M, N+M)` 形状的掩码（M=后缀长度, N=前缀长度）。
* 它的作用是告诉模型里的 `ActionExpert` (动作大脑)：
  1. 你的 `suffix` token（`M`个）可以回头看 `prefix` token（`N`个）。( `prefix_pad_2d_masks` 部分)
  2. 你的 `suffix` token 之间必须因果地（causally）相互关注。( `suffix_att_2d_masks` 部分)

Python

```
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
```

* 解释：计算 `position_ids`（位置ID）。
* `ActionExpert` 需要知道 `suffix_embs` 是在 `prefix_embs` 之后的。
* `prefix_offsets` 计算出前缀的实际长度（比如596）。
* `position_ids` 就会是 `[596, 597, 598, ...]`。这告诉 `ActionExpert`：“你们是序列中排在第596号位置之后的token”。

Python

```
        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        ...

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,  # <-- 传入“便签”！
            inputs_embeds=[None, suffix_embs],  # <-- 只传入后缀！
            use_cache=False,                    # <-- 不再需要缓存
            adarms_cond=[None, adarms_cond],
        )
```

* 解释：这是 `denoise_step` 最核心的调用！
* `inputs_embeds=[None, suffix_embs]`：只传入后缀。`None` 告诉 `PaliGemma`（视觉大脑）：“你这次没事做，休息。”
* `past_key_values=past_key_values`：传入“便签”。
* 发生了什么？
  1. `PaliGemma` (视觉大脑) 完全跳过。
  2. `ActionExpert` (动作大脑) 启动。
  3. `ActionExpert` 接收 `suffix_embs` 作为它的查询（Query）。
  4. 它使用它自己的 `suffix_embs`（用于后缀内部的自注意力）和\*\*`past_key_values`（用于交叉注意力，即“查便签”）作为键（Key）和值（Value）\*\*。
  5. `attention_mask` 确保了这种查询是合法的（后缀可以看前缀）。
  6. `adarms_cond` 在 Pi0.5 模式下传入时间 $$ $t$ $$，来调节 `ActionExpert`。
* 结果：`ActionExpert` 高效地计算出了输出，它完全没有重新计算图像和语言，只是“查阅”了它们。

Python

```
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
```

* 解释：和 `forward` (训练) 的结尾一模一样。
  1. `outputs_embeds[1]`：`paligemma_with_expert.forward` 的输出是 `(prefix_out, suffix_out)`。`prefix_out` 此时是 `None`，我们只取 `suffix_out`。
  2. `suffix_out[:, -self.config.action_horizon :]`：只保留动作对应的token。
  3. `self.action_out_proj(suffix_out)`：将高维输出\*\*投影（压缩）\*\*回 `32` 维的动作空间。
* 返回：返回预测的速度 $$ $v_t$ $$。

这个 $$ $v_t$ $$ 会被返回给 `sample_actions` 中的 `while` 循环，用于执行 `x_t = x_t + dt * v_t`，从而完成这一步的“面团修正”。

***

至此，我们就完整地分析了 `PI0Pytorch` 类的所有核心逻辑：

1. `__init__`：搭建两个大脑（PaliGemma, ActionExpert）。
2. `embed_prefix` / `embed_suffix`：准备“菜谱”和“动作”的嵌入。
3. `make_att_2d_masks`：创建巧妙的注意力规则。
4. `forward` (训练)：看题（$$ $x_t$ $$）、预测（$$ $v_t$ $$）、对答案（$$ $u_t$ $$）。
5. `sample_actions` (推理)：先做“便签”（KV缓存），然后循环调用 `denoise_step`。
6. `denoise_step` (推理核心)：“查便签”、“修面团”（$$ $x_t$ $$），返回 $$ $v_t$ $$。

