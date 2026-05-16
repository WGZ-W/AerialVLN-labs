
import torch
# from mamba_ssm import Mamba
# from mamba_ssm import Mamba2
from torch import nn


class VisionMambaBlock(nn.Module):
    """视觉Mamba块"""

    def __init__(self,
                 d_model,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 # device=None,
                 ):
        super().__init__()


        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # self.device = device

        # 前向Mamba
        self.model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            # device=device,
        )

        # 归一化和前馈网络
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x, conv_state=None, ssm_state=None, inference=False):
        # x: (B, num_patches, dim)

        batch_size, seq_len, dim = x.shape

        outputs = []

        if conv_state is None:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=x.device,
                dtype=x.dtype
            )

        if ssm_state is None:
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=x.device,
                dtype=x.dtype
            )

        if inference:
            for t in range(x.shape[1]):
                output, conv_state, ssm_state = self.model.step(
                    x[:, t:t+1, :],
                    conv_state,
                    ssm_state,
                )
                outputs.append(output)

            outputs = torch.cat(outputs, 1)
        else:
            outputs = self.model(x)

        # output = self.model(x)
        # hidden_states = output
        # output = self.norm(output)

        return outputs, conv_state, ssm_state


class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=384,
                 patch_size=24,
                 in_chans=3,
                 embed_dim=768,
                 depth=4,
                 dtype:str = None,
                 # device=None,
                 ):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.hidden_states = None
        self.dtype = eval(dtype)
        # self.device = device

        self.conv_state = None
        self.ssm_state = None

        # 块嵌入
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            # device=device,
            dtype=self.dtype,
        )

        # 可学习的位置嵌入
        self.pos_embed_history = nn.Parameter(
            # 原始 Mamba 设置，8张图像
            torch.randn(1, self.num_patches * 8, embed_dim, dtype=self.dtype)

            # Key + Mamba 设置图像数量为4
            # torch.randn(1, self.num_patches * 4, embed_dim, dtype=self.dtype)
        )
        # ).to(self.device)

        # 可学习的位置嵌入
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim, dtype=self.dtype)
        )
        # ).to(self.device)

        # 视觉Mamba块堆叠
        self.blocks = nn.ModuleList([
            VisionMambaBlock(embed_dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):

        multi_values = []
        if x.dim() == 5:
            for idx in range(x.shape[1]):
                frame = self.patch_embed(x[:, idx, :, :, :])  # (B, C, H, W)
                # B, C, H, W = x.shape
                frame = frame.flatten(2).transpose(1, 2)  # (B, num_patches, C)

                multi_values.append(frame)
            frame = torch.cat(multi_values, dim=1)
            frame = frame + self.pos_embed_history
            for block in self.blocks:
                frame, self.conv_state, self.ssm_state = block(
                    frame,
                    self.conv_state,
                    self.ssm_state,
                    inference=False,
                )

            # for idx in range(x.shape[1]):
            #     frame = self.patch_embed(x[:, idx, :, :, :])  # (B, C, H, W)
            #     #     # B, C, H, W = x.shape
            #     frame = frame.flatten(2).transpose(1, 2)  # (B, num_patches, C)
            #     # 添加位置编码
            #     frame = frame + self.pos_embed
            #     for block in self.blocks:
            #         frame, self.conv_state, self.ssm_state = block(
            #             frame,
            #             self.conv_state,
            #             self.ssm_state,
            #             inference=True,
            #         )

            self.reset_state()

        else:

            frame = self.patch_embed(x)
            frame = frame.flatten(2).transpose(1, 2)
            frame = frame + self.pos_embed

            for block in self.blocks:
                frame, self.conv_state, self.ssm_state = block(
                    frame,
                    self.conv_state,
                    self.ssm_state,
                    inference=True,
                )


        # 块嵌入
        # for x in multi_images:
        #     x = self.patch_embed(x)  # (B, C, H, W)
        #     B, C, H, W = x.shape
        #     x = x.flatten(2).transpose(1, 2)  # (B, num_patches, C)
        #
        #     # 添加位置编码
        #     x = x + self.pos_embed
        #
        # # 通过Mamba块
        # for block in self.blocks:
        #     x, self.conv_state, self.ssm_state = block(x)

        # 全局平均池化
        frame = self.norm(frame)

        return frame


    def reset_state(self):
        self.ssm_state = None
        self.conv_state = None




class HistoryMamba:
    def __init__(self):
        self.model = Mamba(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
            conv_bias=True,
            bias=False
        )

        self.hidden_states = None



if __name__ == "__main__":
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionMamba(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        device=device
    )

    x = torch.randn(1, 3, 224, 224).to(device)
    out = model(x)
    print(out.shape)
