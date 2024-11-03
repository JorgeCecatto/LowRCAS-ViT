# --------------------------------------------------------
# References:
# https://github.com/sun-hailong/CVPR24-Ease/blob/main/backbone/vit_ease.py
# --------------------------------------------------------

import torch.nn as nn
import torch
import numpy as np
import math

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 drop_dimensions=None):
        super().__init__()
       
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.drop_dimensions = drop_dimensions
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.config = config
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.down_size)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        emb_size = int(int(self.down_size/4) ** 2)

        self.pooling_layer = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.unpool = nn.MaxUnpool2d(4, stride=4)
        self.conv_layer = nn.Conv2d(in_channels=self.n_embd, out_channels=1, kernel_size=(1, 1))
        self.down_proj = nn.Linear(emb_size, int(self.n_embd/12))
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(int(self.n_embd/12), emb_size)
        self.deconv_layer = nn.ConvTranspose2d(in_channels=1, out_channels=self.n_embd, kernel_size=(1, 1), stride=(1, 1))
        if len(self.drop_dimensions) != 0:
          self.drop_dim = nn.Conv2d(self.drop_dimensions[0], self.drop_dimensions[1], 3, stride=2, padding=1)
        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        original_dim = x
        residual = x if residual is None else residual
        x = self.conv_layer(x)
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        x, indices = self.pooling_layer(x)
        dim = x.size()
        x = x.view(x.size(0), -1)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        up = up.unsqueeze(1)
        up = up.view(dim[0], 1, dim[2], dim[3])
        up = self.unpool(up, indices, output_size=original_dim.size())
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        up = self.deconv_layer(up)
        if len(self.drop_dimensions) != 0:
            up = self.drop_dim(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output