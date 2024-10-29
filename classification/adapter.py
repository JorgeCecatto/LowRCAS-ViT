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
        print("adapter", self.n_embd)
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.config = config
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
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
        dim = x
        residual = x if residual is None else residual
        x = x.flatten(2).mean(-1)
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
            
        up = up.unsqueeze(-1).unsqueeze(-1).expand(dim.size(0), dim.size(1), dim.size(2), dim.size(3))
        if len(self.drop_dimensions) != 0:
            up = self.drop_dim(up)
        if add_residual:
            output = up + residual
        else:
            output = up

        return output