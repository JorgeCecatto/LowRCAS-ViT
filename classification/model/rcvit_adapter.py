# --------------------------------------------------------
# References:
# https://github.com/sun-hailong/CVPR24-Ease/blob/main/backbone/vit_ease.py
# --------------------------------------------------------


import torch
import torch.nn as nn
from model import *
from timm.models import create_model
import utils as utils
from adapter import Adapter
import math
import numpy as np
class RCViTWithAdapters(nn.Module):
    def __init__(self, model='rcvit_xs', pretrained=False, num_classes=1000, drop_path_rate=0.0, layer_scale_init_value=1e-6, head_init_scale=1.0,input_res=384,
        classifier_dropout=0.0, distillation=False, tuning_config = None, device = 'cuda:0', training = False, finetune=None, **kwargs):

        super(RCViTWithAdapters, self).__init__()
        self.training = training
        self.dist = distillation
        #self.input_res=input_res
        self.input_res = input_res
        self.rcvit = create_model(
                                    model,
                                    pretrained=False,
                                    num_classes=1000,
                                    drop_path_rate=0.0,
                                    layer_scale_init_value=layer_scale_init_value,
                                    head_init_scale = head_init_scale,
                                    input_res = input_res,
                                    classifier_dropout = classifier_dropout,
                                    distillation=False
        )
        self.config = tuning_config
        self.config.d_model = tuning_config.ffn_num
        self.config._device = tuning_config._device
        if finetune is not None:
                checkpoint = torch.load(finetune, map_location="cpu")
                state_dict = checkpoint["model"]
                rcvit = utils.load_state_dict(self.rcvit, state_dict)
                for name, p in rcvit.named_parameters():
                    if name in rcvit.missing_keys:
                        p.requires_grad = True  # Descongelar se a chave está faltando
                    else:
                        p.requires_grad = False  # Congelar se a chave está presente
        self.adapter_list = nn.ModuleList()
        self.dimensions_in = []
        self.dimensions_out = []
        self.get_size_of_embeddings()
        self.get_new_adapter()
        self.head = nn.Linear(in_features=220, out_features=2, bias=True)

    def get_new_adapter(self):
        config = self.config
        if config.ffn_adapt:
            for i in range(len(self.rcvit.network)):
                self.config.d_model = self.dimensions_in[i]
                config.ffn_num = self.dimensions_out[i]
                drop_dimensions = []
                if self.dimensions_in[i] !=self.dimensions_out[i]:
                  drop_dimensions = [self.rcvit.embed_dims[int(np.floor(i/2))], self.rcvit.embed_dims[int(np.ceil(i/2))]]
                adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                        init_option=config.ffn_adapter_init_option,
                                        adapter_scalar=config.ffn_adapter_scalar,
                                        adapter_layernorm_option=config.ffn_adapter_layernorm_option, drop_dimensions = drop_dimensions
                                        ).to(self.config._device)
                self.adapter_list.append(adapter)
            self.adapter_list.requires_grad_(True)
        else:
            print("====Not use adapter===")

    def get_size_of_embeddings(self):
      x = torch.randn(1, 3, self.input_res, self.input_res)
      with torch.no_grad():
        x = self.rcvit.patch_embed(x)
        for idx, block in enumerate(self.rcvit.network):
          self.dimensions_in.append(x.size(3))
          x = block(x)
          self.dimensions_out.append(x.size(3))

    def forward_tokens_train(self, x):
        outs = []
        for idx, block in enumerate(self.rcvit.network):
          adapt = self.adapter_list[idx]
          if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
          else:
            adapt_x = None
          x = block(x)
          if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                   raise ValueError(self.rcvit.config.ffn_adapt)

          if self.rcvit.fork_feat and idx in self.rcvit.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
          if self.rcvit.fork_feat:
            return outs
        return x

    def forward_tokens(self, x):
        outs = []

        for idx, block in enumerate(self.rcvit.network):
            x = block(x)
            if self.rcvit.fork_feat and idx in self.rcvit.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.rcvit.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.rcvit.patch_embed(x)
        if self.training:
          x = self.forward_tokens_train(x)
        else:
          x = self.forward_tokens_train(x)
        if self.rcvit.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.rcvit.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.rcvit.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out