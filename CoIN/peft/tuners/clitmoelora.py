# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    mark_only_current_lora_as_trainable,
    mark_current_lora_wo_router_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)

from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class CoINMOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_CoIN`]
    """
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)
    task_id: int = field(default=8)
    lora_method: str = field(default=None)
    trans_hidden_dim: int = field(default=100)
    lora_add_method: str = field(default="full")
    # target_layers: Optional[List[int]] = field(default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    target_layers: Optional[List[int]] = field(default_factory=list)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CoIN


class CoINMOELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_clitmoelora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        if self.peft_config[adapter_name].lora_method == None:
            mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
            if self.peft_config[adapter_name].inference_mode:
                _freeze_adapter(self.model, adapter_name)
        elif self.peft_config[adapter_name].lora_method == "proglora":
            # pdb.set_trace()
            mark_current_lora_wo_router_as_trainable(self.model, self.peft_config[adapter_name].bias, self.peft_config[adapter_name].task_id)
            if self.peft_config[adapter_name].inference_mode:
                _freeze_adapter(self.model, adapter_name)


    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        if "c_attn" in lora_config.target_modules and "w1" in lora_config.target_modules:
            lora_config.target_modules = lora_config.target_modules
        elif lora_config.lora_method == "proglora":
            # pdb.set_trace()
            target_modules=['k_proj', 'v_proj', 'down_proj', 'up_proj', 'gate_proj', 'q_proj', 'o_proj']
            lora_config.target_modules = target_modules
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

####################### ADD on 2025.11 for fix bottom layers ###################################
            # if lora_config.lora_add_method != "full" and lora_config.target_layers is not None:
            #     # 尝试提取层索引。Llama层的命名通常包含 '.layers.<index>.'
            #     match = re.search(r"\.layers\.(\d+)\.", key)
            #     if match:
            #         layer_idx = int(match.group(1))
            #         if layer_idx not in lora_config.target_layers:
            #             continue # 跳过不在目标列表中的层
            #     else:
            #         # 模块不在 layer 中 (如 embeddings, lm_head)，根据需要选择是否跳过
            #         pass
##########################################################

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
            "task_id": lora_config.task_id,
            "lora_method": lora_config.lora_method,
            "trans_hidden_dim": lora_config.trans_hidden_dim,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = CoINMOELoraLinear(adapter_name, in_features, out_features, 
                                                    bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_clitmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

class CoINMOELoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int, task_id: int, lora_method: str, trans_hidden_dim: int):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        self.task_id = task_id
        self.lora_method = lora_method
        self.trans_hidden_dim = trans_hidden_dim
        if self.lora_method == "proglora":
            self.expert_num = task_id+1
    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: CoINMOELinearA(self.in_features, r, self.expert_num, self.task_id, self.lora_method, self.trans_hidden_dim)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: CoINMOELinearB(r, self.out_features, self.expert_num, self.task_id, self.lora_method, self.trans_hidden_dim)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            if self.lora_method == None:
                for i in range(self.expert_num):
                    nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)
            elif self.lora_method == "proglora":
                for i in range(self.expert_num):
                    nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)

class CoINMOELoraLinear(nn.Linear, CoINMOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.task_id = kwargs.pop("task_id", True)
        self.lora_method = kwargs.pop("lora_method", True)
        self.trans_hidden_dim = kwargs.pop("trans_hidden_dim", True)

        if self.lora_method == None:
            nn.Linear.__init__(self, in_features, out_features, **kwargs)
            CoINMOELoraLayer.__init__(self, in_features=in_features, 
                                out_features=out_features, 
                                expert_num=self.expert_num,
                                task_id=self.task_id,
                                lora_method=self.lora_method,
                                trans_hidden_dim=self.trans_hidden_dim,)
        
            # init the Gate network
            self.lora_router = nn.ModuleDict({})
            self.lora_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            self.fan_in_fan_out = fan_in_fan_out
            if fan_in_fan_out:
                self.weight.data = self.weight.data.T

            nn.Linear.reset_parameters(self)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name
        elif self.lora_method == "proglora":
            self.expert_num = self.task_id +1
            nn.Linear.__init__(self, in_features, out_features, **kwargs)
            CoINMOELoraLayer.__init__(self, in_features=in_features, 
                                out_features=out_features, 
                                expert_num=self.expert_num,
                                task_id=self.task_id,
                                lora_method=self.lora_method,
                                trans_hidden_dim=self.trans_hidden_dim)
            self.weight.requires_grad = False
            self.fan_in_fan_out = fan_in_fan_out
            if fan_in_fan_out:
                self.weight.data = self.weight.data.T

            nn.Linear.reset_parameters(self)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

    def cal_attention(self, prompt_key, text_input, adapter_name, return_logits=False):
        # equation (2)
        avg_inputs_embeds = text_input.max(dim=1, keepdim=True).values # shape = bs, 1, dim
        self.trans_input = self.trans_input.to(text_input.device)
        x = self.trans_input(avg_inputs_embeds).to(text_input.device) # shape = bs, 1, dim
        
        # self.peft_config[adapter_name].attn_temperature == 1
        attn_temperature = math.sqrt(self.model_dim)
        
        attn_scores = prompt_key(x) / attn_temperature # bs, 1, num_expert
        attn_scores = attn_scores.squeeze(1).to(text_input.device) # bs, num_expert
        attn_scores_m = attn_scores - torch.max(attn_scores)
        
        weights = torch.nn.functional.softmax(attn_scores_m, dim=-1) # bs, num_expert

        if not return_logits:
            return weights  
        else:
            return attn_scores_m.to(text_input.device)  # shape (B, L, 1)

    def init_weights_trans_input(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)  # Xavier
            if layer.bias is not None:
                nn.init.zeros_(layer.bias) 
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight) 
            nn.init.zeros_(layer.bias) 


    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data += (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data -= (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.r[self.active_adapter] > 0 and self.merged: # merge the adapter to linear
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and self.lora_method == None:   # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            self.lora_router = self.lora_router.to(x.device)
            router = self.lora_router[self.active_adapter](x)
            router = torch.softmax(router, dim=-1)
            for i in range(self.expert_num):
                result += ( # lora process
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                    )
                    * self.scaling[self.active_adapter]
                    * router[:,:,i].unsqueeze(-1)
                )
        elif self.r[self.active_adapter] > 0 and self.lora_method == "proglora":
            import pdb
            # pdb.set_trace()
            model_args = kwargs.pop("model_args", True)
            attention_all_weight = kwargs.pop("attention_all_weight", True)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            # x_input_attention = self.trans_input(x) # shape = bs, len, dim
            # if self.task_id > 0:
            #     attention_current = kwargs.pop("attention_current", True)
            #     attention_previous = kwargs.pop("attention_previous", True)
            #     attention_all = torch.cat((attention_previous,attention_current),dim=-1).to(x.device) # bs, num_expert
            #     attention_all_weight = torch.nn.functional.softmax(attention_all, dim=-1).to(x.device) # bs, num_expert
            # elif self.task_id == 0:
            #     attention_current = kwargs.pop("attention_current", True)
            #     attention_all_weight = torch.nn.functional.softmax(attention_all, dim=-1).to(x.device)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            # self.lora_router = self.lora_router.to(x.device)
            # router = self.lora_router[self.active_adapter](x)
            # router = torch.softmax(router, dim=-1)
            # attention_all_weight = self.interpolate_router(attention_all_weight, self.router_w, self.router_top)
            for i in range(self.expert_num):
                result += ( # lora process
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                    )
                    * self.scaling[self.active_adapter]
                    * attention_all_weight[:,i].unsqueeze(-1).unsqueeze(-1)
                )
            result = result.to(previous_dtype)

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result
    


class CoINMOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, task_id, lora_method, trans_hidden_dim) -> None:

        super().__init__()

        if lora_method == None:
            self.expert_num = expert_num
            self.in_features, self.out_features = in_features, out_features
            self.loraA = nn.ModuleList([])

            assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
            self.r = self.out_features // self.expert_num
            
            for _ in range(self.expert_num):
                self.loraA.append(CoINMOEExpert(self.in_features, self.r))
        elif lora_method == "proglora":
            self.expert_num = 1 + task_id
            self.in_features, self.out_features = in_features, out_features
            self.loraA = nn.ModuleList([])

            # assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
            self.r = self.out_features
            
            for _ in range(self.expert_num):
                self.loraA.append(CoINMOEExpert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs
    
class CoINMOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, task_id, lora_method, trans_hidden_dim) -> None:

        super().__init__()

        if lora_method == None:
            self.expert_num = expert_num
            self.in_features, self.out_features = in_features, out_features
            self.loraB = nn.ModuleList([])

            assert self.in_features % self.expert_num == 0
            self.r = self.in_features // self.expert_num
            
            for _ in range(self.expert_num):
                self.loraB.append(CoINMOEExpert(self.r, self.out_features))
        elif lora_method == "proglora":
            self.expert_num = 1 + task_id
            self.in_features, self.out_features = in_features, out_features
            self.loraB = nn.ModuleList([])

            # assert self.in_features % self.expert_num == 0
            self.r = self.in_features
            
            for _ in range(self.expert_num):
                self.loraB.append(CoINMOEExpert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs



class CoINMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y



class CoINMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class CoINMOERouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: CoINMOELoraConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits