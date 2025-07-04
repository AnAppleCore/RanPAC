import copy
import logging
import math
import torch
from torch import nn
import timm
from torch.nn import functional as F

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP=False
        self.W_rand=None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
            out = F.linear(inn,self.weight)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


def get_convnet(args, pretrained=False):

    name = args["convnet_type"].lower()
    #Resnet
    if name=="pretrained_resnet50":
        from resnet import resnet50
        model=resnet50(pretrained=True,args=args)
        return model.eval()
    elif name=="pretrained_resnet152":
        from resnet import resnet152
        model=resnet152(pretrained=True,args=args)
        return model.eval()
    elif name=="vit_base_patch32_224_clip_laion2b":
        #note: even though this is "B/32" it has nearly the same num params as the standard ViT-B/16
        model=timm.create_model("vit_base_patch32_224_clip_laion2b", pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    #NCM or NCM w/ Finetune
    elif name=="pretrained_vit_b16_224" or name=="vit_base_patch16_224":
        model=timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    elif name=="pretrained_vit_b16_224_in21k" or name=="vit_base_patch16_224_in21k":
        model=timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    # SSF 
    elif '_ssf' in name:
        if args["model_name"]=="ssf":
            from petl import vision_transformer_ssf #registers vit_base_patch16_224_ssf
            if name=="pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_ssf":
                model=timm.create_model("vit_base_patch16_224_in21k_ssf",pretrained=True, num_classes=0)
                model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"]=="vpt":
            from petl.vpt import build_promptmodel
            if name=="pretrained_vit_b16_224_vpt":
                basicmodelname="vit_base_patch16_224" 
            elif name=="pretrained_vit_b16_224_in21k_vpt":
                basicmodelname="vit_base_patch16_224_in21k"
            
            #print("modelname,",name,"basicmodelname",basicmodelname)
            VPT_type="Deep"
            #if args["vpt_type"]=='shallow':
            #    VPT_type="Shallow"
            Prompt_Token_num=5#args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname,  Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num=64#args["ffn_num"]
        if args["model_name"]=="adapter" :
            from petl import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name=="pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

class ResNetCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

# class SimpleVitNet(BaseNet):
#     def __init__(self, args, pretrained):
#         super().__init__(args, pretrained)

#     def update_fc(self, nb_classes):
#         fc = CosineLinear(self.feature_dim, nb_classes).cuda()
#         if self.fc is not None:
#             nb_output = self.fc.out_features
#             weight = copy.deepcopy(self.fc.weight.data)
#             fc.sigma.data = self.fc.sigma.data
#             weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
#             fc.weight = nn.Parameter(weight)
#         del self.fc
#         self.fc = fc

#     def forward(self, x):
#         x = self.convnet(x)
#         out = self.fc(x)
#         return out
    
class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.class_to_task = {}
        self.task_to_class = {}
        self.task_wise_fc_ready = False
        self.task_wise_fc = nn.ModuleList([])

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def init_task_wise_fc(self, task_id, W_rand: torch.Tensor):
        fc = CosineLinear(self.fc.in_features, self.fc.out_features).cuda()
        fc.W_rand = W_rand.to(device='cuda')
        fc.use_RP = True
        self.task_wise_fc.append(fc)
        self.task_wise_fc[task_id].requires_grad = False
        self.task_wise_fc_ready = True

    def update_map(self, task_id: int, class_ids: list[int]):
        if task_id not in self.task_to_class:
            self.task_to_class[task_id] = []
        for class_id in class_ids:
            self.class_to_task[class_id] = task_id
            self.task_to_class[task_id].append(class_id)

    def forward(self, x, targets=None):
        x = self.convnet(x)
        out = self.fc(x)

        if targets is None:
            return out

        batch_size = out['logits'].shape[0]
        new_logits = out['logits'].clone()
        raw_class_ids = out['logits'].argmax(dim=-1)
        real_class_ids = targets

        pred_task_ids = []
        real_task_ids = []

        for i in range(batch_size):

            raw_class_id = raw_class_ids[i].item()
            real_class_id = real_class_ids[i].item()
            pred_task_id = self.class_to_task[raw_class_id]
            real_task_id = self.class_to_task[real_class_id]

            pred_task_ids.append(pred_task_id)
            real_task_ids.append(real_task_id)

            if self.task_wise_fc_ready and not self.training:
                new_out = self.task_wise_fc[pred_task_id](x[i])
                new_logit = new_out['logits'].view(1, -1)
                new_logits[i, :new_logit.shape[1]] = new_logit

        out.update({'pred_task_ids': torch.tensor(pred_task_ids).cuda(),
                    'real_task_ids': torch.tensor(real_task_ids).cuda(),
                    'raw_logits': out['logits'],
                    'new_logits': new_logits})

        return out

class ExpertMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None):
        super(ExpertMLP, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class MoEBlock(nn.Module):
    def __init__(self, num_experts=8, M=10000, top_k=2):
        super(MoEBlock, self).__init__()
        self.num_experts = num_experts
        self.M = M
        self.top_k = top_k

        self.experts = nn.ModuleList([
            ExpertMLP(M, hidden_dim=512, output_dim=M) 
            for _ in range(num_experts)
        ])

        self.router = nn.Linear(M, num_experts, bias=False)

        for param in self.parameters():
            param.requires_grad = False

    def get_routing_weights(self, features):
        routing_logits = self.router(features) # [B, num_experts]
        topk_logits, topk_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        routing_weights = torch.zeros_like(routing_logits).to(features.device)
        routing_weights.scatter_(-1, topk_indices, F.softmax(topk_logits, dim=-1))
        return routing_weights, topk_indices
    
    def forward(self, x, return_routing_info=False):
        routing_weights, expert_indices = self.get_routing_weights(x)

        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)  # [B, M]
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, M]

        combined_output = torch.sum(
            routing_weights.unsqueeze(-1) * expert_outputs, dim=1
        )  # [B, M]

        # add residual connection
        # combined_output = combined_output + x

        result = {'moe_output': combined_output}

        if return_routing_info:
            result.update({
                'routing_weights': routing_weights,
                'expert_indices': expert_indices
            })
        return result
    
class MoEHead(nn.Module):
    def __init__(self, in_features, out_features, num_experts=8, M=10000, top_k=2, sigma=True):
        super(MoEHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        if num_experts > 0:
            self.moe_block = MoEBlock(num_experts, M, top_k)
        else:
            self.register_parameter('moe_block', None)

        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

        self.use_RP=False
        self.W_rand=None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input, only_pre_logits=False, return_routing_info=False):

        result_dict = {}

        if not self.use_RP:
            result_dict.update({'moe_output': input})
            if only_pre_logits:
                out = input
            else:
                out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn = input
            if self.moe_block is not None:
                result_dict.update(self.moe_block(inn, return_routing_info=return_routing_info))
                inn = result_dict['moe_output']
            else:
                result_dict.update({'moe_output': inn})
            if only_pre_logits:
                out = inn
            else:
                out = F.linear(inn, self.weight)

        if self.sigma is not None:
            out = self.sigma * out

        result_dict.update({'logits': out})

        return result_dict

class MoEViTNet(BaseNet):
    """
    MoE-ViT: naive mean integration of multiple views
    """
    def __init__(self, args, pretrained, num_views=5):
        super().__init__(args, pretrained)

        self.num_views = num_views
        self.fc = nn.ModuleList([None] * num_views)

    def update_fc(self, nb_classes):

        for i in range(self.num_views):
            fc = CosineLinear(self.feature_dim, nb_classes).cuda()
            if self.fc[i] is not None:
                nb_output = self.fc[i].out_features
                weight = copy.deepcopy(self.fc[i].weight.data)
                fc.sigma.data = self.fc[i].sigma.data
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
                fc.weight = nn.Parameter(weight)
            self.fc[i] = fc

    def forward(self, x, is_train=False):
        x = self.convnet(x)

        if is_train:
            out = self.fc[0](x)
        else:
            out = []
            for i in range(self.num_views):
                out.append(self.fc[i](x)["logits"])
            out = torch.stack(out, dim=1)
            out = {"logits": out.mean(dim=1)}
        return out
