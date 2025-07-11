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

class MoEViTNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.class_to_task = {}
        self.task_to_class = {}


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

    def _add_task_info(self, out, targets):
        batch_size = out['logits'].shape[0]
        raw_class_ids = out['logits'].argmax(dim=-1)
        real_class_ids = targets

        pred_task_ids = []
        real_task_ids = []

        for i in range(batch_size):
            raw_class_id = raw_class_ids[i].item()
            real_class_id = real_class_ids[i].item()
            
            pred_task_id = self.class_to_task.get(raw_class_id, 0)
            real_task_id = self.class_to_task.get(real_class_id, 0)

            pred_task_ids.append(pred_task_id)
            real_task_ids.append(real_task_id)

        out.update({
            'pred_task_ids': torch.tensor(pred_task_ids).cuda(),
            'real_task_ids': torch.tensor(real_task_ids).cuda(),
        })

        return out