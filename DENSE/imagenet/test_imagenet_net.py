import logging
import random
import torch
import torch.optim as optim
import torch.nn as nn
from robustbench.data import load_imagenetc
from robustbench.data import load_cifar100
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
# from robustbench.utils import clean_accuracy_vit as accuracy
import torchvision.transforms
import my_transforms as my_transforms
import numpy as np
import os
from copy import deepcopy
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch
from robustbench.model_zoo.architectures import vision_transformer
from robustbench.model_zoo.architectures import prompt
from timm.models import create_model
from torch.nn import ModuleList
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import torchvision.transforms as transforms
from sklearn.cluster import MiniBatchKMeans




def mixup(batch_x, batch_y, alpha=0.5, use_cuda=False):

    if alpha > 0:

        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        # print(lam * batch_y + (1 - lam) * batch_y[index, :])
    else:
        index = torch.randperm(batch_size)


    return lam * batch_x + (1 - lam) * batch_x[index, :] ,batch_y,batch_y[index],lam


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]
    clip_min, clip_max = 0.0, 1.0
    p_hflip = 0.5
    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        # transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

def get_common_transform():
    transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.ToTensor()
    ])
    return transform

def return_high_confidence_sample(P_L):
    sample_list =[]
    sample_values = []
    mid_sample_list = []
    mid_sample_values = []
    low_sample_list = []
    P_L_S = torch.nn.Softmax(dim=1)(P_L)
    for instance in range(50):
        if float(P_L_S[instance].max(0).values)>= 0.9  and float(sorted(P_L_S[instance])[-2])<=0.3:
            sample_list.append(int(instance))
            sample_values.append(float(P_L_S[instance].max(0).values))
    if len(mid_sample_values) ==0:
        mid_value = 0
    else:
        mid_value = sum(mid_sample_values)/len(mid_sample_values)
    # high_value = sum(sample_values)/len(sample_values)

    # print(len(low_sample_list))
    return torch.tensor(sample_list).type(torch.int64),torch.tensor(low_sample_list).type(torch.int64),mid_sample_list,mid_value
# Fix random seed for reproducibility
def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)
		    torch.cuda.manual_seed_all(seed)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
def initialize_weights_to_ones(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, 1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

def main():
    transforms_cotta = get_tta_transforms()
    original_model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        drop_block_rate=None,
        img_size=224,
        prompt_pool =False,
        pool_size =None,
        prompt_length = None,
    )

    model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        drop_block_rate=None,
        img_size=224,
        prompt_pool =False,
        pool_size =None,
        prompt_length = None,
    )


    class MOE_Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Sequential(nn.Linear(768, 10),
            nn.Linear(10, 4))
        def build_expert(self):
            expert_build = nn.Sequential([nn.Linear(768, 10),
            nn.Linear(10, 768)])
            return expert_build
        def build_router(self):
            router_build = nn.Linear(768, 4)
            return router_build
        def initialize_weights_to_ones(self,model):
            for layer in model.modules():
                if isinstance(layer, nn.Linear):  # 针对线性层
                    nn.init.constant_(layer.weight, 0.1)  # 将权重初始化为1
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)  # 将偏置初始化为0
        def forward(self, x,fes):
            # self.initialize_weights_to_ones(self.router)
            print(fes)
            weight = self.router(fes)
            weight = nn.Softmax(dim=1)(weight)
            weight_top, weight_indices = torch.topk(weight, 2)

            # x =self.experts[weight_indices[0][0]](x)*float(weight_top[0][0]) + self.experts[weight_indices[0][0]](x)*float(weight_top[0][0])
            return weight_top,weight_indices

    class MOE_Net(nn.Module):
        def __init__(self,base_model):
            super().__init__()
            self.base_model = base_model
            self.moe_blocks = nn.ModuleList([ MOE_Block() for _ in range(12)])
            self.expert_blocks = nn.ModuleList([ base_model.blocks for _ in range(4)])
        def forward(self, x,fes):
            x = self.base_model.patch_embed(x)
            if self.base_model.cls_token is not None:
                x = torch.cat((self.base_model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # cls token拼接在x前
            x = self.base_model.pos_drop(x + self.base_model.pos_embed)
            for i in range(1):
                result = []
                x_1 = self.expert_blocks[0][i](x)
                result.append(x_1)
                x_2 = self.expert_blocks[1][i](x)
                result.append(x_2)
                x_3 = self.expert_blocks[2][i](x)
                result.append(x_3)
                x_4 = self.expert_blocks[3][i](x)
                result.append(x_4)
                x_weight_top,x_weight_indices =self.moe_blocks[i](x,fes)
                # x_weight_top = nn.Softmax(dim=1)(x_weight_top)
                print(x_weight_top[0], x_weight_indices[0])
                x =result[x_weight_indices[0][0]]*float(x_weight_top[0][0]) + result[x_weight_indices[0][1]]*float(x_weight_top[0][1])
            for i in range(1,12):
                x = self.base_model.blocks[i](x)
            x = self.base_model.norm(x)
            x = x[:, 0]
            x = self.base_model.fc_norm(x)
            x =self.base_model.head(x)
            return x




    same_seeds(0)
    # freeze args.freeze[blocks, patch_embed, cls_token] parameters

    model_pool = nn.ModuleList([deepcopy(model) for _ in range(4)])
    for n, p in model_pool.named_parameters():
        if n.startswith(tuple(['0.blocks.0','0.blocks.1.','0.blocks.2.','0.blocks.3','0.blocks.4',
                               '1.blocks.0','1.blocks.1.','1.blocks.2.','1.blocks.3','1.blocks.4',
                               '2.blocks.0','2.blocks.1.','2.blocks.2.','2.blocks.3','2.blocks.4',
                               '3.blocks.0','3.blocks.1.','3.blocks.2.','3.blocks.3','3.blocks.4',
                               ])):
            p.requires_grad = True
            print(n)
        else:
            p.requires_grad =False

    original_model = original_model.cuda(1)
    model_pool = model_pool.cuda(1)

    corruptions = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow',
                    'frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression']
    severitys = [5]
    acc_list=[]
    for corruption in corruptions:
        for severity in severitys:
            x_test, y_test = load_imagenetc(n_examples=5000, corruptions=[corruption], severity=severity ,shuffle=False)
            acc =0
            for i in range(100):
                x_test_batch =x_test[i*50:(i+1)*50,]
                y_test_batch = y_test[i*50:(i+1)*50,]
                x_test_batch,y_test_batch = x_test_batch.cuda(1),y_test_batch.cuda(1)

                with torch.no_grad():
                    result = original_model(x_test_batch)
                    result_pre = result['pre_logits']
                    result_pre = result_pre.mean(0)
                router = nn.Sequential(nn.Linear(768, 1024),
                                       nn.ReLU(),
                                       nn.Linear(1024, 4),
                                       )
                router = router.cuda(1)
                y = router(result_pre)
                y = nn.Softmax(dim=0)(y)

                weight_top, weight_indices = torch.topk(y, 2)
                x_1 = model_pool[weight_indices[0]](x_test_batch)['logits']
                x_1_soft = nn.Softmax(dim=1)(x_1)
                x_2 = model_pool[weight_indices[1]](x_test_batch)['logits']
                x_2_soft = nn.Softmax(dim=1)(x_2)
                x_source = nn.Softmax(dim=1)(result['logits'])

                x_sum = x_1*weight_top[0]+x_2*weight_top[1]
                x_sum_out = nn.Softmax(dim=1)(x_sum)

                one_hot = torch.tensor(np.eye(1000)[np.array((x_sum_out.max(1)[1]).cpu())], device='cuda:1')

                # Screening for high confidence samples
                sample_list_aug, low_sample_list_aug ,mid_sample_list,mid_value= return_high_confidence_sample(x_sum)

                optimizer = optim.Adam([
                    {'params': [
                                model_pool[weight_indices[0]].blocks[1].norm1.weight,model_pool[weight_indices[0]].blocks[1].norm1.bias,model_pool[weight_indices[0]].blocks[1].attn.qkv.weight,model_pool[weight_indices[0]].blocks[1].attn.qkv.bias,model_pool[weight_indices[0]].blocks[1].attn.proj.weight,model_pool[weight_indices[0]].blocks[1].attn.proj.bias,model_pool[weight_indices[0]].blocks[1].norm2.weight,model_pool[weight_indices[0]].blocks[1].norm2.bias,model_pool[weight_indices[0]].blocks[1].mlp.fc1.weight,model_pool[weight_indices[0]].blocks[1].mlp.fc1.bias,model_pool[weight_indices[0]].blocks[1].mlp.fc2.weight,model_pool[weight_indices[0]].blocks[1].mlp.fc2.bias,
                        model_pool[weight_indices[1]].blocks[1].norm1.weight,
                        model_pool[weight_indices[1]].blocks[1].norm1.bias,
                        model_pool[weight_indices[1]].blocks[1].attn.qkv.weight,
                        model_pool[weight_indices[1]].blocks[1].attn.qkv.bias,
                        model_pool[weight_indices[1]].blocks[1].attn.proj.weight,
                        model_pool[weight_indices[1]].blocks[1].attn.proj.bias,
                        model_pool[weight_indices[1]].blocks[1].norm2.weight,
                        model_pool[weight_indices[1]].blocks[1].norm2.bias,
                        model_pool[weight_indices[1]].blocks[1].mlp.fc1.weight,
                        model_pool[weight_indices[1]].blocks[1].mlp.fc1.bias,
                        model_pool[weight_indices[1]].blocks[1].mlp.fc2.weight,
                        model_pool[weight_indices[1]].blocks[1].mlp.fc2.bias,
                        model_pool[weight_indices[0]].blocks[0].norm1.weight,
                        model_pool[weight_indices[0]].blocks[0].norm1.bias,
                        model_pool[weight_indices[0]].blocks[0].attn.qkv.weight,
                        model_pool[weight_indices[0]].blocks[0].attn.qkv.bias,
                        model_pool[weight_indices[0]].blocks[0].attn.proj.weight,
                        model_pool[weight_indices[0]].blocks[0].attn.proj.bias,
                        model_pool[weight_indices[0]].blocks[0].norm2.weight,
                        model_pool[weight_indices[0]].blocks[0].norm2.bias,
                        model_pool[weight_indices[0]].blocks[0].mlp.fc1.weight,
                        model_pool[weight_indices[0]].blocks[0].mlp.fc1.bias,
                        model_pool[weight_indices[0]].blocks[0].mlp.fc2.weight,
                        model_pool[weight_indices[0]].blocks[0].mlp.fc2.bias,
                        model_pool[weight_indices[1]].blocks[0].norm1.weight,
                        model_pool[weight_indices[1]].blocks[0].norm1.bias,
                        model_pool[weight_indices[1]].blocks[0].attn.qkv.weight,
                        model_pool[weight_indices[1]].blocks[0].attn.qkv.bias,
                        model_pool[weight_indices[1]].blocks[0].attn.proj.weight,
                        model_pool[weight_indices[1]].blocks[0].attn.proj.bias,
                        model_pool[weight_indices[1]].blocks[0].norm2.weight,
                        model_pool[weight_indices[1]].blocks[0].norm2.bias,
                        model_pool[weight_indices[1]].blocks[0].mlp.fc1.weight,
                        model_pool[weight_indices[1]].blocks[0].mlp.fc1.bias,
                        model_pool[weight_indices[1]].blocks[0].mlp.fc2.weight,
                        model_pool[weight_indices[1]].blocks[0].mlp.fc2.bias,
                        router[0].weight,router[0].bias,
                        router[2].weight, router[2].bias,
                        model_pool[weight_indices[0]].blocks[2].norm1.weight,
                        model_pool[weight_indices[0]].blocks[2].norm1.bias,
                        model_pool[weight_indices[0]].blocks[2].attn.qkv.weight,
                        model_pool[weight_indices[0]].blocks[2].attn.qkv.bias,
                        model_pool[weight_indices[0]].blocks[2].attn.proj.weight,
                        model_pool[weight_indices[0]].blocks[2].attn.proj.bias,
                        model_pool[weight_indices[0]].blocks[2].norm2.weight,
                        model_pool[weight_indices[0]].blocks[2].norm2.bias,
                        model_pool[weight_indices[0]].blocks[2].mlp.fc1.weight,
                        model_pool[weight_indices[0]].blocks[2].mlp.fc1.bias,
                        model_pool[weight_indices[0]].blocks[2].mlp.fc2.weight,
                        model_pool[weight_indices[0]].blocks[2].mlp.fc2.bias,
                        model_pool[weight_indices[1]].blocks[2].norm1.weight,
                        model_pool[weight_indices[1]].blocks[2].norm1.bias,
                        model_pool[weight_indices[1]].blocks[2].attn.qkv.weight,
                        model_pool[weight_indices[1]].blocks[2].attn.qkv.bias,
                        model_pool[weight_indices[1]].blocks[2].attn.proj.weight,
                        model_pool[weight_indices[1]].blocks[2].attn.proj.bias,
                        model_pool[weight_indices[1]].blocks[2].norm2.weight,
                        model_pool[weight_indices[1]].blocks[2].norm2.bias,
                        model_pool[weight_indices[1]].blocks[2].mlp.fc1.weight,
                        model_pool[weight_indices[1]].blocks[2].mlp.fc1.bias,
                        model_pool[weight_indices[1]].blocks[2].mlp.fc2.weight,
                        model_pool[weight_indices[1]].blocks[2].mlp.fc2.bias,
                        model_pool[weight_indices[0]].blocks[2].norm1.weight,
                        model_pool[weight_indices[0]].blocks[3].norm1.bias,
                        model_pool[weight_indices[0]].blocks[3].attn.qkv.weight,
                        model_pool[weight_indices[0]].blocks[3].attn.qkv.bias,
                        model_pool[weight_indices[0]].blocks[3].attn.proj.weight,
                        model_pool[weight_indices[0]].blocks[3].attn.proj.bias,
                        model_pool[weight_indices[0]].blocks[3].norm2.weight,
                        model_pool[weight_indices[0]].blocks[3].norm2.bias,
                        model_pool[weight_indices[0]].blocks[3].mlp.fc1.weight,
                        model_pool[weight_indices[0]].blocks[3].mlp.fc1.bias,
                        model_pool[weight_indices[0]].blocks[3].mlp.fc2.weight,
                        model_pool[weight_indices[0]].blocks[3].mlp.fc2.bias,
                        model_pool[weight_indices[1]].blocks[3].norm1.weight,
                        model_pool[weight_indices[1]].blocks[3].norm1.bias,
                        model_pool[weight_indices[1]].blocks[3].attn.qkv.weight,
                        model_pool[weight_indices[1]].blocks[3].attn.qkv.bias,
                        model_pool[weight_indices[1]].blocks[3].attn.proj.weight,
                        model_pool[weight_indices[1]].blocks[3].attn.proj.bias,
                        model_pool[weight_indices[1]].blocks[3].norm2.weight,
                        model_pool[weight_indices[1]].blocks[3].norm2.bias,
                        model_pool[weight_indices[1]].blocks[3].mlp.fc1.weight,
                        model_pool[weight_indices[1]].blocks[3].mlp.fc1.bias,
                        model_pool[weight_indices[1]].blocks[3].mlp.fc2.weight,
                        model_pool[weight_indices[1]].blocks[3].mlp.fc2.bias,
                        model_pool[weight_indices[0]].blocks[4].attn.qkv.weight,
                        model_pool[weight_indices[0]].blocks[4].attn.qkv.bias,
                        model_pool[weight_indices[0]].blocks[4].attn.proj.weight,
                        model_pool[weight_indices[0]].blocks[4].attn.proj.bias,
                        model_pool[weight_indices[0]].blocks[4].norm2.weight,
                        model_pool[weight_indices[0]].blocks[4].norm2.bias,
                        model_pool[weight_indices[0]].blocks[4].mlp.fc1.weight,
                        model_pool[weight_indices[0]].blocks[4].mlp.fc1.bias,
                        model_pool[weight_indices[0]].blocks[4].mlp.fc2.weight,
                        model_pool[weight_indices[0]].blocks[4].mlp.fc2.bias,
                        model_pool[weight_indices[1]].blocks[4].norm1.weight,
                        model_pool[weight_indices[1]].blocks[4].norm1.bias,
                        model_pool[weight_indices[1]].blocks[4].attn.qkv.weight,
                        model_pool[weight_indices[1]].blocks[4].attn.qkv.bias,
                        model_pool[weight_indices[1]].blocks[4].attn.proj.weight,
                        model_pool[weight_indices[1]].blocks[4].attn.proj.bias,
                        model_pool[weight_indices[1]].blocks[4].norm2.weight,
                        model_pool[weight_indices[1]].blocks[4].norm2.bias,
                        model_pool[weight_indices[1]].blocks[4].mlp.fc1.weight,
                        model_pool[weight_indices[1]].blocks[4].mlp.fc1.bias,
                        model_pool[weight_indices[1]].blocks[4].mlp.fc2.weight,
                        model_pool[weight_indices[1]].blocks[4].mlp.fc2.bias,
                                ],'lr': 1e-5},
                    ])
                loss_div = nn.CrossEntropyLoss()((torch.mean(x_sum_out, dim=0)).reshape(1, 1000),
                                               torch.mean(x_sum_out, dim=0).reshape(1, 1000))
                loss_ent = nn.CrossEntropyLoss()(x_sum_out,x_sum_out)
                loss_pl = nn.CrossEntropyLoss()(x_sum_out[sample_list_aug], one_hot[sample_list_aug])
                loss_mse = nn.MSELoss()(x_sum_out,x_source)
                loss_router = 1*nn.CrossEntropyLoss()(y.reshape(-1,4),y.reshape(-1,4))
                loss = loss_ent-loss_div+loss_mse-loss_router+loss_pl
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    logit = model_pool[weight_indices[0]](x_test_batch)['logits'] * float(weight_top[0])+\
                            model_pool[weight_indices[1]](x_test_batch)['logits'] * float(weight_top[1])


                acc += (logit.max(1)[1] == y_test_batch).float().sum()

            acc = acc.item() / x_test.shape[0]
            acc_list.append(acc)
            print('__________________________________________________________' + corruption + str(severity) + '____________________________________________________')
            print((acc))
    print(sum(acc_list)/len(acc_list))



if __name__ == '__main__':
    main()

