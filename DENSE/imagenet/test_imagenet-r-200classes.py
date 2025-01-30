
import os
import random
import torch
import PIL
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.optim as optim
import torchvision.transforms.functional as trnF
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import my_transforms as my_transforms
import torch.nn.functional as F
import numpy as np
from timm.models import create_model
from PIL import Image
from copy import deepcopy
from robustbench.model_zoo.architectures import vision_transformer

def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)
		    torch.cuda.manual_seed_all(seed)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True

def criterion(batch_x, batch_y, alpha=0.1, use_cuda=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        # print(lam)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        # print(lam * batch_y + (1 - lam) * batch_y[index, :])
    else:
        index = torch.randperm(batch_size)
    return lam * batch_x + (1 - lam) * batch_x[index, :] ,batch_y,batch_y[index],lam

def  softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]
    clip_min, clip_max = 0.0, 1.0
    p_hflip = 0.5
    tta_transforms = trn.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        # transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        trn.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        trn.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        trn.CenterCrop(size=n_pixels),
        trn.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

def return_high_confidence_sample(P_L, batchsize = 0):
    sample_list =[]
    low_sample_list = []
    sample_values = []
    P_L_S = torch.nn.Softmax(dim=1)(P_L)
    for instance in range(batchsize):
        if float(P_L_S[instance].max(0).values)>= 0.6 :
            sample_list.append(int(instance))
            sample_values.append(P_L_S[instance].max(0).values)
        else:
            low_sample_list.append(int(instance))
    if len(sample_values) == 0:
        a = 0
    else:
        a =sum(sample_values)/len(sample_values)

    return torch.tensor(sample_list).type(torch.int64),torch.tensor(low_sample_list).type(torch.int64),a


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

name_list = ['imagenet-r_art','imagenet-r_cartoon','imagenet-r_deviantart','imagenet-r_embroidery','imagenet-r_graffiti','imagenet-r_graphic','imagenet-r_misc','imagenet-r_origami','imagenet-r_painting',
             'imagenet-r_sculpture','imagenet-r_sketch','imagenet-r_sticker','imagenet-r_tattoo','imagenet-r_toy','imagenet-r_video game']
original_model = create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=1000,
    drop_rate=0,
    drop_path_rate=0,
    drop_block_rate=None,
    img_size=224,
    prompt_pool=False,
    pool_size=None,
    prompt_length=None,
    # checkpoint_path = '/home/yrzhen/cotta-main/vit_checkpoint/imagenet21k+imagenet2012_ViT-B_16-224.pth'

)
model = create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=1000,
    drop_rate=0,
    drop_path_rate=0,
    drop_block_rate=None,
    img_size=224,
    prompt_pool=False,
    pool_size=None,
    prompt_length=None,
)

model_pool = nn.ModuleList([deepcopy(model) for _ in range(4)])
original_model  =original_model.cuda(1)
model_pool = model_pool.cuda(1)

for p in original_model.parameters():
    p.requires_grad = False

for n, p in model_pool.named_parameters():
    if n.startswith(tuple(['0.blocks.0', '0.blocks.1.', '0.blocks.2.','0.blocks.3.','0.blocks.4.'
                           '1.blocks.0', '1.blocks.1.', '1.blocks.2.','1.blocks.3.','1.blocks.4.',
                           '2.blocks.0', '2.blocks.1.', '2.blocks.2.','2.blocks.3.','2.blocks.4.'
                           ,'3.blocks.0', '3.blocks.1.', '3.blocks.2.','3.blocks.3.','3.blocks.4.',
                           ])):
        p.requires_grad = True
        print(n)
    else:
        p.requires_grad = False
same_seeds(0)
preprocess = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
transforms_cotta = get_tta_transforms()
test_transform = trn.Compose(
    [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

label_r =[1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]
erbai = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
label_r_index =['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']
# print(len(label_r_index),len(erbai),len(label_r))
for name in name_list:
    imagenet_r = dset.ImageFolder(root="/media/shared_space/yrzhen/data/data/{}".format(name), transform=test_transform)
    print("-----------------{}-----------------".format(name))
    label_now = []
    for class_image  in imagenet_r.classes:
    # print(label_r[label_r_index.index(class_image)])
        label_now.append(erbai[label_r_index.index(class_image)])           #当前类别的数据集映射到label now中
        # print(label_r_index.index(class_image))
    # print(label_now)
    # label_un = []
    # # for i  in range(200):
    # #     if i not in label_now:
    # #         label_un.append(i)
    # # print(label_un)

    imagenet_r_loader = torch.utils.data.DataLoader(imagenet_r, batch_size=50, shuffle=True,)
    # print(len(imagenet_r_loader.dataset))
    acc = 0

    for data,target in imagenet_r_loader:
        # print(target)
        # for i in target:
        #     for j in label_un:
        #         if i>j:
        #             i+=1
        #         else:
        #             continue
        # print(target)
        labels_trans = []               #label transform
        for i in target:
            label_trans = label_now[i]
            labels_trans.append(label_trans)
        labels_trans = torch.Tensor(labels_trans)
        data,labels_trans = data.cuda(1),labels_trans.cuda(1)
        # print(labels_trans)
        to_pil =trn.ToPILImage()
        with torch.no_grad():
            result = original_model(data)
            result = result['pre_logits']
            result = result.mean(0)
            # print(result.shape)
        # y = moe_net(x_test_batch,result)
        router = nn.Sequential(nn.Linear(768, 1024),
                               nn.ReLU(),
                               nn.Linear(1024, 4),
                               )

        router = router.cuda(1)
        y = router(result)
        y = nn.Softmax(dim=0)(y)
        weight_top, weight_indices = torch.topk(y, 2)
        x_1 = model_pool[weight_indices[0]](data)['logits'][:,label_r]
        # x_1_soft = nn.Softmax(dim=1)(x_1)
        x_2 = model_pool[weight_indices[1]](data)['logits'][:,label_r]
        # x_2_soft = nn.Softmax(dim=1)(x_2)

        x_sum = x_1 * weight_top[0] + x_2 * weight_top[1]
        x_sum_out = nn.Softmax(dim=1)(x_sum)
        one_hot = torch.tensor(np.eye(200)[np.array((x_sum_out.max(1)[1]).cpu())], device='cuda:1')
        # Screening for high confidence samples
        sample_list_aug, low_sample_list_aug,value = return_high_confidence_sample(x_sum,batchsize=len(data))

        optimizer = optim.Adam([
            {'params': [
                model_pool[weight_indices[0]].blocks[1].norm1.weight,
                model_pool[weight_indices[0]].blocks[1].norm1.bias,
                model_pool[weight_indices[0]].blocks[1].attn.qkv.weight,
                model_pool[weight_indices[0]].blocks[1].attn.qkv.bias,
                model_pool[weight_indices[0]].blocks[1].attn.proj.weight,
                model_pool[weight_indices[0]].blocks[1].attn.proj.bias,
                model_pool[weight_indices[0]].blocks[1].norm2.weight,
                model_pool[weight_indices[0]].blocks[1].norm2.bias,
                model_pool[weight_indices[0]].blocks[1].mlp.fc1.weight,
                model_pool[weight_indices[0]].blocks[1].mlp.fc1.bias,
                model_pool[weight_indices[0]].blocks[1].mlp.fc2.weight,
                model_pool[weight_indices[0]].blocks[1].mlp.fc2.bias,
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
                router[0].weight, router[0].bias,
                # router[1].weight, router[1].bias,
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

                model_pool[weight_indices[0]].blocks[3].norm1.weight,
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

                model_pool[weight_indices[0]].blocks[4].norm1.weight,
                model_pool[weight_indices[0]].blocks[4].norm1.bias,
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

            ], 'lr': 2e-5},
        ])

        loss_3 = nn.CrossEntropyLoss()(x_sum_out[sample_list_aug], one_hot[sample_list_aug])
        loss_2 = nn.CrossEntropyLoss()(torch.mean(x_sum_out,dim=0).reshape(1,-1),torch.mean(x_sum_out,dim=0).reshape(1,-1) )
        loss_r = 1 * nn.CrossEntropyLoss()(y.reshape(-1, 4), y.reshape(-1, 4))
        loss_ent = nn.CrossEntropyLoss()(x_sum_out,x_sum_out)
        loss = loss_3-loss_r-loss_2
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            # logit = original_model(data)['logits'][:,label_r]
            logit = model_pool[weight_indices[0]](data)['logits'][:,label_r] * float(weight_top[0])+\
                    model_pool[weight_indices[1]](data)['logits'][:,label_r] * float(weight_top[1])
        if name =='imagenet-r_tattoo':
            print(logit.max(1)[1])
        acc += (logit.max(1)[1] == labels_trans).float().sum()



    print(acc/len(imagenet_r_loader.dataset))

