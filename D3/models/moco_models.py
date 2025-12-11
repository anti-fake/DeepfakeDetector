import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os
from models import vits
from models.transformer_attention import TransformerAttention
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import torch.nn.functional as F
import models.moco_v1.builder as builder
import torchvision.models as models


CHANNELS = {
    "MOCO" : 128,
    "ViT-L/14" : 768
}

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}



class MOCOModel_v3(nn.Module):
    def __init__(self, opt, num_classes=1):
        super(MOCOModel_v3, self).__init__()
        self.opt = opt
        model_root = "path to your moco pretrained weight"
        self.model = self.create_model(model_root) # self.preprecess will not be used during training, which is handled in Dataset class 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])
 
    def create_model(self, model_root):
        opt = self.opt
        opt.pretrained = model_root
        dim_in = None
        # create model
        print("=> creating model '{}'".format(opt.arch))
        if opt.arch.lower().find('vit') != -1:
            model = vits.__dict__["vit_base"]()
            linear_keyword = 'head'
        else:
            model = torchvision_models.__dict__[opt.arch]()
            linear_keyword = 'fc'

        # freeze all layers and remove the last fc
        remove_name = None
        for name, param in model.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
        model.head = nn.Linear(model.head.weight.shape[-1], 1)
        self.fc = model.head

        # init the fc layer
        getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
        getattr(model, linear_keyword).bias.data.zero_()

        # load from pre-trained, before DistributedDataParallel constructor
        if opt.pretrained:
            if os.path.isfile(opt.pretrained):
                print("=> loading checkpoint '{}'".format(opt.pretrained))
                checkpoint = torch.load(opt.pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                opt.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

                print("=> loaded pre-trained model '{}'".format(opt.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pretrained))



        return model


    def forward(self, x, return_feature=False):
        features = self.model(x) 
        return features



class MOCOModel_v3_ShuffleAttention(nn.Module):
    def __init__(self, opt, num_classes=1, shuffle_times=3):
        self.shuffle_times = shuffle_times
        super(MOCOModel_v3_ShuffleAttention, self).__init__()
        self.opt = opt
        self.attention_head = TransformerAttention(768, shuffle_times, 1)
        model_root = "path to your moco pretrained weight"
        self.model = self.create_model(model_root) # self.preprecess will not be used during training, which is handled in Dataset class 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])
 
    def create_model(self, model_root):
        opt = self.opt
        opt.pretrained = model_root
        dim_in = None
        # create model
        print("=> creating model '{}'".format(opt.arch))
        if opt.arch.lower().find('vit') != -1:
            model = vits.__dict__["vit_base"]()
            linear_keyword = 'head'
        else:
            model = torchvision_models.__dict__[opt.arch]()
            linear_keyword = 'fc'

        # init the fc layer
        # getattr(model, linear_keyword).weight.data.one_()
        # getattr(model, linear_keyword).bias.data.zero_()
        for name, param in model.named_parameters():
            # if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False


        # load from pre-trained, before DistributedDataParallel constructor
        if opt.pretrained:
            if os.path.isfile(opt.pretrained):
                print("=> loading checkpoint '{}'".format(opt.pretrained))
                checkpoint = torch.load(opt.pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                opt.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

                print("=> loaded pre-trained model '{}'".format(opt.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pretrained))

        # remove last layer
        model.head = nn.Identity()


        return model

    def shuffle_patches(self, x, patch_size):
        B, C, H, W = x.size()
        # Unfold the input tensor to extract non-overlapping patches
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size, dilation=1)
        # Reshape the patches to (B, C, patch_H, patch_W, num_patches)
        shuffled_patches = patches[:, :, torch.randperm(patches.size(-1))]
        # Fold the shuffled patches back into images
        shuffled_images = F.fold(shuffled_patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return shuffled_images


    def forward(self, x, return_feature=False):
        features = []
        for i in range(self.shuffle_times):
            features.append(self.model(self.shuffle_patches(x, patch_size=32)))
        features = self.attention_head(torch.stack(features, dim=-2))
        return features





class MOCOModel_v1(nn.Module):
    def __init__(self, num_classes=1):
        super(MOCOModel_v1, self).__init__()
        model_root = "path to your moco pretrained weight"
        self.model = self.create_model(model_root) # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS["MOCO"], num_classes)


    def create_model(self, model_root):
        print("=> creating model '{}'".format("resnet50"))
        model = builder.MoCo(
            models.__dict__["resnet50"],
            128,
            65536,
            0.999,
            0.07,
            False,
        )

        model = model.encoder_q
        pretrained_dict = torch.load(model_root, map_location="cpu")["state_dict"]
        model_dict = model.state_dict()

        # Create a new dictionary that maps the keys in the pretrained model to the keys in the built model
        new_dict = {}
        for k, v in pretrained_dict.items():
            if 'module.encoder_q' in k:
                k = k.replace('module.encoder_q.', '')
            new_dict[k] = v

        # Load the pretrained weights into the built model
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)


        return model


    def forward(self, x, return_feature=False):
        features = self.model(x) 
        if return_feature:
            return features
        return self.fc(features)





class MOCOModel_v1_ShuffleAttention(nn.Module):
    def __init__(self, num_classes=1,shuffle_times=3):
        super(MOCOModel_v1_ShuffleAttention, self).__init__()
        model_root = "path to your moco pretrained weight"
        self.model = self.create_model(model_root) # self.preprecess will not be used during training, which is handled in Dataset class 
        self.shuffle_times = shuffle_times
        self.attention_head = TransformerAttention(CHANNELS["MOCO"], shuffle_times + 1, 1)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def create_model(self, model_root):
        print("=> creating model '{}'".format("resnet50"))
        model = builder.MoCo(
            models.__dict__["resnet50"],
            128,
            65536,
            0.999,
            0.07,
            False,
        )

        model = model.encoder_q
        pretrained_dict = torch.load(model_root, map_location="cpu")["state_dict"]
        model_dict = model.state_dict()

        # Create a new dictionary that maps the keys in the pretrained model to the keys in the built model
        new_dict = {}
        for k, v in pretrained_dict.items():
            if 'module.encoder_q' in k:
                k = k.replace('module.encoder_q.', '')
            new_dict[k] = v

        # Load the pretrained weights into the built model
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)


        return model


    def shuffle_patches(self, x, patch_size):
        B, C, H, W = x.size()
        # Unfold the input tensor to extract non-overlapping patches
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size, dilation=1)
        # Reshape the patches to (B, C, patch_H, patch_W, num_patches)
        shuffled_patches = patches[:, :, torch.randperm(patches.size(-1))]
        # Fold the shuffled patches back into images
        shuffled_images = F.fold(shuffled_patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return shuffled_images


    def forward(self, x, return_feature=False):
        features = []
        for i in range(self.shuffle_times):
            features.append(self.model(self.shuffle_patches(x, patch_size=32)))
        features.append(self.model(x))
        features = self.attention_head(torch.stack(features, dim=-2))
        return features



class MOCOModel_v1_PatchesAttention(nn.Module):
    def __init__(self, num_classes=1, divide_nums=3):
        super(MOCOModel_v1_PatchesAttention, self).__init__()
        self.divide_nums = divide_nums
        model_root = "path to your moco pretrained weight"
        self.model = self.create_model(model_root) # self.preprecess will not be used during training, which is handled in Dataset class 
        self.normal = transforms.Normalize( mean=MEAN["imagenet"], std=STD["imagenet"] )
        self.attention_head = TransformerAttention(CHANNELS["MOCO"], divide_nums**2 + 1, 1)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def create_model(self, model_root):
        print("=> creating model '{}'".format("resnet50"))
        model = builder.MoCo(
            models.__dict__["resnet50"],
            128,
            65536,
            0.999,
            0.07,
            False,
        )

        model = model.encoder_q
        pretrained_dict = torch.load(model_root, map_location="cpu")["state_dict"]
        model_dict = model.state_dict()

        # Create a new dictionary that maps the keys in the pretrained model to the keys in the built model
        new_dict = {}
        for k, v in pretrained_dict.items():
            if 'module.encoder_q' in k:
                k = k.replace('module.encoder_q.', '')
            new_dict[k] = v

        # Load the pretrained weights into the built model
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)


        return model

    def divide(self, x):
        patch_size = 224 // self.divide_nums
        patches = []
        for i in range(self.divide_nums):
            for j in range(self.divide_nums):
                patches.append(transforms.Resize([224, 224], antialias=True)(x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]))
        return patches


    def forward(self, x, return_feature=False):
        features = []
        patches = self.divide(x)
        for patch in patches:
            features.append(self.model(self.normal(patch)))
        features.append(self.model(self.normal(x)))
        features = self.attention_head(torch.stack(features, dim=-2))
        return features
    
