from .clip_models import CLIPModel, CLIPModelShuffleAttentionPenultimateLayer
from .imagenet_models import ImagenetModel
from .moco_models import MOCOModel_v3, MOCOModel_v3_ShuffleAttention, MOCOModel_v1, MOCOModel_v1_PatchesAttention, MOCOModel_v1_ShuffleAttention


VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',

    'MOCO_v3:ViT',
    'MOCO_v3:ViT_shuffle_attention',
    'MOCO_v1:RN50',
]





def get_model(opt):
    name = opt.arch
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:]) 
    elif name.startswith("CLIP:"):
        if opt.head_type == "fc":
            if opt.penultimate_feature:
                return CLIPModelPenultimateLayer(name[5:])
            else:
                return CLIPModel(name[5:])
        elif opt.head_type == "attention":
            # penultimate feature
            if opt.shuffle:                
                return CLIPModelShuffleAttentionPenultimateLayer(name[5:], shuffle_times=opt.shuffle_times, original_times=opt.original_times, patch_size=opt.patch_size)

    elif name.startswith("MOCO_v1:"):
        if opt.head_type == "fc":
            return MOCOModel_v1()
        elif opt.head_type == "attention":
            if opt.patch_base:
                return MOCOModel_v1_PatchesAttention()
            else:
                return MOCOModel_v1_ShuffleAttention()


    elif name.startswith("MOCO_v3:"):
        if opt.head_type == "attention":
            return MOCOModel_v3_ShuffleAttention(opt)
        else:
            return MOCOModel_v3(opt)
    else:
        assert False 
