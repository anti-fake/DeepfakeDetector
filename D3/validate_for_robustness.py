import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from glob import glob
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
import argparse
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer
import pandas as pd
from calculate_global_ap import cal_global_ap


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}





def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    if np.isnan(r_acc): r_acc = 0
    if np.isnan(f_acc): f_acc = 0
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, find_thres=False, data_augment=None, threshold=0.5):
    drop_out_rate = 0.0
    shuffle_rate = 0.0
    if hasattr(model, 'drop_out_rate') and model.drop_out_rate != 0.0:
        drop_out_rate = model.drop_out_rate
        model.drop_out_rate = 0.0
    if hasattr(model, 'shuffle_rate') and model.shuffle_rate != 0.0:
        shuffle_rate = model.shuffle_rate
        model.shuffle_rate = 0.0

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Number of batches: %d" %(len(loader)))
        print ("Total number of images: %d" %(len(loader.dataset)))
        for img, label in loader:
            in_tens = img.cuda()
            if data_augment != None:
                in_tens = data_augment(in_tens)
            y_pred_temp = model(in_tens)
            if y_pred_temp.shape[-1] == 2:
                # anomaly
                # y_pred_temp = torch.absolute(y_pred_temp[:, 1] - y_pred_temp[:, 0])
                y_pred_temp = y_pred_temp[:, 0]
            y_pred.extend(y_pred_temp.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    if hasattr(model, 'drop_out_rate'):
        model.drop_out_rate = drop_out_rate
    if hasattr(model, 'shuffle_rate'):
        model.shuffle_rate = shuffle_rate
        
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true



def decoupled_validate(model, loader, find_thres=False, data_augment=None):
    
    with torch.no_grad():
        y_true, y_orig_pred, y_shuffle_pred = [], [], []
        print ("Number of batches: %d" %(len(loader)))
        print ("Total number of images: %d" %(len(loader.dataset)))
        for img, label in loader:
            in_tens = img.cuda()
            if data_augment != None:
                in_tens = data_augment(in_tens)
            model.set_input((in_tens, label.cuda()))
            model.decoupled_input_forward()
            model_output = model.output
            orig_output = model_output[:, 1]
            shuffle_output = model_output[:, 2]
            y_orig_pred.extend(orig_output.sigmoid().flatten().tolist())
            y_shuffle_pred.extend(shuffle_output.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_orig_pred, y_shuffle_pred = np.array(y_true), np.array(y_orig_pred), np.array(y_shuffle_pred)
    avg_pred = np.array([(y_orig_pred[i]+y_shuffle_pred[i])/2 for i in range(len(y_orig_pred))])
    pred = [y_orig_pred, y_shuffle_pred, avg_pred]

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = [average_precision_score(y_true, p) for p in pred]


    # Acc based on 0.5
    acc = [calculate_acc(y_true, p, 0.5)[2] for p in pred]
    # r_acc0, f_acc0, orig_acc0 = calculate_acc(y_true, y_orig_pred, 0.5)
    # r_acc0, f_acc0, orig_acc0 = calculate_acc(y_true, y_orig_pred, 0.5)
    # r_acc0, f_acc0, orig_acc0 = calculate_acc(y_true, y_orig_pred, 0.5)
    if not find_thres:
        return ap, acc


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_orig_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_orig_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

    
    



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 




def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "PNG"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list





class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        is_norm=True):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        if is_norm:
            self.transform = transforms.Compose([
                # transforms.CenterCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.CenterCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])


    def read_path(self, real_path, fake_path, max_sample):

        real_list = get_list(real_path, must_contain='')
        fake_list = get_list(fake_path, must_contain='')



        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]
        else:
            max_sample = min(len(fake_list), len(real_list))
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]
        assert len(real_list) == len(fake_list)  
        print(f"Found {len(real_list)} real images and {len(fake_list)} fake images")

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label





if __name__ == '__main__':

    # --- ADDED
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='UFD', type=str, required=True, help="Is the dataset from UFD or GenImage")
    parser.add_argument("--gen_model", type=str, required=True, help="Generative model name (e.g., stylagn) as equal to the name of the folder in the dataset")
    args = parser.parse_args()

    # --- END ADDED

    print(f"!!! Using Dataset & generative model: {args.dataset} & {args.gen_model} !!!")

    # basic configuration
    set_seed(418)
    # TODO: 
    result_folder = "./D3_results/"
    checkpoint = "ckpt/classifier.pth"
    # TODO: Your test/evaluation image folder containing "real" and "fake" subfolders
    # test_folders = ["/root/to/validation/folder1", "/root/to/validation/folder2"]
    if args.dataset == 'UFD':
        test_folders = glob(f'/data/dataset/UFD/test/{args.gen_model}/')
    elif  args.dataset == 'GenImage':
        test_folders = glob(f'/data/dataset/GenImage/{args.gen_model}/val/')
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    batch_size = 128
    jpeg_quality = None # jpeg compression, none means no compression
    gaussian_sigma = None # gaussian blurring, none means no blurring
    exp_name = f"validation"

    max_sample = None 

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create and load the detector
    granularity = 14
    model = CLIPModelShuffleAttentionPenultimateLayer("ViT-L/14", shuffle_times=1, original_times=1, patch_size=[granularity])
    is_norm = True
    print(f"using checkpoint {checkpoint}")
    state_dict = torch.load(checkpoint, map_location='cpu')
    model.attention_head.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda()

    
    results_dict = {}
    arch = "clip"

    # data to record
    acc_list = []
    ap_list = []
    b_acc_list = []
    threshold_list = []
    y_pred_list = []
    y_true_list = []
    for path in test_folders:
        dataset = RealFakeDataset(  os.path.join(path, "0_real"), 
                                    os.path.join(path, "1_fake"), 
                                    max_sample, 
                                    arch,
                                    jpeg_quality=jpeg_quality, 
                                    gaussian_sigma=gaussian_sigma,
                                    is_norm=is_norm
                                    )

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true = validate(model, loader, find_thres=True)
        print(acc0)

        acc_list.append(acc0)
        ap_list.append(ap)
        b_acc_list.append(acc1)
        threshold_list.append(best_thres)
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

    acc_list.append(sum(acc_list)/len(acc_list))
    ap_list.append(sum(ap_list)/len(ap_list))
    b_acc_list.append(sum(b_acc_list)/len(b_acc_list))
    threshold_list.append(sum(threshold_list)/len(threshold_list))

    results_dict[f'{gaussian_sigma}_{jpeg_quality}_ap'] = ap_list
    results_dict[f'{gaussian_sigma}_{jpeg_quality}_acc'] = acc_list
    results_dict[f'{gaussian_sigma}_{jpeg_quality}_b_acc'] = b_acc_list
    results_dict[f'{gaussian_sigma}_{jpeg_quality}_b_threshold'] = threshold_list
    results_df = pd.DataFrame(results_dict)
    results_df.to_excel(os.path.join(result_folder, f'{exp_name}.xlsx'), sheet_name='sheet1', index=False)

    np.savez(os.path.join(result_folder, f'{gaussian_sigma}_{jpeg_quality}_ypred.npz'), *y_pred_list)
    np.savez(os.path.join(result_folder, f'{gaussian_sigma}_{jpeg_quality}_ytrue.npz'), *y_true_list)
    
    # calculating global ap
    combined_list = [[y_p, y_t] for y_p, y_t in zip(y_pred_list, y_true_list)]

    cal_global_ap(combined_list)
