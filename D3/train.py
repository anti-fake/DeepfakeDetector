import os
import numpy as np
import pandas as pd
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
import torch


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt



if __name__ == '__main__':
    seed = 418
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    # TODO: Your test/evaluation image folder containing "real" and "fake" subfolders
    val_data_root = ["folder1/containing/evaluation/images/from/generator1", "folder2/containing/evaluation/images/from/generator2"]

    # TODO: Your training data folders
    real_folders = ["folder1/containing/real/images", "folder2/containing/real/images"]
    fake_folders = ["folder1/containing/fake/images", "folder2/containing/fake/images"]
    data_loader = create_dataloader(opt, real_folders, fake_folders)

    # initialize detector
    model = Trainer(opt)

    # initialize val datasets 
    val_loader_list = [] # a list of data loader
    for root in val_data_root:
        # initialize training datasets
        real_folders = [os.path.join(root, "real")]
        fake_folders = [os.path.join(root, "fake")]
        val_loader_list.append(create_dataloader(val_opt, real_folders, fake_folders))

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    results_dict = {}
    for epoch in range(opt.niter):
        for i, data in enumerate(tqdm(data_loader)):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        model.eval()
        acc_list = []
        ap_list = []
        b_acc_list = []
        threshold_list = []
        y_pred_list = []
        y_true_list = []
        for i, val_loader in enumerate(val_loader_list):
            ap, r_acc0, f_acc0, acc, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true = validate(model.model, val_loader, find_thres=True)
            acc_list.append(acc)
            ap_list.append(ap)
            b_acc_list.append(acc1)
            threshold_list.append(best_thres)
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print(f"(Val on {val_data_root[i]} @ epoch {epoch}) acc: {acc}; ap: {ap};r_acc0:{r_acc0}, f_acc0:{f_acc0}, r_acc1:{r_acc1}, f_acc1:{f_acc1}, acc1:{acc1}, best_thres:{best_thres}")
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)


        ap_list.append(sum(ap_list) / len(val_loader_list))
        acc_list.append(sum(acc_list) / len(val_loader_list))
        b_acc_list.append(sum(b_acc_list) / len(val_loader_list))
        threshold_list.append(sum(threshold_list) / len(val_data_root))
        results_dict[f'epoch_{epoch}_ap'] = ap_list
        results_dict[f'epoch_{epoch}_acc'] = acc_list
        results_dict[f'epoch_{epoch}_b_acc'] = b_acc_list
        results_dict[f'epoch_{epoch}_b_threshold'] = threshold_list
        results_df = pd.DataFrame(results_dict)
        results_df.to_excel(os.path.join(opt.checkpoints_dir, opt.name, 'results.xlsx'), sheet_name='sheet1', index=False)
        print(f"(average Val on all dataset @ epoch {epoch}) acc: {acc_list[-1]}; ap: {ap_list[-1]}")
        np.savez(os.path.join(opt.checkpoints_dir, opt.name, f'y_pred_eval_{epoch}.npz'), *y_pred_list)
        np.savez(os.path.join(opt.checkpoints_dir, opt.name, f'y_true_eval_{epoch}.npz'), *y_true_list)
        
        # early stop using avg acc
        acc = acc_list[-1]
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()

