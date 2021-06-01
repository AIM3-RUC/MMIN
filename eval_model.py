import os, sys
import time
import json
import numpy as np
from tqdm import tqdm
from data import create_dataset, create_trn_val_tst_dataset
from models.utils.config import OptConfig
from models import create_model
from utils.logger import get_logger
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, save_dir, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    # A_total_feats = []
    # V_total_feats = []
    # L_total_feats = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        # A_feats = model.feat_A.detach().cpu().numpy()
        # A_total_feats.append(A_feats)
        # V_feats = model.feat_V.detach().cpu().numpy()
        # V_total_feats.append(V_feats)
        # L_feats = model.feat_L.detach().cpu().numpy()
        # L_total_feats.append(L_feats)
        total_pred.append(pred)
        total_label.append(label)
    
    # calculate metrics
    # A_total_feats = np.concatenate(A_total_feats)
    # V_total_feats = np.concatenate(V_total_feats)
    # L_total_feats = np.concatenate(L_total_feats)
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    model.train()
    
    # save test results
    # np.save(os.path.join(save_dir, 'A_{}_feat.npy'.format(phase)), A_total_feats)
    # np.save(os.path.join(save_dir, 'V_{}_feat.npy'.format(phase)), V_total_feats)
    # np.save(os.path.join(save_dir, 'L_{}_feat.npy'.format(phase)), L_total_feats)
    np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm

def eval_mix(model, val_iter, save_dir, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        miss_type = np.array(data['miss_type'])
        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    
    # save test whole results
    if save_dir != None:
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    # save part results
        for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
            part_index = np.where(total_miss_type == part_name)
            part_pred = total_pred[part_index]
            part_label = total_label[part_index]
            np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
            np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)

    return acc, uar, f1, cm

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt


if __name__ == '__main__':
    
    # teacher_path = 'checkpoints/EF_raw_A_run1'
    # teacher_path = 'checkpoints/multi_fusion_manyCE_run1'
    # teacher_path = 'checkpoints/new_cra_single_AE256,128,64_blocks5_run1'
    teacher_path = 'checkpoints/emobert_test_parameter_V_run1'
    opt_path = os.path.join(teacher_path, 'train_opt.conf')
    opt = load_from_opt_record(opt_path)
    opt.isTrain = False                             # teacher model should be in test mode
    opt.gpu_ids = [4]
    opt.serial_batches = True
    # opt.dataset_mode = "iemocap_miss"
    # setattr(opt, 'miss_num', 'mix')
    opt.dataset_mode = 'iemocap_10fold'
    # setattr(opt, 'miss_num', 'mix')
    for cv in range(1, 11):
        opt.cvNo = cv
        teacher_path_cv = os.path.join(teacher_path, str(cv))
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)
        model.cuda()
        model.load_networks_cv(teacher_path_cv)

        save_root = 'checkpoints/new_cra_single_AE256,128,64_blocks5_run1_test/' + str(cv)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        # acc, uar, f1, cm = eval(model, dataset, save_root, phase='trn')
        # print('Trn result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        # print('\n{}'.format(cm))

        acc, uar, f1, cm = eval(model, val_dataset, save_root, phase='val')
        print('Val result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        print('\n{}'.format(cm))

        acc, uar, f1, cm = eval(model, tst_dataset, save_root, phase='test')
        print('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        print('\n{}'.format(cm))
        