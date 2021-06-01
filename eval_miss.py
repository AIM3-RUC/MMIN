import os
import json
import numpy as np
import argparse

from torch import int16
from opts.test_opts import TestOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from models.utils.config import OptConfig

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval_miss(model, val_iter, is_save=False, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        missing_index = data['missing_index']
        data['A_feat'] = data['A_feat'] * missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        data['L_feat'] = data['L_feat'] * missing_index[:, 2].unsqueeze(1).unsqueeze(2)
        data['V_feat'] = data['V_feat'] * missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        model.set_input(data)            # unpack data from dataset and apply preprocessing
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
    
    if is_save:
        # save test whole results
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)
    
        # save part results
        for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
            part_index = np.where(total_miss_type == part_name)
            part_pred = total_pred[part_index]
            part_label = total_label[part_index]
            acc_part = accuracy_score(part_label, part_pred)
            uar_part = recall_score(part_label, part_pred, average='macro')
            f1_part = f1_score(part_label, part_pred, average='macro')
            np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
            np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
            if phase == 'test':
                recorder_lookup[part_name].write_result_to_tsv({
                    'acc': acc_part,
                    'uar': uar_part,
                    'f1': f1_part
                }, cvNo=opt.cvNo)

    model.train()
    return acc, uar, f1, cm

def eval(model, val_iter, is_save=False, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)            # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    if phase == 'test':
        recorder_lookup["total"].write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1
        }, cvNo=opt.cvNo)

    if is_save:
        # save test whole results
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    model.train()
    return acc, uar, f1, cm

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt

def parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_mode', default=0, type=str)
    parser.add_argument('--model_path', default=0, type=str)
    parser.add_argument('--checkpoints_dir', default=0, type=str)
    parser.add_argument('--log_dir', default=0, type=str)
    parser.add_argument('--name', default=0, type=str)
    parser.add_argument('--cvNo', default=0, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--eval_miss', action='store_true')
    parser.add_argument('--total_cv', type=int, default=10)
    opt, _ = parser.parse_known_args()
    return opt
        
if __name__ == '__main__':
    opt = parse()
    # opt 加载模型 数据集合即可
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo)) # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.makedirs(logger_path)
    suffix = '_'.join([opt.model_path, opt.dataset_mode])    # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger

    result_dir = os.path.join(opt.log_dir, opt.name, 'results')
    if not os.path.exists(result_dir):                 # make sure result path exists
        os.mkdir(result_dir)
    
    recorder_lookup = {                                 # init result recoreder
        "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=opt.total_cv),
        "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv')),
        "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv')),
        "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv')),
        "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv')),
        "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv')),
        "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv')),
    }
    
    model_path = os.path.join(opt.checkpoints_dir, opt.model_path) #'checkpoints/emobert_test_parameter_V_run1'
    model_opt_path = os.path.join(model_path, 'train_opt.conf')
    model_opt = load_from_opt_record(model_opt_path)
    model_opt.isTrain = False                             # teacher model should be in test mode
    model_opt.gpu_ids = [opt.gpu_id]
    model_opt.serial_batches = True
    model_opt.dataset_mode = opt.dataset_mode
    model_opt.cvNo = opt.cvNo

    model = create_model(model_opt)      # create a model given opt.model and other options
    model.setup(model_opt)               # regular setup: load and print networks; create schedulers
    model.cuda()
    model.load_networks_cv(os.path.join(model_path, str(opt.cvNo)))
    
    val_dataset, tst_dataset = create_dataset_with_args(model_opt, set_name=['val', 'tst'])  
    _ = eval(model, val_dataset, is_save=True, phase='val')
    if opt.eval_miss:
        acc, uar, f1, cm = eval_miss(model, tst_dataset, is_save=True, phase='test')
    else:
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
    logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
    logger.info('\n{}'.format(cm))
    recorder_lookup['total'].write_result_to_tsv({
        'acc': acc,
        'uar': uar,
        'f1': f1
    }, cvNo=opt.cvNo)
