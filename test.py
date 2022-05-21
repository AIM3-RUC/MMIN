import os
import json
import numpy as np
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from models.utils.config import OptConfig


def eval_miss(model, val_iter):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
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
    
    print(f'Total acc:{acc:.4f} uar:{uar:.4f} f1:{f1:.4f}')
    for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
        part_index = np.where(total_miss_type == part_name)
        part_pred = total_pred[part_index]
        part_label = total_label[part_index]
        acc_part = accuracy_score(part_label, part_pred)
        uar_part = recall_score(part_label, part_pred, average='macro')
        f1_part = f1_score(part_label, part_pred, average='macro')
        print(f'{part_name}, acc:{acc_part:.4f}, {uar_part:.4f}, {f1_part:.4f}')
    
    return acc, uar, f1, cm


def eval_all(model, val_iter):
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
    
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    
    print(f'Total acc:{acc:.4f} uar:{uar:.4f} f1:{f1:.4f}')
    return acc, uar, f1, cm


if __name__ == '__main__':
    test_miss = True
    test_base = False
    in_men = True
    total_cv = 10
    gpu_ids = [0]
    ckpt_path = "checkpoints/CAP_utt_fusion_AVL_run1"
    config = json.load(open(os.path.join(ckpt_path, 'train_opt.conf')))
    opt = OptConfig()
    opt.load(config)
    if test_base:
        opt.dataset_mode = 'multimodal'
    if test_miss:
        opt.dataset_mode = 'multimodal_miss'
        
    opt.gpu_ids = gpu_ids
    setattr(opt, 'in_mem', in_men)
    model = create_model(opt)
    model.setup(opt)
    results = []
    for cv in range(1, 1+total_cv):
        opt.cvNo = cv
        tst_dataloader = create_dataset_with_args(opt, set_name='tst')
        model.load_networks_cv(os.path.join(ckpt_path, str(cv)))
        model.eval()
        if test_base:
            acc, uar, f1, cm = eval_all(model, tst_dataloader)
        if test_miss:
            acc, uar, f1, cm = eval_miss(model, tst_dataloader)
        results.append([acc, uar, f1])
    
    mean_acc = sum([x[0] for x in results])/total_cv
    mean_uar = sum([x[1] for x in results])/total_cv
    mean_f1 = sum([x[2] for x in results])/total_cv
    print(f'Avg acc:{mean_acc:.4f} uar:{mean_uar:.4f} f1:{mean_f1:.4f}')
    