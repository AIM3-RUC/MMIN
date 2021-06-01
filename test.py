import os
import time
import numpy as np
import sys
from opts.test_opts import TestOptions
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score

def test(opt, phase):
    if opt.method == 'concat':
        total_label = []
        total_pred = []
        root = os.path.join(opt.checkpoints_dir, opt.name)
        for cv in range(1, 11):
            cv_dir = os.path.join(root, str(cv))
            cur_cv_pred = np.load(os.path.join(cv_dir, '{}_pred.npy'.format(phase)))
            cur_cv_label = np.load(os.path.join(cv_dir, '{}_label.npy'.format(phase)))
            total_label.append(cur_cv_label)
            total_pred.append(cur_cv_pred)
        
        total_label = np.concatenate(total_label)
        total_pred = np.concatenate(total_pred)
        acc = accuracy_score(total_label, total_pred)
        wap = precision_score(total_label, total_pred, average='weighted')
        wf1 = f1_score(total_label, total_pred, average='weighted')
        uar = recall_score(total_label, total_pred, average='macro')
        f1 = f1_score(total_label, total_pred, average='macro')
        cm = confusion_matrix(total_label, total_pred)
        print(opt.name)
        # print('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        print('{} result acc %.4f wap %.4f wf1 %.4f uar %.4f f1 %.4f' % (phase.upper(), acc, wap, wf1, uar, f1))
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (acc, wap, wf1, uar, f1))
        print('\n{}'.format(cm))
    
    elif opt.method == 'mean':
        total_acc = []
        total_wap = []
        total_wf1 = []
        total_uar = []
        total_f1 = []
        root = os.path.join(opt.checkpoints_dir, opt.name)
        for cv in range(1, 11):
            cv_dir = os.path.join(root, str(cv))
            cur_cv_pred = np.load(os.path.join(cv_dir, '{}_pred.npy'.format(phase)))
            cur_cv_label = np.load(os.path.join(cv_dir, '{}_label.npy'.format(phase)))
            acc = accuracy_score(cur_cv_label, cur_cv_pred)
            wap = precision_score(cur_cv_label, cur_cv_pred, average='weighted')
            wf1 = f1_score(cur_cv_label, cur_cv_pred, average='weighted')
            uar = recall_score(cur_cv_label, cur_cv_pred, average='macro')
            f1 = f1_score(cur_cv_label, cur_cv_pred, average='macro')
            total_acc.append(acc)
            total_wap.append(wap)
            total_wf1.append(wf1)
            total_uar.append(uar)
            total_f1.append(f1)
            if opt.verbose:
                cm = confusion_matrix(cur_cv_label, cur_cv_pred)
                print('In cv {} confusion matrix:\n{}'.format(cv, cm))
            
            if not opt.simple:
                print('{:.4f}\t{:.4f}\t{:.4f}'.format(acc, uar, f1))
        
        acc = '{:.4f}±{:.4f}'.format(float(np.mean(total_acc)), float(np.std(total_acc)))
        wap = '{:.4f}±{:.4f}'.format(float(np.mean(total_wap)), float(np.std(total_wap)))
        wf1 = '{:.4f}±{:.4f}'.format(float(np.mean(total_wf1)), float(np.std(total_wf1)))
        uar = '{:.4f}±{:.4f}'.format(float(np.mean(total_uar)), float(np.std(total_uar)))
        f1 = '{:.4f}±{:.4f}'.format(float(np.mean(total_f1)), float(np.std(total_f1)))
        print(opt.name)
        # print('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        if opt.simple:
            print('{} result:\n{:.4f}\t{:.4f}\t{:.4f}'.format(phase.upper(), float(np.mean(total_acc)), float(np.mean(total_uar)), float(np.mean(total_f1))))
        else:
            print('%s result:\nacc %s wap %s wf1 %s uar %s f1 %s' % (phase.upper(), acc, wap, wf1, uar, f1))

if __name__ == '__main__':
    opt = TestOptions().parse()    # get training options    
    test(opt, 'val')
    print('----------------------')
    test(opt, 'test')