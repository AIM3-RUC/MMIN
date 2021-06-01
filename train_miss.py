import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test'):
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
    
    if is_save:
        # save test whole results
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)
    
        # save part results
        for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl', 'avl']:
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

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

if __name__ == '__main__':
    opt = TrainOptions().parse()                        # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo)) # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.mkdir(logger_path)
    
    result_dir = os.path.join(opt.log_dir, opt.name, 'results')
    if not os.path.exists(result_dir):                 # make sure result path exists
        os.mkdir(result_dir)
    
    recorder_lookup = {                                 # init result recoreder
        "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv')),
        "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv')),
        "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv')),
        "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv')),
        "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv')),
        "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv')),
        "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv')),
        "avl": ResultRecorder(os.path.join(result_dir, 'result_avl.tsv')),
    }

    suffix = '_'.join([opt.model, opt.dataset_mode])    # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger

    if opt.has_test:                                    # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])  
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    best_eval_uar = 0              # record the best eval UAR
    best_eval_epoch = -1           # record the best eval epoch
    best_eval_acc, best_eval_f1 = 0, 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)           # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.
        
        # eval
        acc, uar, f1, cm = eval(model, val_dataset)
        logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (epoch, opt.niter + opt.niter_decay, acc, uar, f1))
        logger.info('\n{}'.format(cm))
        
        # show test result for debugging
        if opt.verbose:
            acc, uar, f1, cm = eval(model, tst_dataset)
            logger.info('Tst result of epoch %d acc %.4f uar %.4f f1 %.4f' % (epoch, acc, uar, f1))
            logger.info('\n{}'.format(cm))

        # record epoch with best result
        if uar > best_eval_uar:
            best_eval_epoch = epoch
            best_eval_uar = uar
            best_eval_acc = acc
            best_eval_f1 = f1

    logger.info('Best eval epoch %d found with uar %f' % (best_eval_epoch, best_eval_uar))
    # test
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        _ = eval(model, val_dataset, is_save=True, phase='val')
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
        logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        logger.info('\n{}'.format(cm))
        recorder_lookup['total'].write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1
        }, cvNo=opt.cvNo)
    else:
        recorder_lookup['total'].write_result_to_tsv({
            'acc': best_eval_acc,
            'uar': best_eval_uar,
            'f1': best_eval_f1
        }, cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
