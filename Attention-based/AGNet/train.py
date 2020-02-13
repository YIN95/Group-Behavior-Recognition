import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from loader import data_loader
from sklearn.model_selection import cross_val_score
from models import *
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import statistics

from dataprocess import *
from sklearn.metrics import confusion_matrix

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='data', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--slidingwindow_len', default=180, type=int)
parser.add_argument('--seq_len', default=60, type=int)

# Optimization
parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--num_iterations', default=100, type=int)
parser.add_argument('--num_epochs', default=50, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--device_ids', default=[0], type=list)
parser.add_argument('--h_dim', default=16, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--clipping_threshold', default=1.5, type=float)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=50, type=int)
parser.add_argument('--checkpoint_every', default=10, type=int)
parser.add_argument('--checkpoint_name', default='group_test')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
    # if isinstance(m, nn.Conv2d):
    #     xavier(m.weight.data)
    #     xavier(m.bias.data)
    elif classname.find('Conv1d')!=-1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    # logger.info('loss is {}'.format(loss))
    return loss.mean()

def model_loss(score, target):
    return bce_loss(score, target)

def check_accuracy(
    args, loader, model, loss_fn
):
    metrics={}
    model.eval()
    losses=[]
    accuracy=[]
    total_pred=[]
    total_gt=[]
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (groupInput, target) = batch
            groupInput = groupInput.to(device)
            # print(groupInput.size())
            target = target.to(device)
            _,_,_,groupOutput = model(groupInput)
            loss = loss_fn(groupOutput, target)
            losses.append(loss)
            pred = torch.argmax(groupOutput, dim=1)
            gt=torch.argmax(target,dim=1)
            right=0
            for i in range(len(pred)):
                if pred[i]==gt[i]:
                    right+=1
            accuracy.append(right/len(pred))
            pred_np = pred.cpu().numpy().tolist()
            target_np = gt.cpu().numpy().tolist()
            total_pred+=pred_np
            total_gt+=target_np
    confumatrix = confusion_matrix(total_gt, total_pred)
    metrics['loss']=sum(losses)/len(losses)
    # logger.info('batch size {}'.format(len(accuracy)))
    model.train()
    return metrics, statistics.mean(accuracy), confumatrix

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    # train_path = get_dset_path(args.dataset_name, 'train')
    # val_path = get_dset_path(args.dataset_name, 'val')
    data_path = get_dset_path(args.dataset_name)

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing dataset")
    data_dset, data_load = data_loader(args, data_path)
    # logger.info("Initializing val dataset")
    # _, val_loader = data_loader(args, val_path)

    logger.info('{} data trials are loaded'.format(len(data_dset)))

    iterations_per_epoch = len(data_dset) / args.batch_size
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    model=GroupEncoder(
        embedding_dim=args.embedding_dim,
        h_dim=args.h_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_markers=37
    )

    cv = KFold(n_splits=5, random_state=42, shuffle=False)
    foldcounter=0
    accuracy10fold=[]
    for ifold, (train_index, valid_index) in enumerate(cv.split(data_dset)):
        train=torch.utils.data.Subset(data_dset,train_index)
        valid=torch.utils.data.Subset(data_dset,valid_index)
        train_loader=torch.utils.data.DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers
        )
        vaild_loader=torch.utils.data.DataLoader(
            valid,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers
        )

        model.apply(init_weights)
        model.type(float_dtype).train()
        logger.info('The {}th fold test'.format(foldcounter))
        foldcounter+=1
        logger.info('Here is the model:')
        logger.info(model)

        model_loss_fn=model_loss
        optimizer=optim.Adam(model.parameters(), lr=args.learning_rate)

        model.to(device)

        restore_path = None
        if args.checkpoint_start_from is not None:
            restore_path = args.checkpoint_start_from
        elif args.restore_from_checkpoint == 1:
            restore_path = os.path.join(args.output_dir,
                                        '%s_%s_with_model.pt' % (args.checkpoint_name, str(ifold)))

        if restore_path is not None and os.path.isfile(restore_path):
            logger.info('Restoring from checkpoint {}'.format(restore_path))
            checkpoint = torch.load(restore_path)
            model.load_state_dict(checkpoint['state'])
            optimizer.load_state_dict(checkpoint['optim_state'])
            t = checkpoint['counters']['t']
            epoch = checkpoint['counters']['epoch']
            checkpoint['restore_ts'].append(t)
        else:
            # Starting from scratch, so initialize checkpoint data structure
            t, epoch = 0, 0
            checkpoint = {
                'args': args.__dict__,
                'losses': [],
                'losses_ts': [],
                'metrics_val': defaultdict(list),
                'metrics_train': defaultdict(list),
                'sample_ts': [],
                'restore_ts': [],
                'norm': [],
                'accuracy':[],
                'confusionmatrix':[],
                'counters': {
                    't': None,
                    'epoch': None,
                },
                'state': None,
                'optim_state': None,
                'best_state': None,
                'best_accuracy': None,
                'd_best_state': None,
                'best_t': None,
            }

        t0 = None
        best_accuracy=0
        while t < args.num_iterations:
            gc.collect()
            epoch += 1
            logger.info('Starting {}th fold epoch {}'.format(foldcounter, epoch))
            for batch in train_loader:
                if args.timing == 1:
                    torch.cuda.synchronize()
                    t1 = time.time()

                losses={}
                # print(batch)
                # batch = [tensor.cuda() for tensor in batch]
                (groupInput, target)=batch
                groupInput=groupInput.to(device)
                target=target.to(device)
                _,_,_,groupOutput=model(groupInput)
                loss=model_loss_fn(groupOutput,target)
                # loss=nn.BCELoss(groupOutput,target)
                logger.info('loss is {}'.format(loss))
                losses['loss']=loss
                optimizer.zero_grad()
                loss.backward()
                # if args.clipping_threshold > 0:
                #     nn.utils.clip_grad_norm_(
                #         model.parameters(), args.clipping_threshold
                #     )
                optimizer.step()
                checkpoint['norm'].append(
                    get_total_norm(model.parameters()))

                if args.timing == 1:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    logger.info('it took {}'.format(t2 - t1))

                if args.timing == 1:
                    if t0 is not None:
                        logger.info('Interation {} took {}'.format(
                            t - 1, time.time() - t0
                        ))
                    t0 = time.time()

                # Maybe save loss
                if t % args.print_every == 0:
                    logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                    # for k, v in sorted(losses.items()):
                    #     logger.info('  [Loss] {}: {:.3f}'.format(k, v))
                    checkpoint['losses'].append(loss)
                    checkpoint['losses_ts'].append(t)

                # Maybe save a checkpoint
                if t > 0 and t % args.checkpoint_every == 0:
                    checkpoint['counters']['t'] = t
                    checkpoint['counters']['epoch'] = epoch
                    checkpoint['sample_ts'].append(t)

                    # Check stats on the validation set
                    logger.info('Checking stats on val ...')
                    metrics_val, accuracy, confumatrix = check_accuracy(
                        args, vaild_loader, model, model_loss_fn
                    )
                    checkpoint['accuracy'].append(accuracy)
                    logger.info('  [val] accuracy: {:.3f}'.format(accuracy))
                    logger.info('  [val] confusion matrix: {}'.format(confumatrix))
                    for k, v in sorted(metrics_val.items()):
                        logger.info('  [val] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_val'][k].append(v)

                    min_loss = min(checkpoint['metrics_val']['loss'])

                    if metrics_val['loss'] == min_loss:
                        logger.info('New low for avg_error')
                        checkpoint['best_t'] = t
                        checkpoint['best_state'] = model.state_dict()
                        checkpoint['best_accuracy']=accuracy
                        checkpoint['confusionmatrix']=confumatrix
                        best_accuracy=accuracy

                    # Save another checkpoint with model weights and
                    # optimizer state
                    checkpoint['state'] = model.state_dict()
                    checkpoint['optim_state'] = optimizer.state_dict()
                    checkpoint_path = os.path.join(
                        args.output_dir, '%s_%s_with_model.pt' % (args.checkpoint_name, str(ifold))
                    )
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)
                    logger.info('Done.')

                t += 1
                if t >= args.num_iterations:
                    break
        accuracy10fold.append(best_accuracy)
    # scores = cross_val_score(model, X_data, y_data, cv=10, scoring="accuracy")
    logger.info('accuracy10fold'.format(accuracy10fold))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)