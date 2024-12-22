import os
import argparse
import time
import datetime
import numpy as np
import torch
import sklearn.metrics
from utils import save_checkpoint, load_best_checkpoint, AverageMeter, Logger
import gpnn_model
import datasetAPI

import config
import warnings

# hyper parameters
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
dataset = 'cad'
project_root = 'D:/ai double major/Core/new_GPNN'



if dataset == 'hico':
    action_class_num = 117
elif dataset == 'vcoco':
    action_class_num = 27
elif dataset == 'cad':
    action_class_num = 22
else:
    raise RuntimeError(f'not defined dataset {dataset}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    start_time = time.time()

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = Logger(os.path.join(args.log_root, timestamp))

    # Load data
    if args.dataset == "hico":
        training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = datasetAPI.get_hico_data(args)
        edge_features, node_features = training_set[0][0], training_set[0][1]
        edge_feature_size, node_feature_size = edge_features.shape[2], node_features.shape[1]
    elif args.dataset == 'vcoco':
        training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = datasetAPI.get_vcoco_data(args)
        edge_features, node_features = training_set[0][0], training_set[0][1]
        edge_feature_size, node_feature_size = edge_features.shape[2], node_features.shape[1]
    else: # cad 120
        training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = datasetAPI.get_cad_data(args)
        edge_features, node_features = training_set[0][0], training_set[0][1]
        edge_feature_size, node_feature_size = edge_features.shape[0], node_features.shape[0]

    message_size = edge_feature_size

    model_args = config.model_args(args,edge_feature_size,node_feature_size,message_size)
    model = gpnn_model.General_GPNN(model_args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # in hico it is this scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    multi_label_loss = torch.nn.MultiLabelSoftMarginLoss

    if args.cuda:
        model = model.to(device)

    if args.if_load_checkpoint:
        loaded_checkpoint = load_best_checkpoint(args, model, optimizer)
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    epoch_errors = []
    avg_epoch_error = np.inf
    best_epoch_error = np.inf


    for epoch in range(args.start_epoch, args.epochs):
        logger.log_value('learning_rate', args.lr).step()
        train(train_loader, model, multi_label_loss, optimizer, epoch, logger)
        epoch_error = validate(valid_loader, model, multi_label_loss, logger)
        epoch_errors.append(epoch_error)
        scheduler.step()
        is_best = True
        best_epoch_error = min(epoch_error, best_epoch_error)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                        'best_epoch_error': best_epoch_error, 'avg_epoch_error': avg_epoch_error,
                                        'optimizer': optimizer.state_dict(), },
                                       is_best=is_best, directory=args.resume)
        print(f'best_epoch_error: {best_epoch_error}, avg_epoch_error: {avg_epoch_error}')

    # For testing
    loaded_checkpoint = load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint
    # to test
    print("\n ======================================= \n |               Testing               | \n ======================================= \n ")
    test_error = validate(test_loader, model, multi_label_loss)
    print(f'Test error : {test_error}Time elapsed: {time.time() - start_time:.2f}s')


def train(train_loader, model, multi_label_loss, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))

    model.train()

    end_time = time.time()

    for i, (edge_features, node_features, adj_mat, node_labels, human_num, obj_num) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        edge_features = edge_features.to(device)
        node_features = node_features.to(device)
        adj_mat = adj_mat.to(device)
        node_labels = node_labels.to(device)

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num, args)
        det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, multi_label_loss, human_num, obj_num)

        # Log and back propagate
        if len(det_indices) > 0:
            y_true, y_score = evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score)
        losses.update(loss.item(), edge_features.size()[0])
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.log_interval == 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}], Time {batch_time.val:.3f} ({batch_time.avg:.3f}),Data {data_time.val:.3f},({data_time.avg:.3f}), \
                  Mean Avg Precision {mean_avg_prec:.4f},({mean_avg_prec:.4f}),Detected HOIs {y_true.shape}')
            break

    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)
        logger.log_value('train_epoch_map', mean_avg_prec)

    print(f'Epoch: [{epoch}] Avg Mean Precision {mean_avg_prec:.4f}; Average Loss {losses.avg:.4f}; Avg Time x Batch {batch_time.avg:.4f}')


def validate(val_loader, model, multi_label_loss, logger=None, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))

    model.eval()
    end = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, human_num, obj_num) in enumerate(val_loader):

        edge_features = edge_features.to(device)
        node_features = node_features.to(device)
        adj_mat = adj_mat.to(device)
        node_labels = node_labels.to(device)

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num, args)
        det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, multi_label_loss, human_num, obj_num)

        if len(det_indices) > 0:
            losses.update(loss.item(), len(det_indices))
            y_true, y_score = evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print(f'Test: [{i}/{len(val_loader)}], Time {batch_time.val:.3f} ({batch_time.avg:.3f}), \
                  Mean Avg Precision {mean_avg_prec:.4f},({mean_avg_prec:.4f}),Detected HOIs {y_true.shape}')
            break

    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)

    print(f'Test Avg Mean Precision {mean_avg_prec:.4f}; Average Loss {losses.avg:.4f}; Avg Time x Batch {batch_time.avg:.4f}')

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_map', mean_avg_prec)

    return 1.0 - mean_avg_prec


def evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score):
    np_pred_node_labels = pred_node_labels.sigmoid().cpu().detach().numpy()
    np_node_labels = node_labels.data.cpu().numpy()

    new_y_true = np.empty((2 * len(det_indices), action_class_num))
    new_y_score = np.empty((2 * len(det_indices), action_class_num))
    for y_i, (batch_i, i, j) in enumerate(det_indices):
        new_y_true[2*y_i, :] = np_node_labels[batch_i, i, :]
        new_y_true[2*y_i+1, :] = np_node_labels[batch_i, j, :]
        new_y_score[2*y_i, :] = np_pred_node_labels[batch_i, i, :]
        new_y_score[2*y_i+1, :] = np_pred_node_labels[batch_i, j, :]
    y_true = np.vstack((y_true, new_y_true))
    y_score = np.vstack((y_score, new_y_score))
    return y_true, y_score


def loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, multi_label_loss, human_num=[], obj_num=[]):    
    batch_size = pred_adj_mat.size()[0]
    loss = 0.0
    det_indices = []

    for batch_i in range(batch_size):
        valid_node_num = human_num[batch_i] + obj_num[batch_i]
        pred_node_label = pred_node_labels[batch_i, :valid_node_num].view(-1, action_class_num)
        node_label = node_labels[batch_i, :valid_node_num].view(-1, action_class_num)

        # 计算权重掩码
        weight_mask = torch.ones_like(node_label, device=device)
        weight_mask += node_label * (args.link_weight if hasattr(args, 'link_weight') else 1.0)

        # 计算加权损失
        loss += multi_label_loss(weight=weight_mask, reduction='mean').to(device)(pred_node_label, node_label)

        # 计算det_indices
        np_pred_adj_mat_batch = pred_adj_mat[batch_i].data.cpu().numpy()
        human_interval = human_num[batch_i]
        obj_interval = human_interval + obj_num[batch_i]
        batch_det_indices = np.where(np_pred_adj_mat_batch > 0.5)

        for i, j in zip(batch_det_indices[0], batch_det_indices[1]):
            if i < human_interval and j >= human_interval and j < obj_interval:
                det_indices.append((batch_i, i, j))

    return det_indices, loss


def compute_mean_avg_prec(y_true, y_score):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            avg_prec = sklearn.metrics.average_precision_score(y_true, y_score, average=None)        
            mean_avg_prec = np.nansum(avg_prec) / len(avg_prec)
    except ValueError:
        mean_avg_prec = 0
    return mean_avg_prec

def parse_arguments():
    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument('--dataset',type= str, default=dataset,help = 'should be hico, vcoco or cad')
    # Path settings
    parser.add_argument('--project-root', default=project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=os.path.join(project_root,'tmp'), help='intermediate result path')
    parser.add_argument('--cad-data-root', default=os.path.join(project_root,'tmp','cad'), help='data path')
    parser.add_argument('--hico-data-root', default=os.path.join(project_root, 'tmp', 'hico'), help='data path') 
    parser.add_argument('--vcoco-data-root', default=os.path.join(project_root, 'tmp', 'vcoco'), help='data path') 
    parser.add_argument('--feature_type',default='vcoco_features')
    parser.add_argument('--log-root', default=os.path.join('log', dataset), help='log files path')
    parser.add_argument('--resume', default=os.path.join(project_root,'tmp', 'checkpoints',dataset,'parsing'), help='path to latest checkpoint')
    parser.add_argument('--if_load_checkpoint',default=False,help='load best checkpoint to test')

    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize final results')
    parser.add_argument('--vis-top-k', type=int, default=1, metavar='N', help='Top k results to visualize')

    # model setting
    parser.add_argument('--model_path',type=str,default=os.path.join(project_root,'tmp', 'checkpoints',dataset,'parsing'), help='path to latest checkpoint')
    parser.add_argument('--LinkFunction',type = str,default = 'graphconv',help='graphconv or graphconvlstm')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',help='Input batch size for training (default: 10)')
    # need modify
    parser.add_argument('--edge_feature_size',type= int,default = -1,help='input_size')
    parser.add_argument('--node_feature_size',type= int,default=-1,help = 'node_feature_size')
    parser.add_argument('--link_hidden_size',type= int,default = -1,help='hidden_size')
    parser.add_argument('--link_hidden_layers',type= int,default = -1,help='hidden_layers')
    parser.add_argument('--message_def',type = str, default='linear',help = 'message function')
    parser.add_argument('--message_size',type = int,default=-1,help = 'message_size')
    parser.add_argument('--update_def',default='gru',help='must be gru')
    parser.add_argument('--hoi_classes',type = int,default=-1,help = 'hoi_classes')
    parser.add_argument('--propagate_layers',type = int,default=-1,help = 'propagate_layers')
    parser.add_argument('--link_relu',type = bool, default=False,help='if use relu in link.')

    # Optimization Options
    parser.add_argument('--cuda', default=torch.cuda.is_available(), help=' if use cuda')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu',help = 'device')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',help='Index of epoch to start (default: 0)')
    parser.add_argument('--link-weight', type=float, default=100, metavar='N',help='Loss weight of existing edges')
    parser.add_argument('--lr', type=float, default = 1e-4, metavar='LR',help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.8, metavar='LR-DECAY',help='Learning rate decay factor [.01, 1] (default: 0.8)')
    parser.add_argument('--step_size',type = int, default=2,help = 'step size')
    # i/o
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--num_workers', type=int, default=1, help='num workers to load the data.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
