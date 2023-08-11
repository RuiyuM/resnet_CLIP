import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import Sampling

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import datasets
import models
import torchvision.models as torch_models
import pickle
from utils import AverageMeter, Logger
from center_loss import CenterLoss

import nets


from mqnet_utils import *

from extract_features import CIFAR100_LOAD_ALL

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['Tiny-Imagenet', 'cifar100', 'cifar10'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
#parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--max-query', type=int, default=10)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--query-strategy', type=str, default='AV_based2',
                    choices=['random', 'uncertainty', 'AV_based', 'AV_uncertainty', 'AV_based2', 'Max_AV',
                             'AV_temperature', 'My_Query_Strategy', 'test_query', 'MQNet', 'test_query_2', 'active_query', "BGADL", "OpenMax", "Core_set", 'BADGE_sampling', "certainty"])
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='ResNet18')
# misc
parser.add_argument('--eval-freq', type=int, default=100)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)
parser.add_argument('--known-T', type=float, default=0.5)
parser.add_argument('--unknown-T', type=float, default=2)
parser.add_argument('--modelB-T', type=float, default=1)
parser.add_argument('--init-percent', type=int, default=16)

# active learning

parser.add_argument('--active', action='store_true', help="whether to use active learning")

parser.add_argument('--k', type=int, default=10)

parser.add_argument('--runs', type=int, default=3)

parser.add_argument('--active_5', action='store_true', help="whether to use active learning")

parser.add_argument('--pre-type', type=str, default='clip')


# MQNet
# Optimizer and scheduler
parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for updating network parameters')
parser.add_argument('--lr-mqnet', type=float, default=0.001, help='learning rate for updating mqnet')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument("--scheduler", default="MultiStepLR", type=str, help="Learning rate scheduler") #CosineAnnealingLR, StepLR, MultiStepLR
parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate for CosineAnnealingLR')
parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")
parser.add_argument('--milestone', type=list, default=[100, 150], metavar='M', help='Milestone for MultiStepLR')
parser.add_argument('--warmup', type=int, default=10, metavar='warmup', help='warmup epochs')


parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--epoch-loss', default=120, type=int, help='number of epochs for training loss module in LL')
parser.add_argument('--epochs-ccal', default=700, type=int, help='number of epochs for training contrastive coders in CCAL')
parser.add_argument('--epochs-csi', default=1000, type=int, help='number of epochs for training CSI')
parser.add_argument('--epochs-mqnet', default=100, type=int, help='number of epochs for training mqnet')
parser.add_argument('--steps-per-epoch', type=int, default=100, metavar='N', help='number of steps per epoch')
parser.add_argument("--test-batch-size", "-tb", default=500, type=int)
parser.add_argument('--ccal-batch-size', default=32, type=int, metavar='N')
parser.add_argument('--csi-batch-size', default=32, type=int, metavar='N')



# for MQNet
parser.add_argument('--mqnet-mode', default="LL", help="specifiy the mode of MQNet to use") #CONF, LL

args = parser.parse_args()

    


def init_mqnet(args, nets, models, optimizers, schedulers):
    models['mqnet'] = nets.__dict__['QueryNet'](input_size=2, inter_dim=64).cuda()

    optim_mqnet = torch.optim.SGD(models['mqnet'].parameters(), lr=args.lr_mqnet)
    sched_mqnet = torch.optim.lr_scheduler.MultiStepLR(optim_mqnet, milestones=[int(args.epochs_mqnet / 2)])

    optimizers['mqnet'] = optim_mqnet
    schedulers['mqnet'] = sched_mqnet
    return models, optimizers, schedulers



def main():


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()


    ordered_feature, ordered_label, index_to_label = CIFAR100_LOAD_ALL(dataset=args.dataset, pre_type=args.pre_type)


    sys.stdout = Logger(osp.join(args.save_dir, args.query_strategy + '_log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))


    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
    )

    testloader, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_A, trainloader_B = dataset.trainloader, dataset.trainloader
    
    negativeloader = None  # init negativeloader none
    invalidList = []


    labeled_ind_train, unlabeled_ind_train, trainset = dataset.labeled_ind_train, dataset.unlabeled_ind_train, dataset.trainset

    per_round = []
     
    per_round.append(list(labeled_ind_train))



    print("Creating model: {}".format(args.model))
    # model = models.create(name=args.model, num_classes=dataset.num_classes)
    #
    # if use_gpu:
    #     model = nn.DataParallel(model).cuda()
    #
    # criterion_xent = nn.CrossEntropyLoss()
    # criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    # optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)
    #
    # if args.stepsize > 0:
    #     scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    Acc = {}
    Err = {}
    Precision = {}
    Recall = {}

    model_B = None

    for query in tqdm(range(args.max_query)):
            
        '''
        # Model initialization
        if args.model == "cnn":
            model = models.create(name=args.model, num_classes=dataset.num_classes)
        elif args.model == "resnet18":
            # 多出的一类用来预测为unknown
            #model_A = resnet18(num_classes=dataset.num_classes + 1)
            model_B = resnet18(num_classes=dataset.num_classes)
        elif args.model == "resnet34":
            #model_A = resnet34(num_classes=dataset.num_classes + 1)
            model_B = resnet34(num_classes=dataset.num_classes)
        elif args.model == "resnet50":
            #model_A = resnet50(num_classes=dataset.num_classes + 1)
            model_B = resnet50(num_classes=dataset.num_classes)
        '''

        model_B = get_models(args, nets, args.model, model_B, dataset)

        #criterion_xent = nn.CrossEntropyLoss()
        #criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
       
        #optimizer_model_A = torch.optim.SGD(model_A.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        #optimizer_model_B = torch.optim.SGD(model_B.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        
        #optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

        #if args.stepsize > 0:
            #scheduler_A = lr_scheduler.StepLR(optimizer_model_A, step_size=args.stepsize, gamma=args.gamma)
        #    scheduler_B = lr_scheduler.StepLR(optimizer_model_B, step_size=args.stepsize, gamma=args.gamma)


        criterion, optimizers, schedulers = get_optim_configurations(args, model_B)


        '''
        # Model training
        for epoch in tqdm(range(args.max_epoch)):

            # Train model B for classifying known classes
            train_B(model_B, criterion_xent, criterion_cent,
                    optimizer_model_B, optimizer_centloss,
                    trainloader_B, use_gpu, dataset.num_classes, epoch)

            if args.stepsize > 0:
                #scheduler_A.step()
                scheduler_B.step()

            if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
                print("==> Test")
                #acc_A, err_A = test(model_A, testloader, use_gpu, dataset.num_classes, epoch)
                acc_B, err_B = test(model_B, testloader, use_gpu, dataset.num_classes, epoch)
                #print("Model_A | Accuracy (%): {}\t Error rate (%): {}".format(acc_A, err_A))
                print("Model_B | Accuracy (%): {}\t Error rate (%): {}".format(acc_B, err_B))
        '''

        train(args, model_B, criterion, optimizers, schedulers, trainloader_B)


        # Record results
        #acc, err = test(model_B, testloader, use_gpu, dataset.num_classes, args.max_epoch)
        
        acc = test(args, model_B, testloader)


        last_acc = acc

        Acc[query] = float(acc)
        
        # Query samples and calculate precision and recall
        queryIndex = []


        if args.query_strategy == "MQNet":
            
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.MQNet(args, model_B, query, len(labeled_ind_train), unlabeledloader, trainloader_B, use_gpu)

            ######################################################################

            models, optimizers, schedulers = init_mqnet(args, nets, model_B, optimizers, schedulers)
            
            #unlabeled_loader = DataLoader(unlabeled_dst, sampler=SubsetRandomSampler(U_index), batch_size=64, num_workers=4)
            
            delta_loader     = DataLoader(trainset, sampler=SubsetRandomSampler(queryIndex), batch_size=max(1, 32), num_workers=0)
            

            models = meta_train(args, models, optimizers, schedulers, criterion, trainloader_B, unlabeledloader, delta_loader)

            ######################################################################

        ##############################################################################################################

        per_round.append(list(queryIndex) + list(invalidIndex))


        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train  = list(set(unlabeled_ind_train) - set(queryIndex))
        
        labeled_ind_train    = list(labeled_ind_train) + list(queryIndex)
        
        invalidList          = list(invalidList) + list(invalidIndex)

        ##############################################################################################################
        

        print("Query round: " + str(query) + " | Query Strategy: " + args.query_strategy + " | Query Batch: " + str(
            args.query_batch) + " | Valid Query Nums: " + str(len(queryIndex)) + " | Query Precision: " + str(
            Precision[query]) + " | Query Recall: " + str(Recall[query]) + " | Training Nums: " + str(
            len(labeled_ind_train)) + " | Unalebled Nums: " + str(len(unlabeled_ind_train)))
        


        dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        )


        trainloader_B, unlabeledloader = dataset.trainloader, dataset.unlabeledloader

    #############################################################################################################

    
    file_name = "./log_AL/temperature_" + args.model + "_" + args.dataset + "_known" + str(args.known_class) + "_init" + str(
                    args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
                    args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
                    args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type) + "_neighbor_" + str(args.k)


    '''
    file_name = "./log_AL/hybrid_temperature_" + args.model + "_" + args.dataset + "_known" + str(args.known_class) + "_init" + str(
                    args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
                    args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
                    args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type) 
    '''

    ## Save results
    with open(file_name + ".pkl", 'wb') as f:
        
        data = {'Acc': Acc, 'Precision': Precision, 'Recall': Recall}
        pickle.dump(data, f)

    #############################################################################################################
    '''
    selected_index = "./log_AL/temperature_" + args.model + "_" + args.dataset + "_known" + str(args.known_class) + "_init" + str(
                    args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
                    args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
                    args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type) + "_neighbor_" + str(args.k)
    '''
    
    selected_index = "./log_AL/hybrid_temperature_" + args.model + "_" + args.dataset + "_known" + str(args.known_class) + "_init" + str(
                    args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
                    args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
                    args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type)


    with open(selected_index + "_per_round_query_index.pkl", 'wb') as f:
        
        pickle.dump(per_round, f)
    #############################################################################################################

    

    f.close()
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))



def calculate_precision_recall():
    precision, recall = 0, 0
    return precision, recall



def train_B(model, criterion_xent, criterion_cent,
            optimizer_model, optimizer_centloss,
            trainloader, use_gpu, num_classes, epoch):
    
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)
        loss_xent = criterion_xent(outputs, labels)
        #loss_cent = criterion_cent(features, labels)
        loss_cent = 0.0

        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        if args.weight_cent > 0.0:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / args.weight_cent)
            optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        #cent_losses.update(loss_cent.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        # if (batch_idx + 1) % args.print_freq == 0:
        #     print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
        #           .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
        #                   cent_losses.val, cent_losses.avg))

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='train')


'''
def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for index, (data, labels) in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='test')

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err
'''

def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch + 1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

# class CustomCIFAR100Dataset(Dataset):
#     def __init__(self, root='./data/', train=True, download=True, transform=None):
#         self.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
#
#     def __getitem__(self, index):
#         data_point, label = self.cifar100_dataset[index]
#         return index, (data_point, label)
#
#     def __len__(self):
#         return len(self.cifar100_dataset)

if __name__ == '__main__':
    main()




