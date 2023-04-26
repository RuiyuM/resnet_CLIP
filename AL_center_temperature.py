import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
import re
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
from resnet_openmax import resnet18 as resnet18_openmax
import Sampling

import datasets
import models
import torchvision.models as torch_models
import pickle
from utils import AverageMeter, Logger
from center_loss import CenterLoss



from extract_features import CIFAR100_EXTRACT_FEATURE_CLIP

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['Tiny-Imagenet', 'cifar100', 'cifar10'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--max-query', type=int, default=10)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--query-strategy', type=str, default='AV_based2',
                    choices=['random', 'uncertainty', 'AV_based', 'AV_uncertainty', 'AV_based2', 'Max_AV',
                             'AV_temperature', 'My_Query_Strategy', 'test_query', "BGADL", "OpenMax", "Core_set", 'certainty'])
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='resnet18')
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

args = parser.parse_args()

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    # indices, sel_idx = CIFAR100_EXTRACT_FEATURE_CLIP()

    if args.use_cpu: use_gpu = False

    #sys.stdout = Logger(osp.join(args.save_dir, args.query_strategy + '_log_' + args.dataset + '.txt'))

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
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train

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
    for query in tqdm(range(args.max_query)):

        # Model initialization
        if args.model == "cnn":
            model = models.create(name=args.model, num_classes=dataset.num_classes)
        elif args.model == "resnet18":
            # 多出的一类用来预测为unknown
            if args.query_strategy == "OpenMax":
                model_A = resnet18_openmax(num_classes=dataset.num_classes + 1)
                model_B = resnet18_openmax(num_classes=dataset.num_classes)
            else:
                model_A = resnet18(num_classes=dataset.num_classes + 1)
                model_B = resnet18(num_classes=dataset.num_classes)
        elif args.model == "resnet34":
            model_A = resnet34(num_classes=dataset.num_classes + 1)
            model_B = resnet34(num_classes=dataset.num_classes)
        elif args.model == "resnet50":
            model_A = resnet50(num_classes=dataset.num_classes + 1)
            model_B = resnet50(num_classes=dataset.num_classes)

        if use_gpu:
            model_A = nn.DataParallel(model_A).cuda()
            model_B = nn.DataParallel(model_B).cuda()


        criterion_xent = nn.CrossEntropyLoss()
        criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
        criterion_cent_special = CenterLoss(num_classes=dataset.num_classes + 1, feat_dim=2, use_gpu=use_gpu)
        optimizer_model_A = torch.optim.SGD(model_A.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        optimizer_model_B = torch.optim.SGD(model_B.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

        if args.stepsize > 0:
            scheduler_A = lr_scheduler.StepLR(optimizer_model_A, step_size=args.stepsize, gamma=args.gamma)
            scheduler_B = lr_scheduler.StepLR(optimizer_model_B, step_size=args.stepsize, gamma=args.gamma)


        # Model training
        for epoch in tqdm(range(args.max_epoch)):

            if args.query_strategy == "OpenMax":
                train_A_open_max(model_A, criterion_xent, criterion_cent,
                        optimizer_model_A, optimizer_centloss,
                        trainloader_A, invalidList, use_gpu, dataset.num_classes, epoch)
                train_B_open_max(model_B, criterion_xent, criterion_cent,
                        optimizer_model_B, optimizer_centloss,
                        trainloader_B, use_gpu, dataset.num_classes, epoch)
            else:
                # Train model A for detecting unknown classes
                train_A(model_A, criterion_xent, criterion_cent,
                    optimizer_model_A, optimizer_centloss,
                    trainloader_A, invalidList, use_gpu, dataset.num_classes, epoch)
                # Train model B for classifying known classes
                train_B(model_B, criterion_xent, criterion_cent,
                    optimizer_model_B, optimizer_centloss,
                    trainloader_B, use_gpu, dataset.num_classes, epoch)

            if args.stepsize > 0:
                scheduler_A.step()
                scheduler_B.step()

            if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
                print("==> Test")
                acc_A, err_A = test(model_A, testloader, use_gpu, dataset.num_classes, epoch, args.dataset)
                acc_B, err_B = test(model_B, testloader, use_gpu, dataset.num_classes, epoch, args.dataset)
                print("Model_A | Accuracy (%): {}\t Error rate (%): {}".format(acc_A, err_A))
                print("Model_B | Accuracy (%): {}\t Error rate (%): {}".format(acc_B, err_B))

        # Record results
        acc, err = test(model_B, testloader, use_gpu, dataset.num_classes, args.max_epoch, args.dataset)
        Acc[query], Err[query] = float(acc), float(err)
        # Query samples and calculate precision and recall
        queryIndex = []

        if args.query_strategy == "random":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.random_sampling(args, unlabeledloader,
                                                                                                 len(labeled_ind_train),
                                                                                                 model_A, use_gpu)
        elif args.query_strategy == "uncertainty":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.uncertainty_sampling(args,
                                                                                                      unlabeledloader,
                                                                                                      len(labeled_ind_train),
                                                                                                      model_A, use_gpu)
        elif args.query_strategy == "AV_based":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_sampling(args, unlabeledloader,
                                                                                             len(labeled_ind_train),
                                                                                             model_A, use_gpu)
        elif args.query_strategy == "AV_temperature":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_sampling_temperature(args,
                                                                                                         unlabeledloader,
                                                                                                         len(labeled_ind_train),
                                                                                                         model_A,
                                                                                                         use_gpu)
        elif args.query_strategy == "AV_uncertainty":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_uncertainty_sampling(args,
                                                                                                         unlabeledloader,
                                                                                                         len(labeled_ind_train),
                                                                                                         model_A,
                                                                                                         use_gpu)
        elif args.query_strategy == "AV_based2":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_sampling2(args, trainloader_B,
                                                                                              unlabeledloader,
                                                                                              len(labeled_ind_train),
                                                                                              model_A, use_gpu)
        elif args.query_strategy == "Max_AV":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.Max_AV_sampling(args, unlabeledloader,
                                                                                                 len(labeled_ind_train),
                                                                                                 model_A, use_gpu)
        # elif args.query_strategy == "My_Query_Strategy":
        #     queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.My_Query_Strategy(args, unlabeledloader,
        #                                                                                          len(labeled_ind_train),
        #                                                                                          model_A, use_gpu, labeled_ind_train, invalidList, indices, sel_idx)
        #
        #
        #
        # elif args.query_strategy == "test_query":
        #     queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.test_query(args,
        #                                                                                  unlabeledloader,
        #                                                                                  len(labeled_ind_train),
        #                                                                                  model_A,
        #                                                                                  use_gpu, labeled_ind_train, invalidList, indices, sel_idx)

        elif args.query_strategy == "BGADL":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.bayesian_generative_active_learning(args, unlabeledloader,
                                                                                                  len(labeled_ind_train), model_A, use_gpu)
        elif args.query_strategy == "OpenMax":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.openmax_sampling(args, unlabeledloader,
                                                                                                  len(labeled_ind_train), model_A, criterion_cent_special, use_gpu)

        elif args.query_strategy == "Core_set":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.core_set(args, unlabeledloader,
                                                                                                  len(labeled_ind_train), model_A, use_gpu)

        elif args.query_strategy == "certainty":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.certainty_sampling(args, unlabeledloader,
                                                                                                  len(labeled_ind_train), model_A, use_gpu)

        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train) - set(queryIndex) - set(invalidIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        invalidList = list(invalidList) + list(invalidIndex)

        print("Query Strategy: " + args.query_strategy + " | Query Batch: " + str(
            args.query_batch) + " | Valid Query Nums: " + str(len(queryIndex)) + " | Query Precision: " + str(
            Precision[query]) + " | Query Recall: " + str(Recall[query]) + " | Training Nums: " + str(
            len(labeled_ind_train)) + " | Unalebled Nums: " + str(len(unlabeled_ind_train)))
        
        dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train, invalidList = invalidList,
        )
        
        trainloader_A, testloader = dataset.trainloader, dataset.testloader
        # labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train
        B_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        )
        trainloader_B, unlabeledloader = B_dataset.trainloader, B_dataset.unlabeledloader


    ## Save results
    with open(
            "log_AL/temperature_" + args.model + "_" + args.dataset + "_known" + str(args.known_class) + "_init" + str(
                    args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
                    args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
                    args.known_T) + "_modelB_T" + str(args.modelB_T) + ".pkl", 'wb') as f:
    
        data = {'Acc': Acc, 'Err': Err, 'Precision': Precision, 'Recall': Recall}
    
        pickle.dump(data, f)
    

    f.close()
    ## Save model
    #save_directory = "save_model"
    #if not os.path.exists(save_directory):
    #    os.makedirs(save_directory)
    
    #if args.is_mini:
    #     torch.save(model_B,
    #               "save_model/AL_center_" + args.dataset + "30_mini_query" + str(args.max_query) + "_batch" + str(
    #                   args.query_batch) + ".pt")
    #else:
    #    torch.save(model_B, "save_model/AL_center_" + args.dataset + "30_query" + str(args.max_query) + "_batch" + str(
    #        args.query_batch) + ".pt")
    


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def calculate_precision_recall():
    precision, recall = 0, 0
    return precision, recall


def train_A(model, criterion_xent, criterion_cent,
            optimizer_model, optimizer_centloss,
            trainloader, invalidList, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    if args.plot:
        all_features, all_labels = [], []

    known_T = args.known_T
    unknown_T = args.unknown_T
    invalid_class = args.known_class
    for batch_idx, (index, (data, labels)) in enumerate(trainloader):

        # Reduce temperature
        T = torch.tensor([known_T] * labels.shape[0], dtype=float)

        '''
        for i in range(len(labels)):
            # Annotate "unknown"
            if index[i] in invalidList:
                labels[i] = invalid_class
                T[i] = unknown_T
        '''

        if use_gpu:
            data, labels, T = data.cuda(), labels.cuda(), T.cuda()

        features, outputs = model(data)
        outputs = outputs / T.unsqueeze(1)
        loss_xent = criterion_xent(outputs, labels)
        # loss_cent = criterion_cent(features, labels)
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
        # cent_losses.update(loss_cent.item(), labels.size(0))

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
        plot_features(all_features, all_labels, num_classes, epoch, prefix='train')


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


def train_A_open_max(model, criterion_xent, criterion_cent,
            optimizer_model, optimizer_centloss,
            trainloader, invalidList, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    if args.plot:
        all_features, all_labels = [], []

    known_T = args.known_T
    unknown_T = args.unknown_T
    invalid_class = args.known_class
    for batch_idx, (index, (data, labels)) in enumerate(trainloader):

        # Reduce temperature
        # T = torch.tensor([known_T] * labels.shape[0], dtype=float)

        '''
        for i in range(len(labels)):
            # Annotate "unknown"
            if index[i] in invalidList:
                labels[i] = invalid_class
                T[i] = unknown_T
        '''

        if use_gpu:
            # data, labels, T = data.cuda(), labels.cuda(), T.cuda()
            data, labels = data.cuda(), labels.cuda()


        features, outputs = model(data)
        # outputs = outputs / T.unsqueeze(1)
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
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
        cent_losses.update(loss_cent.item(), labels.size(0))

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
        plot_features(all_features, all_labels, num_classes, epoch, prefix='train')


def train_B_open_max(model, criterion_xent, criterion_cent,
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
        loss_cent = criterion_cent(features, labels)
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
        cent_losses.update(loss_cent.item(), labels.size(0))

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


def get_image_label(index, dataset):
    if index < 0 or index >= len(dataset):
        raise ValueError("Invalid index: must be between 0 and {} (inclusive).".format(len(dataset) - 1))

    _, (_, label) = dataset[index]
    return label

def parse_val_annotations_index(root):
    annotation_file = os.path.join(root, "val", "val_annotations.txt")
    image_to_label = {}
    with open(annotation_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            image_number = int(re.search(r'\d+', parts[0]).group())
            image_to_label[image_number] = parts[1]
    return image_to_label

def parse_val_annotations(root):
    annotation_file = os.path.join(root, "val", "val_annotations.txt")
    image_to_label = {}
    with open(annotation_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            image_to_label[parts[0]] = parts[1]
    return image_to_label

def test(model, testloader, use_gpu, num_classes, epoch, dataset):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []
    # especially for loading tinyimage net
    if dataset == 'Tiny-Imagenet':
        root = './data/tiny-imagenet-200'
        index_to_label = parse_val_annotations_index(root)
        image_to_label = parse_val_annotations(root)
        label_to_index = {label: index for index, label in enumerate(sorted(set(image_to_label.values())))}
        # label_to_index = {label: index for index, label in enumerate(sorted(set(image_to_label.values())))}
    #
    with torch.no_grad():
        for batch_idx, (index, (data, labels)) in enumerate(testloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            if dataset == 'Tiny-Imagenet':  # Add this condition to handle Tiny ImageNet dataset
                labels = []
                for i in range(len(index)):
                    labels_in_tiny_format = index_to_label[index[i].item()]
                    true_label = label_to_index[labels_in_tiny_format]
                    labels.append(true_label)
                if use_gpu:
                    labels_tensor = torch.tensor(labels, dtype=torch.long)  # Convert the list to a PyTorch tensor
                    labels = labels_tensor.cuda()

            # print("Batch:", batch_idx, "Labels:", labels)  # Add this line to print labels
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