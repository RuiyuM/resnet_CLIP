import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets
import random
import transforms

known_class = -1
init_percent = -1


class CustomCIFAR100Dataset_train(Dataset):
    # cifar100_dataset = None
    # targets = None

    # @classmethod
    # def load_dataset(cls, root="./data/cifar100", train=True, download=True, transform=None):
    #    cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
    #    cls.targets = cls.cifar100_dataset.targets

    def __init__(self, root="./data/cifar100", train=True, download=True, transform=None, invalidList=None):
        # if CustomCIFAR100Dataset_train.cifar100_dataset is None:
        #    raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

        self.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        self.targets = self.cifar100_dataset.targets

        if invalidList is not None:
            targets = np.array(self.cifar100_dataset.targets)
            targets[targets >= known_class] = known_class
            self.cifar100_dataset.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, label = self.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.cifar100_dataset)


class CustomCIFAR100Dataset_test(Dataset):
    cifar100_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar100", train=False, download=True, transform=None):
        cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar100_dataset.targets

    def __init__(self):
        if CustomCIFAR100Dataset_test.cifar100_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR100Dataset_test.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR100Dataset_test.cifar100_dataset)

class CustomMNISTDataset_train(Dataset):
    # cifar100_dataset = None
    # targets = None

    # @classmethod
    # def load_dataset(cls, root="./data/cifar100", train=True, download=True, transform=None):
    #    cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
    #    cls.targets = cls.cifar100_dataset.targets

    def __init__(self, root='./data/mnist', train=True, download=True, transform=None, invalidList=None):
        # if CustomCIFAR100Dataset_train.cifar100_dataset is None:
        #    raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download, transform=transform)
        self.targets = self.mnist_dataset.targets

        if invalidList is not None:
            targets = np.array(self.mnist_dataset.targets)
            targets[targets >= known_class] = known_class
            self.mnist_dataset.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, label = self.mnist_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.mnist_dataset)


class CustomMNISTDataset_test(Dataset):
    mnist_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root='./data/mnist', train=False, download=True, transform=None):
        cls.mnist_dataset = datasets.MNIST(root, train=train, download=download, transform=transform)
        cls.targets = cls.mnist_dataset.targets

    def __init__(self):
        if CustomMNISTDataset_test.mnist_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomMNISTDataset_test.mnist_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomMNISTDataset_test.mnist_dataset)

class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList

        trainset = CustomMNISTDataset_train(transform=transform, invalidList=invalidList)
        ## 初始化
        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        # testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
        CustomMNISTDataset_test.load_dataset(transform=transform)
        testset = CustomMNISTDataset_test()

        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        # 随机选
        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 1000]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 1000:]
        return labeled_ind, unlabeled_ind


class CIFAR100(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        pin_memory = True if use_gpu else False

        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList

        # CustomCIFAR100Dataset_train.load_dataset(transform=transform_train)
        # trainset = torchvision.datasets.CIFAR100("./data/cifar100", train=True, download=True,
        #                                          transform=transform_train)

        trainset = CustomCIFAR100Dataset_train(transform=transform_train, invalidList=invalidList)

        ## 初始化
        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:

            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")

            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )

            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        # testset = torchvision.datasets.CIFAR100("./data/cifar100", train=False, download=True, transform=transform_test)
        CustomCIFAR100Dataset_test.load_dataset(transform=transform_test)
        testset = CustomCIFAR100Dataset_test()

        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        # 随机选
        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
        return labeled_ind, unlabeled_ind


class CustomCIFAR10Dataset_train(Dataset):
    # cifar100_dataset = None
    # targets = None

    # @classmethod
    # def load_dataset(cls, root="./data/cifar100", train=True, download=True, transform=None):
    #    cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
    #    cls.targets = cls.cifar100_dataset.targets

    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None, invalidList=None):
        # if CustomCIFAR100Dataset_train.cifar100_dataset is None:
        #    raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

        self.cifar10_dataset = datasets.CIFAR10(root, train=train, download=download, transform=transform)
        self.targets = self.cifar10_dataset.targets

        if invalidList is not None:
            targets = np.array(self.cifar10_dataset.targets)
            targets[targets >= known_class] = known_class
            self.cifar10_dataset.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, label = self.cifar10_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.cifar10_dataset)


class CustomCIFAR10Dataset_test(Dataset):
    cifar10_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar10", train=False, download=True, transform=None):
        cls.cifar10_dataset = datasets.CIFAR10(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar10_dataset.targets

    def __init__(self):
        if CustomCIFAR10Dataset_test.cifar10_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR10Dataset_test.cifar10_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR10Dataset_test.cifar10_dataset)


class CIFAR10(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        pin_memory = True if use_gpu else False
        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList

        trainset = CustomCIFAR10Dataset_train(transform=transform_train, invalidList=invalidList)
        ## 初始化
        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        CustomCIFAR10Dataset_test.load_dataset(transform=transform_test)
        testset = CustomCIFAR10Dataset_test()

        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        # 随机选
        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
        return labeled_ind, unlabeled_ind


# TinyImageNet data set

class TinyImageNet(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        pin_memory = True if use_gpu else False

        trainset = CustomTinyImageNetDataset_train(transform=transform)

        if unlabeled_ind_train is None and labeled_ind_train is None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        CustomTinyImageNetDataset_test.load_dataset(transform=transform)
        testset = CustomTinyImageNetDataset_test()

        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 1000]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 1000:]
        return labeled_ind, unlabeled_ind


__factory = {
    'Tiny-Imagenet': TinyImageNet,
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
}


def create(name, known_class_, init_percent_, batch_size, use_gpu, num_workers, is_filter, is_mini, SEED,
           unlabeled_ind_train=None, labeled_ind_train=None, invalidList=None):
    global known_class, init_percent
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    known_class = known_class_
    init_percent = init_percent_

    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train, labeled_ind_train,
                           invalidList)