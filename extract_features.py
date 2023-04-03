import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import clip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter


class CustomCIFAR100Dataset_train(Dataset):
    cifar100_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar100", train=True, download=True, transform=None):
        cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar100_dataset.targets

    def __init__(self):
        if CustomCIFAR100Dataset_train.cifar100_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR100Dataset_train.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR100Dataset_train.cifar100_dataset)


def CIFAR100_EXTRACT_FEATURE_CLIP(labeled_ind_train):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    model.eval()

    crop = transforms.RandomCrop(32, padding=4)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    preprocess_rand = transforms.Compose([crop, transforms.RandomHorizontalFlip(), preprocess])

    # train_data = datasets.CIFAR100('./data/', train=True, download=True, transform=preprocess_rand)
    CustomCIFAR100Dataset_train.load_dataset(transform=preprocess_rand)
    train_data = CustomCIFAR100Dataset_train()
    record = [[] for _ in range(100)]

    batch_size = 2048
    print('Data Loader')
    data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dict with key = index and value = label

    for batch_idx, (index, (data, labels)) in enumerate(data_loader):
        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            extracted_feature = model.encode_image(data)

        for i in range(extracted_feature.shape[0]):
            record[labels[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': (index[i]).item()})

    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])

    origin_label = torch.zeros(total_len).long()
    index_rec = np.zeros(total_len, dtype=int)
    cnt, lb = 0, 0

    for item in record:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            index_rec[cnt] = i['index']
            cnt += 1

        lb += 1

    data_set = {'feature': origin_trans[:cnt], 'label': origin_label[:cnt], 'index': index_rec[:cnt]}

    order_to_index = {}

    index_to_order = {}
    for i in range(data_set['index'].shape[0]):
        order_to_index[i] = data_set['index'][i]

        index_to_order[data_set['index'][i]] = i

    KINDS = 100
    all_point_cnt = data_set['feature'].shape[0]

    # sample = np.random.choice(np.arange(data_set['feature'].shape[0]), all_point_cnt, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature']  # [sample]
    sel_idx = data_set['index']  # [sample]
    labels = data_set['label']  # [sample]
    # dict with key = index and value = label
    index_label_dict = {index.item(): label.item() for index, label in zip(sel_idx, labels)}
    # dict match the known_label with index
    known_label_index_dict = {}
    for i in range(len(labeled_ind_train)):
        current_index = labeled_ind_train[i]
        if current_index not in known_label_index_dict:
            known_label_index_dict[current_index] = []
        known_label_index_dict[current_index].append(index_label_dict[current_index])


    Dist = cosDistance(final_feat)

    # min_similarity = 0.0
    values, indices = Dist.topk(k=10, dim=1, largest=False, sorted=True)

    values[:, 0] = 2.0 * values[:, 1] - values[:, 2]

    # score = F.nll_loss(torch.log(knn_labels_prob + 1e-8), labels, reduction='none')

    for k in range(indices.size()[0]):

        for j in range(indices.size()[1]):
            indices[k][j] = order_to_index[indices[k][j].item()]

    # X_train, X_test, y_train, y_test = train_test_split(final_feat, labels, test_size=0.33, random_state=42)

    # neigh = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    # neigh.fit(X_train, y_train)
    # print (neigh.score(X_test, y_test))

    return indices, sel_idx, values, known_label_index_dict# , labels, index_to_order


def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1)  # each feature's l2-norm should be 1
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix


if __name__ == "__main__":

    indices, sel_idx, labels, index_to_order = CIFAR100_EXTRACT_FEATURE_CLIP()

    index_knn = {}
    for i in range(len(sel_idx)):

        if sel_idx[i] not in index_knn:
            index_knn[sel_idx[i]] = []

        index_knn[sel_idx[i]].append(indices[i])

    print(index_knn)

    neighbor_unknown = {}

    correct = 0.0
    cnt = 0.0

    for i, (key, value) in enumerate(index_knn.items()):

        cnt += 1

        value = value[0].numpy()

        key_label = labels[i].item()

        # if key_label not in neighbor_unknown:
        #    neighbor_unknown[key_label] = []
        neighbor_labels = []

        for k in range(len(value)):
            neighbor_label = labels[index_to_order[value[k]]].item()

            neighbor_labels.append(neighbor_label)

        # print (neighbor_labels)

        x = Counter(neighbor_labels)

        # print (x)
        top_1 = x.most_common(1)[0][0]

        # print (top_1)
        # print ("\n")
        if key_label == top_1:
            correct += 1

        '''
        if neighbor_label >= 20:
            neighbor_label = 20
        if neighbor_label >= 20:
            c_uk += 1
            '''
        # neighbor_unknown[key_label].append(c_uk)

    print(correct / cnt)