import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
from collections import Counter
from torch.nn.functional import softmax
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans

def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        queryIndex += index
        labelArr += list(np.array(labels.data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def uncertainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        features, outputs = model(data)

        uncertaintyArr += list(
            np.array((-torch.softmax(outputs, 1) * torch.log(torch.softmax(outputs, 1))).sum(1).cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((uncertaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def certainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    certaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        features, outputs = model(data)

        certaintyArr += list(
            # Certainty(P) = max(P(x))
            np.array(torch.softmax(outputs, 1).max(1).values.cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((certaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(-tmp_data[:, 0])]  # Use negative sign to sort in descending order
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def Max_AV_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    All_Arr = []
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        _, outputs = model(data)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            All_Arr.append([tmp_value, tmp_index, tmp_label])

    tmp_data = np.array(All_Arr)
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def AV_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        _, outputs = model(data)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        if len(activation_value) < 2:
            continue
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        # 得到为known类别的概率
        prob = prob[:, gmm.means_.argmax()]

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)
    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def AV_uncertainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        _, outputs = model(data)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        uncertainty = -(F.softmax(outputs) * F.log_softmax(outputs)).sum(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_uncertainty = np.array(uncertainty.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i] * tmp_uncertainty
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        # 得到为known类别的概率
        prob = prob[:, gmm.means_.argmax()]

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)
    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def AV_sampling2(args, labeledloader, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    S_ij_unlabeled = {}
    for batch_idx, (index, (data, labels)) in enumerate(labeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        _, outputs = model(data)
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij_unlabeled:
                S_ij_unlabeled[tmp_class] = []
            S_ij_unlabeled[tmp_class].append([tmp_value, tmp_index, tmp_label])
        if batch_idx > 10: break

    # fit a one-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij_unlabeled:
        if tmp_class not in S_ij:
            continue
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        S_ij_unlabeled[tmp_class] = np.array(S_ij_unlabeled[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        # print(tmp_class)
        activation_value_unlabeled = S_ij_unlabeled[tmp_class][:, 0]
        gmm = GaussianMixture(n_components=1, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        # 预测unlabeledloader为known类别的概率
        prob = gmm.predict_proba(np.array(activation_value_unlabeled).reshape(-1, 1))

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij_unlabeled[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij_unlabeled[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)
    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def VAE_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    All_Arr = []
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

        dec, _ = model(data.view(-1, 3 * 32 * 32))
        # define normal distribution
        p = torch.distributions.Normal(torch.zeros_like(model.z_mean), torch.ones_like(model.z_sigma))
        log_pz = p.log_prob(model.z)
        # get prob
        pz = np.exp(log_pz.sum(1).detach().cpu())
        # print(pz)
        # print(pz.shape)
        # print(ca)

        for i in range(len(labels.data.cpu())):
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(pz)[i]
            All_Arr.append([tmp_value, tmp_index, tmp_label])

    tmp_data = np.array(All_Arr)
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


# unlabeledloader is int 800
def AV_sampling_temperature(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        _, outputs = model(data)
        # 当前的index 128 个 进入queryIndex array
        queryIndex += index
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])
    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        if len(activation_value) < 2:
            continue
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        # 得到为known类别的概率
        prob = prob[:, gmm.means_.argmax()]
        # 如果为unknown类别直接为0
        if tmp_class == args.known_class:
            prob = [0] * len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            # np.hstack 就是说把stack水平堆起来
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            # np。vstack 就是把stack竖直堆起来
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    
    # 取前1500个index
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)
    
    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def My_query(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    unlabel_index_arr = []
    label_index_arr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        _, outputs = model(data)
        outputs = outputs.cpu().data.numpy()
        test = outputs.T
        top_indices = np.argsort(test, axis=1)[:, -1:]
        top_values = np.take_along_axis(test, top_indices, axis=1)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

        my_dict = {num: [] for num in range(128)}
        num_rows = top_indices.shape[0]
        for number_of_row in range(num_rows):
            my_dict[top_indices[number_of_row][0]].append(number_of_row)
        key_with_multi_element = []
        for key, value in my_dict.items():
            if len(value) > 1:
                key_with_multi_element.append(key)
        for key in key_with_multi_element:
            key_with_max = outputs[key][0]
            max_index = my_dict[key][0]
            for i in range(1, len(my_dict[key])):
                if outputs[key][i] > key_with_max:
                    key_with_max = my_dict[key][i]
                    max_index = my_dict[key][i]
            my_dict[key] = [max_index]
        for key, value in my_dict.items():
            if len(value) > 0:
                tmp_class = value[0]
                tmp_label = labelArr[key]
                tmp_index = queryIndex[key]
                tmp_value = outputs[key][value[0]]
                if tmp_class not in S_ij:
                    S_ij[tmp_class] = []
                S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])
            else:
                unlabel_index_arr.append(queryIndex[key])
        # 当前的index 128 个 进入queryIndex array
        # queryIndex += index
        # # my_test_for_outputs = outputs.cpu().data.numpy()
        # # print(my_test_for_outputs)
        # # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        # labelArr += list(np.array(labels.cpu().data))
        # # activation value based
        # # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        # v_ij, predicted = outputs.max(1)
        # for i in range(len(predicted.data)):
        #     tmp_class = np.array(predicted.data.cpu())[i]
        #     tmp_index = index[i]
        #     tmp_label = np.array(labels.data.cpu())[i]
        #     tmp_value = np.array(v_ij.data.cpu())[i]
        #     if tmp_class not in S_ij:
        #         S_ij[tmp_class] = []
        #     S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])
    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        # if len(activation_value) < 2:
        #     continue
        # gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        # gmm.fit(np.array(activation_value).reshape(-1, 1))
        # prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        # 得到为known类别的概率
        prob = activation_value
        # 如果为unknown类别直接为0
        if tmp_class == args.known_class:
            prob = [0] * len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            # np.hstack 就是说把stack水平堆起来
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            # np。vstack 就是把stack竖直堆起来
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    # 取前1500个index
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)
    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def My_Query_Strategy(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labeled_ind_train, invalidList,
                      indices, sel_idx):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)
        # 当前的index 128 个 进入queryIndex array
        queryIndex += index
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i].item()
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_index not in S_ij:
                S_ij[tmp_index] = []
            S_ij[tmp_index].append([tmp_class, tmp_value, tmp_label])
    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # my_sampling

    # Resnet : 预测出unknown，附近的10个点中的6个及以上都是unknown就铁定是unkown
    #               Else：known
    #
    # Resnet: 预测出known，附件的10个点中的5个及其以上都是known就是认为是known
    #           else：unknown
    # 创建一个dict，key是index，value是十个长度的邻居point。

    # queryIndex 存放known class的地方
    queryIndex = []
    queryIndex_unknown = []
    index_knn = {}
    for i in range(len(sel_idx)):
        if sel_idx[i] not in index_knn:
            index_knn[sel_idx[i]] = []
        index_knn[sel_idx[i]].append(indices[i])

    # known_indicator = False
    # unknown_indicator = True
    # tmp_class, tmp_value, tmp_label
    # 预测的class， activationvalue， 真实的class
    for key, value in S_ij.items():
        count_known = 0
        count_unknown = 0
        index_Neighbor = index_knn[key]


        # known 的情况
        if value[0][0] < 20:
            for i in range(len(index_Neighbor[0])):
                current_index = (index_Neighbor[0][i]).item()
                if current_index in labeled_ind_train:
                    count_known += 1
                elif current_index in invalidList:
                    count_unknown += 1
                else:
                    if S_ij[current_index][0][0] < 20:
                        count_known += 1
                    else:
                        count_unknown += 1
            
            if count_known >= 6:
                queryIndex.append([key, value[0]])
        

        # 假设20个known class 那么第21位就是unknown
        if value[0][0] == 20:
            for i in range(len(index_Neighbor[0])):
                current_index = (index_Neighbor[0][i]).item()
                if current_index in labeled_ind_train:
                    count_known += 1
                elif current_index in invalidList:
                    count_unknown += 1
                else:
                    if S_ij[current_index][0][0] < 20:
                        count_known += 1
                    else:
                        count_unknown += 1
            
            if count_unknown >= 4:
                queryIndex_unknown.append([key, value[0]])
    

    queryIndex = sorted(queryIndex, key=lambda x: x[1][1], reverse=True)
    queryIndex_unknown = sorted(queryIndex_unknown, key=lambda x: x[1][1], reverse=True)


    print("queryIndex: ", len(queryIndex))
    # 取前1500个index
    # final_chosen_index = [item[1][0] for item in queryIndex[:1500]]
    final_chosen_index = []
    invalid_index = []
    for item in queryIndex[:1500]:
        num = item[0]
        num3 = item[1][2]

        if num3 < args.known_class:
            final_chosen_index.append(num)
        elif num3 >= args.known_class:
            invalid_index.append(num)

    '''
    if len(queryIndex_unknown) > 1000:
        for item in queryIndex_unknown[:1000]:
            num = item[0]
            num3 = item[1][2]

            if num3 < args.known_class:
                final_chosen_index.append(num)
            elif num3 >= args.known_class:
                invalid_index.append(num)
    else:
        for item in queryIndex_unknown:
            num = item[0]
            num3 = item[1][2]

            if num3 < args.known_class:
                final_chosen_index.append(num)
            elif num3 >= args.known_class:
                invalid_index.append(num)
    '''

    precision = len(final_chosen_index) / 1500
    print(len(queryIndex_unknown))
    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)
    return final_chosen_index, invalid_index, precision, recall



def knn_prediction(k_neighbors, S_index):

    neighbor_labels = []

    for k in k_neighbors:

        label = S_index[k][1][2]

        neighbor_labels.append(label)


    x = Counter(neighbor_labels)
    #print (x)
    top_1 = x.most_common(1)[0][0]

    return top_1

def active_learning(index_knn, queryIndex, S_index):

    print ("active learning")
    #S_index[n_index][0][1]

    for i in range(len(queryIndex)):

        neighbors_prediction = []

        # all the indices for neighbors
        neighbors = index_knn[queryIndex[i][0]]

        change = 0.0
        cur_predict = None

        for j in range(1, len(neighbors)):

            # current knn prediction
            k_predict = knn_prediction(neighbors[:j], S_index)

            if not cur_predict:
                cur_predict = k_predict

            else:
                if cur_predict != k_predict:
                    change += 1

                cur_predict = k_predict

        # how many changes
        score = change / len(neighbors)

        queryIndex[i][1] = np.append(queryIndex[i][1], score)


    return queryIndex


# unlabeledloader is int 800
def test_query(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labeled_ind_train, invalidList,
                      indices, sel_idx):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []

    #################################################################

    S_per_class = {}
    S_index = {}

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)
        # 当前的index 128 个 进入queryIndex array
        queryIndex += index
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)

        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i].item()

            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            
            if tmp_class not in S_per_class:
                S_per_class[tmp_class] = []

            S_per_class[tmp_class].append([tmp_value, tmp_class, tmp_index, tmp_label])

            if tmp_index not in S_index:
                S_index[tmp_index] = []
            
            S_index[tmp_index].append([tmp_value, tmp_class, tmp_label])

    #################################################################

    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # queryIndex 存放known class的地方
    queryIndex = []
    queryIndex_unknown = []
    index_knn = {}

    for i in range(len(sel_idx)):
        
        if sel_idx[i] not in index_knn:
            index_knn[sel_idx[i]] = []

        index_knn[sel_idx[i]].append(indices[i])


    neighbor_unknown = {}

    detected_unknown = 0.0
    detected_known = 0.0


    for tmp_class in S_per_class:

        if tmp_class == args.known_class:
            continue

        S_per_class[tmp_class] = np.array(S_per_class[tmp_class])


        for i in range(S_per_class[tmp_class].shape[0]):

            current_index   = S_per_class[tmp_class][i][2]

            current_predict = S_per_class[tmp_class][i][1]

            current_value   = S_per_class[tmp_class][i][0]
            
            count_known = 0.0
            count_unknown = 0.0

            index_Neighbor = index_knn[current_index] 

            for k in range(len(index_Neighbor[0])):

                n_index = (index_Neighbor[0][k]).item()
                

                if n_index in labeled_ind_train:
                    count_known += 1
            
                elif n_index in invalidList:
                    count_unknown += 1
                
                '''
                else:
                    if S_index[n_index][0][1] < 20:
                        count_known += 1
                    else:
                        count_unknown += 1
                '''

            # known 的情况
            if tmp_class < 20:

                if count_unknown < count_known:

                    queryIndex.append([current_index, S_per_class[tmp_class][i]])               

                else:
                    detected_unknown += 1         


    print ("detected_unknown: ", detected_unknown)
    print ("\n")
    #for key, value in neighbor_unknown.items():
    #    print (key, sum(value)/len(value))

    queryIndex = sorted(queryIndex, key=lambda x: x[1][0], reverse=True)

    #################################################################
    if args.active:

        queryIndex = queryIndex[:2*args.query_batch]

        #################################################################
        queryIndex = active_learning(index_knn, queryIndex, S_index)

        queryIndex = sorted(queryIndex, key=lambda x: x[1][-1], reverse=True)
    

    #################################################################

    # 取前1500个index
    # final_chosen_index = [item[1][0] for item in queryIndex[:1500]]
    

    final_chosen_index = []
    invalid_index = []

    for item in queryIndex[:args.query_batch]:
        num = item[0]

        if args.active:

            num3 = item[1][-2]

        else:

            num3 = item[1][-1]


        if num3 < args.known_class:
            final_chosen_index.append(int(num))
        
        elif num3 >= args.known_class:
            invalid_index.append(int(num))




    precision = len(final_chosen_index) / args.query_batch
    #print(len(queryIndex_unknown))
    
    #recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)
    
    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall

def bayesian_generative_active_learning(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertainty_scores = []

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            _, outputs = model(data)


        queryIndex += list(np.array(index.cpu().data))
        labelArr += list(np.array(labels.cpu().data))

        _, predicted = outputs.max(1)
        proba_out = softmax(outputs, dim=1)
        proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

        uncertainty = 1 - proba_out.squeeze().cpu().numpy()
        uncertainty_scores += list(uncertainty)

    uncertainty_scores = np.array(uncertainty_scores)
    sorted_indices = np.argsort(-uncertainty_scores)
    selected_indices = sorted_indices[:args.query_batch]

    query_indices = np.array(queryIndex)[selected_indices]
    query_labels = np.array(labelArr)[selected_indices]

    known_indices = query_indices[query_labels < args.known_class]
    unknown_indices = query_indices[query_labels >= args.known_class]

    precision = len(known_indices) / len(query_indices)
    recall = (len(known_indices) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return known_indices, unknown_indices, precision, recall

def core_set(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    embedding_vectors = []
    S_per_class = {}
    S_index = {}
    labelArr = []
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            embeddings, outputs = model(data)
        # 当前的index 128 个 进入queryIndex array
        queryIndex += list(np.array(index.cpu().data))
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)
        embedding_vectors += list(np.array(embeddings.cpu().data))
        # proba_out = torch.nn.functional.softmax(outputs, dim=1)

        # proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i].item()

            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]

            if tmp_class not in S_per_class:
                S_per_class[tmp_class] = []

            S_per_class[tmp_class].append([tmp_value, tmp_class, tmp_index, tmp_label])

            if tmp_index not in S_index:
                S_index[tmp_index] = []

            S_index[tmp_index].append([tmp_value, tmp_class, tmp_label])
    kmeans = KMeans(n_clusters=args.query_batch)
    kmeans.fit(embedding_vectors)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embedding_vectors)

    selected_indices = [queryIndex[i] for i in closest]

    final_chosen_index = []
    invalid_index = []

    for i in range(len(selected_indices)):
        if S_index[selected_indices[i]][0][2] < args.known_class:
            final_chosen_index.append(selected_indices[i])
        elif S_index[selected_indices[i]][0][2] >= args.known_class:
            invalid_index.append(selected_indices[i])

    #
    precision = len(final_chosen_index) / args.query_batch
    # print(len(queryIndex_unknown))

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall


def openmax_sampling(args, unlabeledloader, Len_labeled_ind_train, model,criterion_cent, use_gpu, openmax_beta=0.5):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertainty_scores = []

    def compute_openmax_score(proba_out, distances, beta=openmax_beta):
        scores = (1 - proba_out) * np.exp(-beta * distances)
        return np.sum(scores, axis=1)

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            features, outputs = model(data)

        queryIndex += list(np.array(index.cpu().data))
        labelArr += list(np.array(labels.cpu().data))

        _, predicted = outputs.max(1)
        proba_out = softmax(outputs, dim=1)
        proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

        distances = np.array([euclidean(features[i].cpu().detach().numpy(),
                                        criterion_cent.centers[predicted[i].cpu()].cpu().detach().numpy())
                              for i in range(features.size(0))])

        uncertainty = compute_openmax_score(proba_out.cpu().numpy(), distances, openmax_beta)
        uncertainty_scores += list(uncertainty)

    uncertainty_scores = np.array(uncertainty_scores)
    sorted_indices = np.argsort(-uncertainty_scores)
    selected_indices = sorted_indices[:args.query_batch]

    query_indices = np.array(queryIndex)[selected_indices]
    query_labels = np.array(labelArr)[selected_indices]

    known_indices = query_indices[query_labels < args.known_class]
    unknown_indices = query_indices[query_labels >= args.known_class]

    precision = len(known_indices) / len(query_indices)
    recall = (len(known_indices) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return known_indices, unknown_indices, precision, recall
# def openmax_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, tail_size=20):
#     model.eval()
#
#     # Calculate MAVs and Weibull parameters
#     mavs = []
#     weibull_params = []
#     activation_vectors = [[] for _ in range(args.known_class + 1)]
#
#     for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
#         if use_gpu:
#             data, labels = data.cuda(), labels.cuda()
#         if args.dataset == 'mnist':
#             data = data.repeat(1, 3, 1, 1)
#
#         features, outputs = model(data)
#         _, predicted = outputs.max(1)
#
#         for i, label in enumerate(predicted):
#             # if label.item() < args.known_class:
#             activation_vectors[label.item()].append(features[i].detach().cpu().numpy())
#
#     for i in range(args.known_class + 1):
#         if activation_vectors[i]:
#             mavs.append(np.mean(activation_vectors[i], axis=0))
#             weibull_params.append(fit_weibull(mavs[i], activation_vectors[i], tail_size))
#         else:
#             mavs.append(np.array([0.01, 0.01]))
#
#             XB = np.array((0.01, 0.01)).reshape((1, 2))
#
#             weibull_params.append(fit_weibull(mavs[i], XB, tail_size))
#
#     # Perform OpenMax sampling
#     queryIndex = []
#     labelArr = []
#     openmax_scores = []
#
#     for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
#         if use_gpu:
#             data, labels = data.cuda(), labels.cuda()
#
#         features, outputs = model(data)
#         _, predicted = outputs.max(1)
#
#         for i, (feature, label) in enumerate(zip(features, labels)):
#             openmax_score = openmax(args.known_class, mavs, weibull_params, feature.detach().cpu().numpy(), tail_size)
#             queryIndex.append(index[i])
#             labelArr.append(label.item())
#             openmax_scores.append(openmax_score)
#
#     # Select samples based on OpenMax scores
#     sorted_indices = np.argsort(openmax_scores)[::-1]
#     queryIndex = np.array(queryIndex)[sorted_indices]
#     labelArr = np.array(labelArr)[sorted_indices]
#
#     queryLabelArr = labelArr[-args.query_batch:]
#     precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
#     recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
#             len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
#
#     return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
#         np.where(queryLabelArr >= args.known_class)[0]], precision, recall


from scipy.spatial.distance import cdist
from scipy.stats import weibull_min


def fit_weibull(mav, activation_vectors, tail_size):
    distances = cdist([mav], activation_vectors, metric='euclidean')[0]
    sorted_distances = np.sort(distances)
    tail_distances = sorted_distances[-tail_size:]

    # Fit a Weibull distribution to the tail distances
    weibull_params = {}
    weibull_params['shape'], weibull_params['loc'], weibull_params['scale'] = weibull_min.fit(tail_distances)
    return weibull_params


# def openmax(class_count, mavs, weibull_params, feature, tail_size, alpha=10):
#     mavs = np.array(mavs)  # Convert mavs list to a numpy array
#     distances = np.linalg.norm(mavs - feature, axis=1)  # Compute Euclidean distances element-wise
#     softmax_outputs = np.exp(-distances) / np.sum(np.exp(-distances))
#
#     # Compute OpenMax probabilities
#     weibull_probs = []
#     for i in range(class_count):
#         weibull_cdf = weibull_min.cdf(distances[i], weibull_params[i]['shape'], loc=weibull_params[i]['loc'],
#                                       scale=weibull_params[i]['scale'])
#         weibull_prob = 1 - weibull_cdf
#         weibull_probs.append(weibull_prob)
#
#     unknown_prob = 1 - np.sum(weibull_probs)
#     weibull_probs.append(unknown_prob)
#     openmax_probs = np.multiply(softmax_outputs, weibull_probs)
#
#     # Adjust the probabilities to include the unknown class
#     if unknown_prob > 0:
#         openmax_probs = (1 - alpha * unknown_prob) * openmax_probs
#         openmax_probs = np.append(openmax_probs, alpha * unknown_prob)
#
#     return openmax_probs[-1]  # Return the probability of the sample belonging to the unknown class
