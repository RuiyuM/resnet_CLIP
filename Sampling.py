import numpy as np
from sklearn.mixture import GaussianMixture
import torch

import torch.nn.functional as F


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


def Max_AV_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    All_Arr = []
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


# unlabeledloader is int 800
def test_query(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labeled_ind_train, invalidList,
               indices, sel_idx):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []

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

    for tmp_class in S_per_class:

        if tmp_class == args.known_class:
            continue

        S_per_class[tmp_class] = np.array(S_per_class[tmp_class])

        for i in range(S_per_class[tmp_class].shape[0]):

            current_index = S_per_class[tmp_class][i][2]

            current_predict = S_per_class[tmp_class][i][1]

            current_value = S_per_class[tmp_class][i][0]

            # true_label      = S_per_class[tmp_class][i][3]

            count_known = 0.0
            count_unknown = 0.0
            index_Neighbor = index_knn[current_index]

            # known 的情况
            if tmp_class < 20:

                for k in range(len(index_Neighbor[0])):

                    n_index = (index_Neighbor[0][k]).item()

                    if n_index in labeled_ind_train:
                        count_known += 1

                    elif n_index in invalidList:
                        count_unknown += 1

                    else:

                        if S_index[n_index][0][1] < 20:
                            count_known += 1
                        else:
                            count_unknown += 1

                # if true_label not in neighbor_unknown:
                #    neighbor_unknown[true_label] = []

            # neighbor_unknown[true_label].append(count_unknown)

            if count_unknown < 5:

                queryIndex.append([current_index, S_per_class[tmp_class][i]])

            else:
                detected_unknown += 1

    print("detected_unknown: ", detected_unknown)
    print("\n")
    # for key, value in neighbor_unknown.items():
    #    print (key, sum(value)/len(value))

    queryIndex = sorted(queryIndex, key=lambda x: x[1][0], reverse=True)
    # queryIndex_unknown = sorted(queryIndex_unknown, key=lambda x: x[1][1], reverse=True)

    # 取前1500个index
    # final_chosen_index = [item[1][0] for item in queryIndex[:1500]]
    final_chosen_index = []
    invalid_index = []

    for item in queryIndex[:args.query_batch]:
        num = item[0]
        num3 = item[1][-1]

        if num3 < args.known_class:
            final_chosen_index.append(int(num))

        elif num3 >= args.known_class:
            invalid_index.append(int(num))

    precision = len(final_chosen_index) / args.query_batch
    # print(len(queryIndex_unknown))

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall


# unlabeledloader is int 800
# sel_idx, indices 的每一个element的index是 sel_idx
def score_query(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labeled_ind_train, invalidList,
                indices, sel_idx, values, err_A, known_label_index_dict):
    # err_A 不直接是 noisy transition matrix 但是是noisy label = j true label ！= j的概率
    # 这个值原本通过 noisy transition matrix 计算得来，这里做了简单调换，把模型对known 和 unknown的预测正确率拿出来。

    err_A = err_A / 100
    value_np = values.numpy()
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []

    S_per_class = {}
    S_index = {}

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)
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

    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # 下面这部分的for loop 主要是把extract feature 里面的load 出来的图片index和上面train set load 出来的index对上。
    # 比如extract feature 里面的的index顺序是[1, 2, 3, 4, 5] 然后train set load的index顺序是[5，4，2，1， 3]，下面的方法需要用到
    # predicted label 这样才可以把extract feature 里面的knn的k个image打上label。然后这个label就可以作为noisy label 然后进行后续的算法。

    current_index_order = []
    values_after_query = []
    # knn_neighbor_predicted_label
    knn_labels = []
    pretend_noisy_label = []
    # 下面这个循环我的意思是：首先因为extract feature 出来的index的顺序和 上面load出来的index不相符
    # 我把上面load出来的index顺序转变为extract feature 的index顺序。比如说第一轮load了49200 个数据，然后extract feature里面有50000 个数据，把这些index 对应起来
    #
    for i in range(len(sel_idx)):
        current_index = sel_idx[i]
        # 如果这个extract feature里面的index也出现在了上面train set load出来的index就进行下面的操作：
        if current_index in S_index:
            # known_label_index_dict 这个是一个key是index，label是true label的dict
            if current_index in known_label_index_dict:
                label_in_s_index = known_label_index_dict[0]
            else:
                label_in_s_index = S_index[current_index][0][1]
            knn_index = indices[i]
            current_near_label = []
            for i in range(len(knn_index)):
                current_knn_index = knn_index[i].item()
                if current_knn_index in S_index:

                    current_knn_label = S_index[current_knn_index][0][1]
                else:
                    current_knn_label = known_label_index_dict[current_knn_index][0]
                current_near_label.append(current_knn_label)
            # knn label 是每个图片附近的10个点的predicted label是什么，

            # load出来的数据的predicted label是known class 才放入特定的list，然后继续后续的计算算出score
            if label_in_s_index < args.known_class:
                knn_labels.append(current_near_label)
                pretend_noisy_label.append(label_in_s_index)
                # 新的index的顺序，和sel_idx 顺序一致但是剔除了label过的数据
                current_index_order.append(current_index)
                # knn距离
                values_after_query.append(value_np[i].tolist())

    # predicted_label_matched_with_sel_index 这个variable 是load出来的image的predicted label
    pretend_noisy_label = np.array(pretend_noisy_label)
    pretend_noisy_label = torch.from_numpy(pretend_noisy_label)
    # knn_labels_np 是每一个image附近10个image的predicted label
    knn_labels_np = np.array(knn_labels)
    values_after_query = np.array(values_after_query)
    # convert the NumPy array to a PyTorch tensor
    knn_labels_tensor = torch.from_numpy(knn_labels_np)
    predicted_label_matched_with_sel_index = torch.tensor(knn_labels_tensor)
    # values是extract feature里面计算出来的knn附近点的距离
    values = torch.from_numpy(values_after_query)
    # 下面这个部分就是把knn的距离变成到不同class的prob
    knn_labels_cnt = torch.zeros(len(current_index_order), args.known_class)
    for i in range(args.known_class):
        # 这一步是这么求的，首先总共有20个class，从第一个class开始，找到knn_labels里面等于i的label对应的value，然后用1 - 这个value，然后加起来。
        test_variable_1 = 1.0 - values
        test_variable_2 = (predicted_label_matched_with_sel_index == i)
        knn_labels_cnt[:, i] += torch.sum((test_variable_1) * (test_variable_2), 1)
    knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
    '''The NLL loss will be low when the predicted probabilities for the correct labels are high, and it will be high 
    when the predicted probabilities for the correct labels are low. Thus, minimizing the NLL loss encourages the 
    model to produce high probabilities for the correct labels, which leads to better classification performance. '''
    score = F.nll_loss(torch.log(knn_labels_prob + 1e-8), pretend_noisy_label, reduction='none')
    score_np = score.cpu().numpy()
    # python list 长度20 每一个element 是一个[], 把上面算出来的socre根据predicted的label分别放入list_store_score_according_to_class
    list_store_score_according_to_class = []
    for i in range(args.known_class):
        list_store_score_according_to_class.append([])
    # 把求出来的score，根据不同的class放到list_store_score_according_to_class
    for i in range(len(score_np)):
        current_score_index = current_index_order[i]
        if current_score_index in labeled_ind_train or current_score_index in invalidList:
            continue
        current_score_label = pretend_noisy_label[i].item()
        if current_score_label == args.known_class:
            continue
        list_store_score_according_to_class[current_score_label].append(score_np[i])


    queryIndex = []
    queryIndex_unknown = []
    index_knn = {}
# 这个index_knn的dictionary就是说把index 还有它附近的10个点的index对应起来
    for i in range(len(sel_idx)):

        if sel_idx[i] not in index_knn:
            index_knn[sel_idx[i]] = []

        index_knn[sel_idx[i]].append(indices[i])

    # index score dict
    # 把score和index对照起来
    index_score = {}
    for i in range(len(current_index_order)):
        if current_index_order[i] not in index_score:
            index_score[current_index_order[i]] = []
            index_score[current_index_order[i]].append(score_np[i])

    for tmp_class in S_per_class:

        if tmp_class == args.known_class:
            continue

        thre_noise_rate_per_class = 1 - err_A
        score_ndarray = np.array(list_store_score_according_to_class[tmp_class])
        # thre 就是某一个class的所有score里面，占右边比例的score的值
        thre = np.percentile(score_ndarray, 100 * (1 - thre_noise_rate_per_class.item()))
        for i in range(len(S_per_class[tmp_class])):

            current_index = S_per_class[tmp_class][i][2]

            current_predict = S_per_class[tmp_class][i][1]

            current_value = S_per_class[tmp_class][i][0]

            true_label = S_per_class[tmp_class][i][3]

            current_score = index_score[current_index]
            # 原来的paper任务大于thre的就是noisy的label 那我就取小于的就是不noisy的，也就是predicted是known，实际也是known
            if current_score >= thre:
                queryIndex.append([current_index, true_label, current_index, current_score[0]])

    # for key, value in neighbor_unknown.items():
    #    print (key, sum(value)/len(value))
    # sort according to the score
    queryIndex = sorted(queryIndex, key=lambda x: x[3], reverse=True)
    # queryIndex_unknown = sorted(queryIndex_unknown, key=lambda x: x[1][1], reverse=True)

    # 取前1500个index
    # final_chosen_index = [item[1][0] for item in queryIndex[:1500]]
    final_chosen_index = []
    invalid_index = []

    valid_label = []

    for item in queryIndex[:args.query_batch]:
        num = item[2]
        num3 = item[1]

        if num3 < args.known_class:
            final_chosen_index.append(int(num))

        elif num3 >= args.known_class:
            invalid_index.append(int(num))

    # put the label into a list and store it for later use
    for i in range(len(final_chosen_index)):
        labels_current = S_index[final_chosen_index[i]][0][1]
        valid_label.append(labels_current)
    #
    precision = len(final_chosen_index) / args.query_batch
    # print(len(queryIndex_unknown))

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall, valid_label
