import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_performance_cifar100():
    known = 20
    init = 8
    model = "resnet18"
    seeds = [1]

    temperature_acc = []

    temperature_precision = []

    temperature_recall = []

    MQ_acc = []

    MQ_precision = []

    MQ_recall = []

    New_MQ_acc = []

    New_MQ_precision = []

    New_MQ_recall = []

    for seed in seeds:
        # with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_random.pkl", 'rb') as f:
        #     data = pickle.load(f)
        #     random_acc.append([data['Acc'][i] for i in data['Acc']])
        #     random_precision.append([data['Precision'][i] for i in data['Precision']])
        #     random_recall.append([data['Recall'][i] for i in data['Recall']])
        # f.close()
        # with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_uncertainty.pkl", 'rb') as f:
        #     data = pickle.load(f)
        #     uncertainty_acc.append([data['Acc'][i] for i in data['Acc']])
        #     uncertainty_precision.append([data['Precision'][i] for i in data['Precision']])
        #     uncertainty_recall.append([data['Recall'][i] for i in data['Recall']])
        # f.close()
        # with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_based.pkl", 'rb') as f:
        #     data = pickle.load(f)
        #     AV_based_acc.append([data['Acc'][i] for i in data['Acc']])
        #     AV_based_precision.append([data['Precision'][i] for i in data['Precision']])
        #     AV_based_recall.append([data['Recall'][i] for i in data['Recall']])
        # f.close()
        # with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_uncertainty.pkl", 'rb') as f:
        #     data = pickle.load(f)
        #     max_av_acc.append([data['Acc'][i] for i in data['Acc']])
        #     max_av_precision.append([data['Precision'][i] for i in data['Precision']])
        #     max_av_recall.append([data['Recall'][i] for i in data['Recall']])
        # f.close()
        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.0.pkl", 'rb') as f:
            data = pickle.load(f)
            temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_My_Query_Strategy_unknown_T0.5_known_T0.5_modelB_T1.0.pkl", 'rb') as f:
            data = pickle.load(f)
            MQ_acc.append([data['Acc'][i] for i in data['Acc']])
            MQ_precision.append([data['Precision'][i] for i in data['Precision']])
            MQ_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_New_My_Query_Strategy_unknown_T0.5_known_T0.5_modelB_T1.0.pkl", 'rb') as f:
            data = pickle.load(f)
            New_MQ_acc.append([data['Acc'][i] for i in data['Acc']])
            New_MQ_precision.append([data['Precision'][i] for i in data['Precision']])
            New_MQ_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
    x = list(range(10))
    plt.figure()
    plt.title("Recall")
    plt.plot(x, np.array(temperature_recall).mean(0), label='AV_temperature_framework')
    plt.plot(x, np.array(MQ_recall).mean(0), label='MQ')
    plt.plot(x, np.array(New_MQ_recall).mean(0), label='New_MQ')

    plt.fill_between(x, np.array(MQ_recall).mean(0) - np.array(MQ_recall).std(0),
                     np.array(MQ_recall).mean(0) + np.array(MQ_recall).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(temperature_recall).mean(0) - np.array(temperature_recall).std(0),
                     np.array(temperature_recall).mean(0) + np.array(temperature_recall).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(New_MQ_recall).mean(0) - np.array(New_MQ_recall).std(0),
                     np.array(New_MQ_recall).mean(0) + np.array(New_MQ_recall).std(0),
                     color='k',
                     alpha=0.2)

    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_"+model+"_init"+str(init)+"_known"+str(known)+"_recall.png")
    plt.show()

    plt.figure()
    plt.title("Precision")

    plt.plot(x, np.array(temperature_precision).mean(0), label='AV_temperature_framework')
    plt.plot(x, np.array(MQ_precision).mean(0), label='MQ')
    plt.plot(x, np.array(New_MQ_precision).mean(0), label='New_MQ')

    plt.fill_between(x, np.array(temperature_precision).mean(0) - np.array(temperature_precision).std(0),
                     np.array(temperature_precision).mean(0) + np.array(temperature_precision).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(MQ_precision).mean(0) - np.array(MQ_precision).std(0),
                     np.array(MQ_precision).mean(0) + np.array(MQ_precision).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(New_MQ_precision).mean(0) - np.array(New_MQ_precision).std(0),
                     np.array(New_MQ_precision).mean(0) + np.array(New_MQ_precision).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_"+model+"_init"+str(init)+"_known"+str(known)+"_precision.png")
    plt.show()

    plt.figure()
    plt.title("Acc")

    plt.plot(x, np.array(temperature_acc).mean(0), label='AV_temperature_framework')
    plt.plot(x, np.array(MQ_acc).mean(0), label='MQ')
    plt.plot(x, np.array(MQ_acc).mean(0), label='New_MQ')

    plt.fill_between(x, np.array(temperature_acc).mean(0) - np.array(temperature_acc).std(0),
                     np.array(temperature_acc).mean(0) + np.array(temperature_acc).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(MQ_acc).mean(0) - np.array(MQ_acc).std(0),
                     np.array(MQ_acc).mean(0) + np.array(MQ_acc).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(New_MQ_acc).mean(0) - np.array(New_MQ_acc).std(0),
                     np.array(New_MQ_acc).mean(0) + np.array(New_MQ_acc).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_"+model+"_init"+str(init)+"_known"+str(known)+"_accuracy.png")
    plt.show()

if __name__ == "__main__":

    x = plot_performance_cifar100()