import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
from matplotlib import cm


datasets = {
    "Tiny-Imagenet": {
        "known_class": ['_pretrained_model_image50', '_pretrained_model_image18', '_pretrained_model_image34',
                        '_pretrained_model_clip'],
        "batch": [400]
    },
    "cifar10": {
        "known_class": ['_pretrained_model_image50', '_pretrained_model_image18', '_pretrained_model_image34', '_pretrained_model_clip'],
        "batch": [400]
    },
    "cifar100": {
        "known_class": ['_pretrained_model_image50', '_pretrained_model_image18', '_pretrained_model_image34',
                        '_pretrained_model_clip'],
        "batch": [400]
    }
}

def load_pkl_files(dataset_name, known_class, batch_size=None):
    pkl_files = {known_class: []}
    log_al_folder = 'log_AL/'
    for file in os.listdir(log_al_folder):
        if file.endswith('.pkl') and f"{dataset_name}_" in file and known_class in file:
            if batch_size is not None and f"batch{batch_size}_" not in file:
                continue
            if known_class == '_pretrained_model_clip':
                if '_neighbor_10' not in file:
                    continue
            if "_per_round_query_index" in file:
                continue
            pkl_files[known_class].append(os.path.join(log_al_folder, file))
    return pkl_files



def plot_graphs(group_name, acc_list, precision_list, recall_list, acc_std_list, precision_std_list, recall_std_list, batch_size, method_colors):
    method_colors = {datasets[dataset_name]['known_class'][i]: plt.cm.tab10(i) for dataset_name in datasets for i in
                     range(len(datasets[dataset_name]['known_class']))}

    # Define widths for each model type
    width_map = {}
    for dataset_name in datasets:
        for item in datasets[dataset_name]['known_class']:
            if item not in width_map:
                width_map[item] = 1
                if '_pretrained_model_clip' in item:
                    width_map[item] = 2

    # Accuracy plot
    query_numbers = list(range(len(acc_list[0])))

    fig, ax = plt.subplots()
    for i, (acc, acc_std) in enumerate(zip(acc_list, acc_std_list)):
        label = list(method_colors.keys())[i]
        ax.plot(query_numbers, acc, label=label, color=method_colors[label], linewidth=width_map[label])
        ax.fill_between(query_numbers, np.array(acc) - 0.8 * np.array(acc_std), np.array(acc) + 0.8 * np.array(acc_std),
                        alpha=0.2, color=method_colors[label])

    # Add a legend


    for tick in ax.get_yticks():
        ax.axhline(tick, linestyle='dashed', alpha=0.3, color='gray')

    ax.xaxis.set_ticks(range(len(query_numbers)))
    for tick in ax.get_xticks():
        ax.axvline(tick, linestyle='dashed', alpha=0.3, color='gray')

    ax.set_xlabel('Query Round', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'{group_name.split()[0]} Batch {batch_size}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    if group_name == "Tiny-Imagenet":
        # Create an inset axis
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper left', borderpad=2.5)
        half_length = len(query_numbers) // 2
        for i, (acc, acc_std) in enumerate(zip(acc_list, acc_std_list)):
            label = list(method_colors.keys())[i]

            ax_inset.plot(query_numbers[half_length:], acc[half_length:], label=label, color=method_colors[label],
                              linewidth=width_map[label])
            ax_inset.fill_between(query_numbers[half_length:],
                                      np.array(acc[half_length:]) - 0.8 * np.array(acc_std[half_length:]),
                                      np.array(acc[half_length:]) + 0.8 * np.array(acc_std[half_length:]), alpha=0.2,
                                      color=method_colors[label])

        ax_inset.set_xlim(7.0, query_numbers[-1])  # Set x-axis from 6.0 to the end
        ax_inset.set_ylim(44.5, 44.5 + 8)  # Set y-axis range from 36 to 42
        ax_inset.xaxis.set_ticklabels([])  # Remove x-axis numbers
        # ax_inset.yaxis.set_ticklabels([]) # Remove x-axis numbers
        ax_inset.tick_params(axis='both', which='major', labelsize=12)
        for tic in ax_inset.xaxis.get_major_ticks() + ax_inset.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    if group_name == "cifar100":
        # Create an inset axis
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper left', borderpad=2.5)
        half_length = len(query_numbers) // 2
        for i, (acc, acc_std) in enumerate(zip(acc_list, acc_std_list)):
            label = list(method_colors.keys())[i]
            ax_inset.plot(query_numbers[half_length:], acc[half_length:], label=label, color=method_colors[label],
                              linewidth=width_map[label])
            ax_inset.fill_between(query_numbers[half_length:],
                                      np.array(acc[half_length:]) - 0.8 * np.array(acc_std[half_length:]),
                                      np.array(acc[half_length:]) + 0.8 * np.array(acc_std[half_length:]), alpha=0.2,
                                      color=method_colors[label])

        ax_inset.set_xlim(7.0, query_numbers[-1])  # Set x-axis from 6.0 to the end
        ax_inset.set_ylim(57.5, 57.5 + 8)  # Set y-axis range from 36 to 42
        ax_inset.xaxis.set_ticklabels([])  # Remove x-axis numbers
        ax_inset.tick_params(axis='both', which='major', labelsize=12)
        # ax_inset.yaxis.set_ticklabels([]) # Remove x-axis numbers
        for tic in ax_inset.xaxis.get_major_ticks() + ax_inset.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)

    if group_name == "cifar10":
        # Create an inset axis
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper left', borderpad=2.5)
        half_length = len(query_numbers) // 2
        for i, (acc, acc_std) in enumerate(zip(acc_list, acc_std_list)):
            label = list(method_colors.keys())[i]
            ax_inset.plot(query_numbers[half_length:], acc[half_length:], label=label, color=method_colors[label],
                          linewidth=width_map[label])
            ax_inset.fill_between(query_numbers[half_length:],
                                  np.array(acc[half_length:]) - 0.8 * np.array(acc_std[half_length:]),
                                  np.array(acc[half_length:]) + 0.8 * np.array(acc_std[half_length:]), alpha=0.2,
                                  color=method_colors[label])

        ax_inset.set_xlim(7.0, query_numbers[-1])  # Set x-axis from 6.0 to the end
        ax_inset.set_ylim(93.5, 92.5 + 6)  # Set y-axis range from 92.5 to 100.5
        ax_inset.xaxis.set_ticklabels([])  # Remove x-axis numbers
        ax_inset.tick_params(axis='both', which='major', labelsize=12)
        for tic in ax_inset.xaxis.get_major_ticks() + ax_inset.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)

    plt.show()

    # Precision plot
    fig, ax = plt.subplots()
    for i, (precision, precision_std) in enumerate(zip(precision_list, precision_std_list)):
        label = list(method_colors.keys())[i]
        ax.plot(query_numbers, np.array(precision) * 100, label=label, color=method_colors[label],
                linewidth=width_map[label])
        ax.fill_between(query_numbers, (np.array(precision) - 0.8 * np.array(precision_std)) * 100,
                        (np.array(precision) + 0.8 * np.array(precision_std)) * 100, alpha=0.2,
                        color=method_colors[label])

    # Add legend to the plot



    for tick in ax.get_yticks():
        ax.axhline(tick, linestyle='dashed', alpha=0.3, color='gray')

    ax.xaxis.set_ticks(range(len(query_numbers)))

    for tick in ax.get_xticks():
        ax.axvline(tick, linestyle='dashed', alpha=0.3, color='gray')

    ax.set_xlabel('Query Round', fontsize=14)
    ax.set_ylabel('Precision (%)', fontsize=14)
    ax.set_title(f'{group_name.split()[0]} Batch {batch_size}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(0, 100)



    plt.show()

    # Recall plot
    fig, ax = plt.subplots()
    for i, (recall, recall_std) in enumerate(zip(recall_list, recall_std_list)):
        label = list(method_colors.keys())[i]
        ax.plot(query_numbers, np.array(recall) * 100, label=label, color=method_colors[label],
                linewidth=width_map[label])
        ax.fill_between(query_numbers, (np.array(recall) - 0.8 * np.array(recall_std)) * 100,
                        (np.array(recall) + 0.8 * np.array(recall_std)) * 100, alpha=0.2, color=method_colors[label])

    # Add legend to the plot


    for tick in ax.get_yticks():
        ax.axhline(tick, linestyle='dashed', alpha=0.3, color='gray')

    ax.xaxis.set_ticks(range(len(query_numbers)))
    for tick in ax.get_xticks():
        ax.axvline(tick, linestyle='dashed', alpha=0.3, color='gray')

    ax.set_xlabel('Query Round', fontsize=14)
    ax.set_ylabel('Recall (%)', fontsize=14)
    ax.set_title(f'{group_name.split()[0]} Batch {batch_size}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


from matplotlib import patches
import matplotlib.pyplot as plt

from matplotlib import patches
import matplotlib.pyplot as plt


from matplotlib import patches
import matplotlib.pyplot as plt

def plot_legend(datasets):
    # Prepare a list of all known classes
    known_classes = [known_class for dataset in datasets.values() for known_class in dataset['known_class']]

    # Define colors for each known_class
    method_colors = {datasets[dataset_name]['known_class'][i]: plt.cm.tab10(i) for dataset_name in datasets for i in
                     range(len(datasets[dataset_name]['known_class']))}

    # Swap the color of 'Clip' with 'Image 50'
    method_colors['_pretrained_model_clip'], method_colors['_pretrained_model_image50'] = method_colors['_pretrained_model_image50'], method_colors['_pretrained_model_clip']

    # Create labels mapping
    labels_map = {
        '_pretrained_model_image50': 'Image 50',
        '_pretrained_model_image18': 'Image 18',
        '_pretrained_model_image34': 'Image 34',
        '_pretrained_model_clip': 'Clip',
    }

    # Replace the labels
    known_class_labels = [labels_map.get(known_class, known_class) for known_class in known_classes]

    # Reorder legend elements
    desired_order = ['Clip', 'Image 18', 'Image 34', 'Image 50']
    other_methods = [method for method in known_class_labels if method not in desired_order]
    ordered_known_classes = desired_order + other_methods

    # Create a separate legend plot
    legend_elements = [patches.Patch(color=method_colors[known_class], label=label)
                       for known_class, label in zip(known_classes, ordered_known_classes)]

    fig, ax = plt.subplots(figsize=(7.6, 2.4))  # Adjust the figsize to 640x480 pixels
    ax.legend(handles=legend_elements, loc='center', ncol=5, bbox_to_anchor=(0.5, 0.5))
    ax.axis('off')
    plt.show()



plot_legend(datasets)




for dataset_name, dataset_info in datasets.items():
    for batch_size in dataset_info['batch']:
        acc_list, precision_list, recall_list = [], [], []
        acc_std_list, precision_std_list, recall_std_list = [], [], []
        method_colors = {}

        for pretrained_model in dataset_info['known_class']:
            group_name = f"{dataset_name}_{pretrained_model}_Batch{batch_size}"
            pkl_files_dict = load_pkl_files(dataset_name, pretrained_model, batch_size)

            for method, files in pkl_files_dict.items():
                acc_vals, precision_vals, recall_vals = [], [], []
                for file in files:
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                        acc_vals.append([data['Acc'][i] for i in data['Acc']])
                        precision_vals.append([data['Precision'][i] for i in data['Precision']])
                        recall_vals.append([data['Recall'][i] for i in data['Recall']])

                if acc_vals:
                    num_seeds = len(files)
                    acc_avg = np.mean(acc_vals, axis=0).tolist()
                    precision_avg = np.mean(precision_vals, axis=0).tolist()
                    recall_avg = np.mean(recall_vals, axis=0).tolist()

                    acc_std = np.std(acc_vals, axis=0).tolist()
                    precision_std = np.std(precision_vals, axis=0).tolist()
                    recall_std = np.std(recall_vals, axis=0).tolist()

                    acc_list.append(acc_avg)
                    precision_list.append(precision_avg)
                    recall_list.append(recall_avg)

                    acc_std_list.append(acc_std)
                    precision_std_list.append(precision_std)
                    recall_std_list.append(recall_std)

                    method_colors[pretrained_model] = plt.cm.tab10(len(method_colors))

        plot_graphs(dataset_name, acc_list, precision_list, recall_list, acc_std_list, precision_std_list,
                    recall_std_list, batch_size, method_colors)
