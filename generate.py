import os

sampling_methods = ['random', 'uncertainty', "BGADL", "OpenMax", "Core_set", 'certainty', "AV_temperature", "BADGE_sampling"]

# datasets = {'Tiny-Imagenet': {'init_percent': 8, 'known_class': [40, 60, 80]},
#             'cifar100': {'init_percent': 8, 'known_class': [20, 30, 40]},
#             'cifar10': {'init_percent': 1, 'known_class': [2, 3, 4]}}
datasets = {'cifar10': {'init_percent': 1, 'known_class': [2]}}

def generate_command(sampling_method, dataset_name, gpu_id, seed):
    dataset = datasets[dataset_name]
    init_percent = dataset['init_percent']
    known_class = dataset['known_class']

    commands = []
    for kc in known_class:
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python AL_center_temperature.py --gpu 0 --save-dir log_AL/ --weight-cent 0 --query-strategy {sampling_method} --init-percent {init_percent} --known-class {kc} --query-batch 400 --seed {seed} --model resnet50 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset {dataset_name} > ./log_AL/{sampling_method}_{init_percent}_{kc}_{dataset_name}_seed{seed}_resnet50.txt &"
        commands.append(command)

    return commands

num_gpus = 4
seeds = [1, 2, 3]

with open("baseline.txt", "w") as f:
    gpu_counter = 0
    for method in sampling_methods:
        for dataset_name in datasets:
            for kc in datasets[dataset_name]['known_class']:
                for seed in seeds:
                    command = generate_command(method, dataset_name, gpu_counter % num_gpus, seed)
                    f.write(command[0] + "\n")
                    gpu_counter += 1
