import pickle

file_name = "temperature_resnet18_Tiny-Imagenet_known40_init8_batch1500_seed1_certainty_unknown_T0.5_known_T0.5_modelB_T1.0.pkl"

# Add the directory to the file path
file_path = f"log_AL/{file_name}"

# Open the file and load the data
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    max_min_temperature_acc = [data['Acc'][i] for i in data['Acc']]
    max_min_temperature_precision = [data['Precision'][i] for i in data['Precision']]
    max_min_temperature_recall = [data['Recall'][i] for i in data['Recall']]

print(max_min_temperature_acc)