# from tkinter import Image
#
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from sklearn.manifold import TSNE
# from sklearn.neighbors import NearestNeighbors
# import clip
# import numpy as np
#
# # Load the pre-trained CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# # Load the CIFAR-10 dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
#
# # Extract features using CLIP
# features = []
# for images, _ in trainloader:
#     images = images.to(device)
#     with torch.no_grad():
#         feature_vectors = model.encode_image(images)
#         features.append(feature_vectors.cpu().numpy())
#
# features = np.vstack(features)
#
# # Perform dimensionality reduction using t-SNE
# tsne = TSNE(n_components=2)
# features_2D = tsne.fit_transform(features)
#
# # Function to preprocess and extract features from an input image
# def preprocess_and_extract_features(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image_transformed = preprocess(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         feature_vector = model.encode_image(image_transformed)
#     return feature_vector.cpu().numpy()
#
# # Given an input image, find the k nearest neighbors in the transformed space
# def find_k_nearest_neighbors(image_path, k=5):
#     input_features = preprocess_and_extract_features(image_path)
#     input_features_2D = tsne.transform(input_features)
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(features_2D)
#     distances, indices = nbrs.kneighbors(input_features_2D)
#     return distances, indices
#
# # Test the function with an input image and k = 5
# image_path = "path/to/your/image.jpg"
# distances, indices = find_k_nearest_neighbors(image_path, k=5)
# print("Distances:", distances)
# print("Indices:", indices)
S_ij = {}
tmp_index = 28695
S_ij[tmp_class] = []
S_ij[tmp_index].append([tmp_class, tmp_value, tmp_label])