import numpy as np

# Define two vectors
A = np.array([[0.6], [0.4], [0.0]])
# A = np.array([[0.34], [0.33], [0.33]])
B = np.array([0.4])

# Calculate dot product
dot_product = np.dot(A, B)

# Calculate norms
norm_a = np.linalg.norm(A)
norm_b = np.linalg.norm(B)

# Calculate cosine similarity
cos_sim = dot_product / (norm_a * norm_b)

print("Cosine similarity:", cos_sim)