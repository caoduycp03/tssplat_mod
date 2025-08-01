import torch

# Suppose you have two triangles sharing a vertex:
# Triangle 1: (0,0,0), (1,0,0), (0,1,0)
# Triangle 2: (0,0,0), (0,1,0), (0,0,1)
points = torch.tensor([
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # Triangle 1
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # Triangle 2
])

# Step 1: Flatten to a list of vertices
all_verts = points.reshape(-1, 3)
print("All vertices:\n", all_verts)

# Step 2: Get unique vertices and inverse indices
unique_verts, inv_idx = torch.unique(all_verts, dim=0, return_inverse=True)
print("Unique vertices:\n", unique_verts)
print("Inverse indices:\n", inv_idx)

# Step 3: Reshape indices to faces
faces = inv_idx.reshape(-1, 3)
print("Faces (indices into unique vertices):\n", faces)