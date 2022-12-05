import numpy as np

# train_info = np.load('./result/07-22-19-23-51_ABIDE_fbnetgen_normal_attention_loss_group_loss_sparsity_loss_8_4/learnable_matrix.npy', allow_pickle=True)
# print(train_info)

health = np.load('./health.npy', allow_pickle=True)
for i in range(90):
    for j in range(90):
        if health[i][j] > 0.8915 and i != j:
            print(i, j)

# asd = np.load('./asd.npy', allow_pickle=True)
# for i in range(90):
#     for j in range(90):
#         if asd[i][j] > 0.8595 and i != j:
#             print(i, j)

edge = np.zeros((90, 90), dtype = int, order = 'C')

edge[33][67] = 1
edge[42][45] = 1
edge[44][67] = 1
edge[67][45] = 1
edge[67][47] = 1

np.savetxt('./aal.edge', edge)