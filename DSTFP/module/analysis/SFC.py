import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

a = "PD"

# ==================== 数据加载 ====================
SC = sio.loadmat(fr"E:\Python\File\dMRIpipeline\data\average_connectivity_{a}.mat")['average_connectivity']
X_RS = sio.loadmat(fr"E:\Python\File\dMRIpipeline\data\z_normalized_{a}.mat")['z_normalized']

n_regions, n_timepoints, n_subjects = X_RS.shape

# ==================== 计算平均 FC ====================
FC_all = np.zeros((n_subjects, n_regions, n_regions))
for subj in range(n_subjects):
    ts = X_RS[:, :, subj]
    FC_all[subj] = np.corrcoef(ts)
FC_mean = np.mean(FC_all, axis=0)

# ==================== 计算 SFC ====================
SFC_matrix = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    for j in range(n_regions):
        if i != j:
            sc_vec = np.delete(SC[i, :], i)
            fc_vec = np.delete(FC_mean[j, :], j)
            SFC_matrix[i, j], _ = pearsonr(sc_vec, fc_vec)




plt.figure(figsize=(8, 6))
im = plt.imshow(SFC_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04, label='SFC (Pearson r)')
plt.axis('off')
plt.tight_layout()
plt.show()
