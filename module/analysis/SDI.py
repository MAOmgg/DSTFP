import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import zscore, spearmanr
import pandas as pd
import seaborn as sns
matplotlib.use('TkAgg')
a="Prodromal"
# ==================== 数据加载与预处理 ====================
# 加载结构连接矩阵
W = sio.loadmat(fr"E:\Python\File\dMRIpipeline\data\average_connectivity_{a}.mat")['average_connectivity']
# 加载fMRI数据 (90, 1200, 56)
X_RS = sio.loadmat(fr"E:\Python\File\dMRIpipeline\data\z_normalized_{a}.mat")['z_normalized']

# 参数定义
n_ROI = W.shape[0]
n_timepoints = X_RS.shape[1]
nsubjs_RS = X_RS.shape[2]

# Z-score标准化
zX_RS = zscore(X_RS, axis=1)

# ==================== 结构网络谐波分析 ====================
# 构建对称归一化拉普拉斯矩阵
deg = W.sum(axis=1)
D_inv_sqrt = np.diag(1 / np.sqrt(deg))
Wsymm = D_inv_sqrt @ W @ D_inv_sqrt
L = np.eye(n_ROI) - Wsymm

# 特征分解
LambdaL, U = eigh(L)
sorted_idx = np.argsort(LambdaL)
LambdaL, U = LambdaL[sorted_idx], U[:, sorted_idx]

# 选择前 N 个特征向量进行可视化
N = 5  # 可视化前 5 个特征向量
for i in range(N):
    # 归一化特征向量
    feature_vector = U[:, i]
    feature_vector_normalized = (feature_vector - np.min(feature_vector)) / (
                np.max(feature_vector) - np.min(feature_vector))

    # 构建连接矩阵
    adj_matrix = np.outer(feature_vector_normalized, feature_vector_normalized)

    # 绘制热图
    plt.figure(figsize=(6, 5))
    sns.heatmap(adj_matrix, cmap='coolwarm', cbar=False)
    plt.title(f'Eigenvector {i + 1}')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.tight_layout()
    plt.savefig(f'eigenvector_{i + 1}_heatmap.png')
    plt.close()

# 计算加权零交叉数
wZC = np.zeros(n_ROI, dtype=int)
for u in range(n_ROI):
    vec = U[:, u]
    sign_flip = np.outer(vec, vec) < 0
    mask = np.triu((W > 1) & sign_flip, 1)
    wZC[u] = mask.sum()

plt.plot(wZC)
plt.savefig('Fig_S1.png')
plt.close()

# ==================== 能量谱分析 ====================
X_hat_L = np.tensordot(U.T, zX_RS, axes=[1, 0])
PSD = np.abs(X_hat_L).mean(axis=(1, 2))  # (90,)

# 截止频率计算
auc_total = np.trapezoid(PSD)
i = np.argmax(np.cumsum(PSD) > auc_total / 2)
NN, NNL = i + 1, n_ROI - i - 1

# ==================== 结构解耦分析 ====================
# 谐波分组
M = np.fliplr(U)
Vlow = np.zeros_like(M)
Vhigh = np.zeros_like(M)
Vlow[:, -NN:] = M[:, -NN:]
Vhigh[:, :NNL] = M[:, :NNL]

# 计算SDI
N_c = np.array([np.linalg.norm(Vlow @ (M.T @ zX_RS[..., s]), axis=1) for s in range(nsubjs_RS)]).T
N_d = np.array([np.linalg.norm(Vhigh @ (M.T @ zX_RS[..., s]), axis=1) for s in range(nsubjs_RS)]).T
mean_SDI = np.mean(N_d / N_c, axis=1)

# SDI 散点图生成与保存
plt.figure(figsize=(10, 6))
plt.scatter(range(n_ROI), mean_SDI, c='indigo', alpha=0.7)
plt.axhline(y=1.0, color='gray', linestyle='--')  # 添加参考线，SDI=1.0 表示无解耦
plt.title('Structural Decoupling Index with Significance')
plt.xlabel('Brain Regionplt')
plt.ylabel('SDI (mean N_d/N_c)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'SDI_Scatter_Plot_{a}.png', dpi=300)
plt.close()
# ==================== 替代信号生成 ====================
nSurr = 19

# SC-informed替代信号
XrandS = np.zeros((nsubjs_RS, nSurr, n_ROI, n_timepoints))
for s in range(nsubjs_RS):
    X = zX_RS[..., s]
    for n in range(nSurr):
        PHI = np.diag(np.random.choice([-1, 1], n_ROI))
        XrandS[s, n] = M @ PHI @ M.T @ X

# SC-ignorant替代信号（配置模型）
deg_cm = W.sum(axis=1)
W_cm = np.outer(deg_cm, deg_cm) / deg_cm.sum()
L_cm = np.diag(deg_cm) - W_cm
_, U_cm = eigh(L_cm)
M_cm = np.fliplr(U_cm)

XrandSran = np.zeros((nSurr, nsubjs_RS, n_ROI, n_timepoints))
for n in range(nSurr):
    for s in range(nsubjs_RS):
        PHI = np.diag(np.random.choice([-1, 1], n_ROI))
        XrandSran[n, s] = M_cm @ PHI @ M_cm.T @ zX_RS[..., s]


# ==================== 功能连接分析 ====================
def compute_fc(data):
    """通用功能连接计算"""
    return np.array([np.corrcoef(x) for x in data])


# 真实FC
real_fc = compute_fc(zX_RS.transpose(2, 0, 1)).mean(axis=0)

# SC-informed替代FC
surr_fc = compute_fc(XrandS.reshape(-1, n_ROI, n_timepoints)).mean(axis=0)

# SC-ignorant替代FC
rand_fc = np.array([compute_fc(XrandSran[n]).mean(axis=0) for n in range(nSurr)]).mean(axis=0)

# ==================== 结果可视化 ====================
# 功能连接矩阵对比
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, title, data in zip(axes, ['Real FC', 'SC-informed FC', 'SC-ignorant FC'],
                           [real_fc, surr_fc, rand_fc]):
    ax.imshow(data, cmap='viridis', vmin=0, vmax=0.7)
    ax.set_title(title)
    ax.axis('off')
plt.savefig(f'FC_comparison_{a}.png', dpi=300)
plt.close()
# ==================== 独立可视化三个功能连接矩阵 ====================
# 设置公共参数
vmin, vmax = 0, 0.7
cmap = 'viridis'
dpi = 300
# ==================== 结构连接矩阵可视化 ====================
# 步骤1：对称归一化处理
deg = W.sum(axis=1)
D_inv_sqrt = np.diag(1 / np.sqrt(deg))
W_symm = D_inv_sqrt @ W @ D_inv_sqrt

# 步骤2：动态范围调整
vmaxs = np.percentile(W_symm, 99)  # 排除前1%的极端值

plt.figure(figsize=(8, 6))
plt.imshow(W_symm, cmap='viridis',
           vmin=0, vmax=vmaxs,
           interpolation='none')
plt.colorbar(label='Normalized SC Strength')
plt.title('Processed Structural Connectivity')
plt.axis('off')
plt.savefig(f'SC_Processed_{a}.png', dpi=300, bbox_inches='tight')
plt.close()

# 1. 真实功能连接矩阵
plt.figure(figsize=(8, 6))
plt.imshow(real_fc, cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar(label='Functional Connectivity', shrink=0.8)
plt.title('Empirical Functional Connectivity', fontsize=14)
plt.axis('off')
plt.savefig(f'{a}_Empirical_FC.png', bbox_inches='tight', dpi=dpi)
plt.close()

# 2. SC-informed替代功能连接矩阵
plt.figure(figsize=(8, 6))
plt.imshow(surr_fc, cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar(label='Functional Connectivity', shrink=0.8)
plt.title('SC-informed Surrogate FC', fontsize=14)
plt.axis('off')
plt.savefig(f'{a}_SC_informed_FC.png', bbox_inches='tight', dpi=dpi)
plt.close()

# 3. SC-ignorant替代功能连接矩阵
plt.figure(figsize=(8, 6))
plt.imshow(rand_fc, cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar(label='Functional Connectivity', shrink=0.8)
plt.title('SC-ignorant Surrogate FC', fontsize=14)
plt.axis('off')
plt.savefig(f'{a}_SC_ignorant_FC.png', bbox_inches='tight', dpi=dpi)
plt.close()

# 节点强度分析
ns_real = real_fc.sum(axis=0)
ns_surr = surr_fc.sum(axis=0)
ns_rand = rand_fc.sum(axis=0)
ns_sc = W.sum(axis=0)

# 保存结果
pd.DataFrame({
    'SC': ns_sc,
    'FC_real': ns_real,
    'FC_surr': ns_surr,
    'FC_rand': ns_rand
}).to_csv('node_strength.csv')

# 相关性分析
results = {
    'SC-informed': spearmanr(ns_sc, ns_surr),
    'Empirical': spearmanr(ns_sc, ns_real),
    'SC-ignorant': spearmanr(ns_sc, ns_rand)
}

# 保存统计结果
with open('correlation_results.txt', 'w') as f:
    for k, (rho, p) in results.items():
        f.write(f"{k}: rho={rho:.3f}, p={p:.4f}\n")

