import os
import pickle
from types import SimpleNamespace

import matplotlib
import numpy as np
import seaborn as sns
import torch
from rdkit.Chem.Draw import MolToMPL
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import Draw

from data_process.dataLoader import loadData
from model_molformer import MolFormerCDR
import matplotlib.pyplot as plt
from rdkit import Chem

from molformer_ablidation import AblationMolFormerCDR

LOAD_CONFIG_GDSC2 = {
    "Gene_expression_ori_file": False,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": False,
    "Methylation_filter_file": True,
    "Gene_mutation_ori_file": False,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": True,
    "Copy_number_file": False,
    "GDSC1": False,
    "GDSC2": True,
    'NAN_GDSC2':False
}

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

device = "cuda:3"
args = {
    "seed":42,
    "train_batch_size": 512,
    "test_batch_size": 512,
    "finetune_layers": 1
}
args_obj = SimpleNamespace(**args)

def getMyMolformer(state_dict_path = "./zoo/model.pth"):
    state_dict = torch.load(state_dict_path, map_location=device)
    model = MolFormerCDR(args_obj, do_down_stream=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def getAbliMolformer(state_dict_path = "/home/liushujia/BertCDR/ablation/gdsc2_base_finetune/model_6.pth"):
    state_dict = torch.load(state_dict_path, map_location=device)
    model = AblationMolFormerCDR(args_obj, do_down_stream=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def getOriMolformer(molformer_dir = "/data0/liushujia/MoLFormer/"):
    model = AutoModel.from_pretrained(molformer_dir)
    model = model.to(device)
    return model

def get_atom_tokens_and_indices(tokens):
    """提取对应真实原子的token及其在序列中的索引"""
    atom_tokens = []
    atom_indices = []
    for idx, token in enumerate(tokens):
        if token in ["<bos>", "<eos>", "<pad>", "<mask>", "<unk>",'(', ')', '=', '#', '/', '\\', '.']:  # 排除非原子token
            continue
        # 可选：进一步排除数字（环标记）或其他非原子符号
        if token.isdigit():
            continue
        atom_tokens.append(token)
        atom_indices.append(idx)
    return atom_tokens, atom_indices

def plot_tsne(x, y, title="t-SNE", save_path=None):
    """
    x: 特征向量，形状为 [num_samples, feature_dim]
    y: IC50 值，形状为 [num_samples]
    title: 图标题
    save_path: 是否保存图像的路径
    """
    # 归一化 y 到 0~1 范围用于 colormap
    x = x.detach().cpu().numpy()
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    x_2d = tsne.fit_transform(x)
    scaler = MinMaxScaler()
    x_2d = scaler.fit_transform(x_2d)  # 自动对每列独立归一化

    # 绘图
    plt.figure(figsize=(6, 6))

    # 获取 colormap 并重采样为 12 个颜色
    cmap = matplotlib.colormaps.get_cmap('coolwarm_r').resampled(10)
    scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y_norm, cmap=cmap, s=10, alpha=0.8)

    # colorbar
    cbar = plt.colorbar(
        scatter,
        ticks=np.linspace(0, 1, 11),
        aspect=30,
        shrink=0.7,
        pad=0.05,
    )
    cbar.set_label("Normalized IC50", fontsize=18)

    # 设置图形属性
    plt.title(title, pad=12, fontsize=20)
    plt.xlabel("t-SNE1", fontsize=18)
    plt.ylabel("t-SNE2", fontsize=18)


    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_claster_graph():
    args_obj.seed = 26
    args_obj.train_batch_size = 1024
    args_obj.test_batch_size = 1024
    _, _, loader = loadData(args_obj, LOAD_CONFIG_GDSC2)
    model = getMyMolformer()
    model.eval()
    for i, batch in tqdm(enumerate(loader), total=len(loader), desc="Test_Loader", leave=False):
        with torch.no_grad():
            smiles = batch["smiles"]
            morgan = batch["morgan"].to(device)
            rdkit = batch["rdkit"].to(device)
            gexpr = batch["gexpr"].to(device)
            methylation = batch["methylation"].to(device)
            mutation = batch["mutation"].to(device)
            miRNA = batch["miRNA"].to(device)
            copynumber = batch["copynumber"].to(device)
            y = batch["y"].to(device)

        with torch.no_grad():
            outputs, x0, x1, x2 = model.forward_with_mid({
                "smiles": smiles,
                "morgan": morgan,
                "rdkit": rdkit,
                "gexpr": gexpr,
                "methylation": methylation,
                "mutation": mutation,
                "miRNA": miRNA,
                "copynumber": copynumber
            })
            mids = [x0, x1, x2]
            titles = ["Initial feature distribution",
                      "After feature extraction",
                      "After fusion"
                    ]

            for m,t in zip(mids,titles):
                plot_tsne(
                    x = m,
                    y = y.detach().cpu().numpy().flatten(),
                    title = t
                )

# 绘制训练阶段图
#plot_claster_graph()

import pickle
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def find_most_dense_cluster(tsne_results, k=10):
    """
    根据 t-SNE 的二维坐标，找到密度最高的 k 个点（包括中心）
    tsne_results: (N, 2)
    k: 簇大小，默认 10
    返回：长度为 k 的 indices
    """

    N = tsne_results.shape[0]

    # pairwise 欧式距离矩阵 (N x N)
    diff = tsne_results[:, None, :] - tsne_results[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    best_cluster = None
    best_score = np.inf  # 距离越小越密集

    for i in range(N):
        d = dist_matrix[i]                  # 以 i 为中心的距离列表
        nearest = np.argsort(d)[:k]         # 最近的 k 个点（包含自己）

        cluster_distances = dist_matrix[np.ix_(nearest, nearest)]
        score = cluster_distances.mean()    # 簇内部平均距离（越小越密）

        if score < best_score:
            best_score = score
            best_center = i
            best_cluster = nearest

    return best_cluster

def show_dense_smiles(smiles_list):

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    for mol in mols:
        # 绘制分子（仅高亮前5个原子）
        img = Draw.MolToImage(mol, size=(400, 400),
                          kekulize=True,
                          highlightRadius=5.0
                          )
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def tsne_find_dense_cluster(k=5):
    model = getMyMolformer()

    with open("/data0/liushujia/omics_new/drugid2smiles.pkl", "rb") as f:
        drugid2smiles = pickle.load(f)

    # 去重（按首次出现顺序）
    seen = set()
    smiles_list = []
    for s in drugid2smiles.values():
        if s not in seen:
            seen.add(s)
            smiles_list.append(s)

    print(f"去重后共有 {len(smiles_list)} 个 SMILES")

    # -------- 2. 获取 embedding --------
    model.eval()
    with torch.no_grad():
        emb = model.get_drug_embedding(smiles_list)
        emb = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)

    # -------- 3. t-SNE 降维 --------
    print("正在执行 t-SNE ...")
    tsne = TSNE(n_components=2, random_state=42, learning_rate="auto", init="pca", perplexity=10)
    tsne_results = tsne.fit_transform(emb)

    # -------- 4. 找最密集簇 --------
    print("寻找最密集的簇 ...")
    dense_indices = find_most_dense_cluster(tsne_results, k=k)
    dense_mask = np.zeros(len(smiles_list), dtype=bool)
    dense_mask[dense_indices] = True

    # -------- 5. 绘图 --------
    plt.figure(figsize=(6, 6))

    # 非密集点
    plt.scatter(tsne_results[~dense_mask, 0], tsne_results[~dense_mask, 1],
                color="#ffa631", s=50)
    # 密集簇点
    plt.scatter(tsne_results[dense_mask, 0], tsne_results[dense_mask, 1],
                color="red", edgecolor="black", s=80)

    plt.title("t-SNE Visualization of drugs", fontsize=20)
    plt.xlabel("t-SNE 1", fontsize=18)
    plt.ylabel("t-SNE 2", fontsize=18)

    # 只保留左和下边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.show()

    # -------- 6. 输出 SMILES --------
    print(f"\n最密集的 {k} 个 SMILES（按 index 排序）：\n")
    for idx in dense_indices:
        print(f"index={idx}, smiles={smiles_list[idx]}")
    show_dense_smiles([smiles_list[i] for i in dense_indices])

    return dense_indices, [smiles_list[i] for i in dense_indices]


#tsne_find_dense_cluster()


def visualize_molformer_attention(model, smiles_list, smiles2drugid):
    model.eval()
    with torch.no_grad():
        cls, attn, tokens = model.forward_drug_only(smiles_list)

    token_ids = tokens['input_ids']
    attn_means = attn[-1].mean(dim=1)  # 平均所有头

    for i in range(attn_means.shape[0]):
        s = smiles_list[i]
        drug_id = smiles2drugid[s]
        tokens_i = model.Tokenizer.convert_ids_to_tokens(token_ids[i])
        os.makedirs(f"./attn_pic/{drug_id}_2", exist_ok=True)
        print(f"SMILES: {s}")
        print("Orign Tokens:", tokens_i)

        # 获取分子对象
        mol = Chem.MolFromSmiles(s)
        num_atoms = mol.GetNumAtoms()

        atom_tokens, atom_indices = get_atom_tokens_and_indices(tokens_i)
        print("Cleaned Tokens:", atom_tokens)
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        print("Atoms:", atom_types)
        assert len(atom_tokens) == num_atoms, f"Token与原子数不匹配: {len(atom_tokens)} vs {num_atoms}"

        atom_attn = attn_means[i][atom_indices, :][:, atom_indices].detach().cpu().numpy()
        atom_attn = (atom_attn + atom_attn.T) / 2

        # 可视化注意力矩阵热力图
        plt.figure(figsize=(10,10))
        ax = sns.heatmap(
            atom_attn,
            xticklabels=[a.GetSymbol() for a in mol.GetAtoms()],
            yticklabels=[a.GetSymbol() for a in mol.GetAtoms()],
            cmap='YlOrRd',
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        plt.title(s, y=1.1, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

        plt.savefig(f"./attn_pic/{drug_id}_2/finetuned_hitmap.png", dpi=300, bbox_inches='tight')
        plt.show()


        atom_weights = atom_attn.sum(axis=1)  # 对每行求和（每个原子的总注意力）
        atom_weights = (atom_weights - atom_weights.min()) / (atom_weights.max() - atom_weights.min())  # 归一化到[0,1]

        # 选择权重最高的前5个原子
        topk = 5
        topk_indices = np.argsort(atom_weights)[-topk:]  # 取权重最大的5个索引
        topk_weights = atom_weights[topk_indices]

        # 构造高亮颜色字典（仅高亮前5个原子）
        highlight_colors = {
            idx: (1.0, 0.0, 0.0, weight)  # RGBA格式（红色，透明度=权重）
            for idx, weight in zip(topk_indices, topk_weights)
        }

        # 绘制分子（仅高亮前5个原子）
        img = Draw.MolToImage(mol, size=(400, 400),
                              kekulize=True,
                              highlightAtoms=[int(idx) for idx in topk_indices],  # 仅高亮选中的原子
                              highlightAtomColors=highlight_colors,
                              highlightRadius=5.0
                              )
        plt.figure(figsize=(6, 6))
        plt.title(s)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"./attn_pic/{drug_id}_2/finetuned.png", dpi=300, bbox_inches='tight')
        plt.show()


def visualize_smiles_attention(model, smiles, topk=5):
    """
    可视化单个 SMILES 的 MolFormer 最后一层 self-attention
    - 显示 atom-level attention heatmap
    - 显示 attention 加权的分子高亮图
    """

    model.eval()
    with torch.no_grad():
        cls, attn, tokens = model.forward_drug_only([smiles])

    # ===== 基本取值 =====
    token_ids = tokens["input_ids"][0]
    tokens_i = model.Tokenizer.convert_ids_to_tokens(token_ids)

    # 取最后一层，平均所有 head
    attn_last = attn[-1][0].mean(dim=0)   # [L, L]

    print("SMILES:", smiles)
    print("Tokens:", tokens_i)

    # ===== SMILES → Mol =====
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, "RDKit 无法解析该 SMILES"

    num_atoms = mol.GetNumAtoms()
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    # ===== token → atom 对齐 =====
    atom_tokens, atom_indices = get_atom_tokens_and_indices(tokens_i)
    assert len(atom_tokens) == num_atoms, \
        f"Token 与原子数不匹配: {len(atom_tokens)} vs {num_atoms}"

    # ===== atom-level attention =====
    atom_attn = attn_last[atom_indices][:, atom_indices].cpu().numpy()
    atom_attn = (atom_attn + atom_attn.T) / 2  # 对称化

    # ===== Heatmap =====
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        atom_attn,
        xticklabels=atom_symbols,
        yticklabels=atom_symbols,
        cmap="YlOrRd",
        square=True,
        cbar=True
    )
    plt.title("Atom-level Attention Heatmap")
    plt.tight_layout()
    plt.show()

    # ===== 原子权重（行和）=====
    atom_weights = atom_attn.sum(axis=1)
    atom_weights = (atom_weights - atom_weights.min()) / (atom_weights.max() - atom_weights.min() + 1e-6)

    # Top-K 原子
    topk_indices = np.argsort(atom_weights)[-topk:]
    topk_weights = atom_weights[topk_indices]

    highlight_colors = {
        int(idx): (1.0, 0.0, 0.0, float(w))
        for idx, w in zip(topk_indices, topk_weights)
    }

    # ===== 分子高亮图 =====
    img = Draw.MolToImage(
        mol,
        size=(400, 400),
        highlightAtoms=[int(i) for i in topk_indices],
        highlightAtomColors=highlight_colors,
        highlightRadius=0.5
    )

    plt.figure(figsize=(5, 5))
    plt.title("Top-Attended Atoms")
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# demo
model = getMyMolformer()
smiles = "CNC1=CC=CC=C1C(=O)[O-]"
visualize_smiles_attention(model, smiles)