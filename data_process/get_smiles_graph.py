import os
import pickle

import deepchem as dc
import pandas as pd
from rdkit import Chem, DataStructs
import numpy as np
import hickle as hkl
from rdkit.Chem import AllChem
import torch
import random
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from main_helper import augment_smiles


def augment_smiles_rdkit(smiles, n_aug=3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    aug_smiles = set()

    while len(aug_smiles) < n_aug:
        rand_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        if rand_smiles != smiles:  # 排除原 SMILES
            aug_smiles.add(rand_smiles)

    return list(aug_smiles)

# smiles = "C[C@@H]1[C@@H](C(=O)N[C@@H](C(=O)N2CCC[C@H]2C(=O)N(CC(=O)N([C@H](C(=O)O1)C(C)C)C)C)C(C)C)NC(=O)C3=C4C(=C(C=C3)C)OC5=C(C(=O)C(=C(C5=N4)C(=O)N[C@H]6[C@H](OC(=O)[C@@H](N(C(=O)CN(C(=O)[C@@H]7CCCN7C(=O)[C@H](NC6=O)C(C)C)C)C)C(C)C)C)N)C"
# mol = Chem.MolFromSmiles(smiles)
# if mol is None:
#     print(f"❌ 无法解析 SMILES: {smiles}")
#
# # 用 DeepChem featurizer 提取图特征
# featurizer = dc.feat.graph_features.ConvMolFeaturizer()
# mol_object = featurizer.featurize([mol])
#
# features = mol_object[0].atom_features
# degree_list = mol_object[0].deg_list
# adj_list = mol_object[0].canon_adj_list
# print("**")

# with open("/data0/liushujia/drugid2graph_dict.pkl", "rb") as f:
#     d = pickle.load(f)

def smiles_to_graph(smiles: str):
    """
    将 SMILES 转换为图特征，返回 (features, edge_index)，
    其中 features: [num_atoms, feat_dim], edge_index: [2, num_edges]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"❌ 无法解析 SMILES: {smiles}")

    # DeepChem featurizer
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mol_object = featurizer.featurize([mol])[0]

    # 节点特征
    features = torch.tensor(mol_object.atom_features, dtype=torch.float)

    # 构建 edge_index
    edge_list = []
    for src, neighbors in enumerate(mol_object.canon_adj_list):
        for dst in neighbors:
            edge_list.append([src, dst])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, num_edges]

    return features, edge_index

# with open("/data0/liushujia/omics_new/drugid2smiles.pkl", "rb") as f:
#     smiles = pickle.load(f)
# drugid2graph_dict = {}
# for id, s in smiles.items():
#     drugid2graph_dict[id] = smiles_to_graph(s)
# with open("/data0/liushujia/drugid2graph_dict.pkl", "wb") as f:
#     pickle.dump(drugid2graph_dict, f)



# -------------------
# 2. 加载 MolFormer
# -------------------
molformer_dir = "/data0/liushujia/MoLFormer/"  # 修改成你自己的路径
tokenizer = AutoTokenizer.from_pretrained(molformer_dir, max_length=512, trust_remote_code=True)
model = AutoModel.from_pretrained(molformer_dir, trust_remote_code=True)
model.eval()


# with open("/data0/liushujia/omics_new/high_missing_smiles.pkl", "rb") as f:
#     aug_smiles_set = pickle.load(f)
# 遍历整个 aug_smiles_set
smiles_list = ["C1CCC(=NNC2=NC(=CS2)C3=CC=C(C=C3)C#N)C1"]
print((smiles_list))
# 生成增强 SMILES
# ours
#aug_smiles = augment_smiles(smiles_list, n_aug=10)
aug_smiles = augment_smiles_rdkit(smiles_list[0], n_aug=10)
all_smiles = smiles_list + aug_smiles
print("增强 SMILES:")
print(all_smiles)

# 计算 embedding
inputs = tokenizer(all_smiles, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]

# 计算余弦相似度
sim_matrix = cosine_similarity(embeddings)
print("\n余弦相似度矩阵:")
print(sim_matrix)

# ========= 热力图可视化 =========
plt.figure(figsize=(8, 6))
sns.heatmap(
    sim_matrix,
    cmap="Blues",  # 白到深蓝的渐变色
    annot=True,
    fmt=".2f",
    cbar=True,  # 右侧颜色条
    square=True,
    vmin=0, vmax=1  # 归一化区间
)
plt.title(f"SMILES augmentation similarity", fontsize=18, pad=20)
plt.xticks(ticks=range(len(all_smiles)), labels=[f"S{j}" for j in range(len(all_smiles))], rotation=45, ha='right', fontsize=14)
plt.yticks(ticks=range(len(all_smiles)), labels=[f"S{j}" for j in range(len(all_smiles))], rotation=0, fontsize=14)
plt.savefig("../pic/rdkit_aug_similarity.png", dpi=600, bbox_inches='tight', transparent=True)
plt.tight_layout()
plt.show()

# from transformers import AutoTokenizer
# molformer_dir = "/data0/liushujia/MoLFormer/"
# tokenizer = AutoTokenizer.from_pretrained(molformer_dir, max_length=512, trust_remote_code=True)
# #aug_smiles = augment_smiles(["CNC1=CC=CC=C1C(=O)[O-]"])
# aug_smiles = ["CNC1=[MASK]C=CC=1C(=O)[O-]"]
# print(aug_smiles)
# token = tokenizer(aug_smiles)
# token_ids = token["input_ids"][0]
# tokens = tokenizer.convert_ids_to_tokens(token_ids)
# print(tokens)
# from rdkit import Chem
# import pickle
# import torch
# from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity
#
# molformer_dir = "/data0/liushujia/MoLFormer/"
# tokenizer = AutoTokenizer.from_pretrained(molformer_dir, max_length=512, trust_remote_code=True)
# model = AutoModel.from_pretrained(molformer_dir, trust_remote_code=True)
# model.eval()
#
# with open("/data0/liushujia/omics_new/high_missing_smiles.pkl", "rb") as f:
#     aug_smiles_set = pickle.load(f)
#
# for i, base_smiles in enumerate(aug_smiles_set):
#     print(f"===== 第 {i + 1} 个分子 =====")
#     print("原始 SMILES:", base_smiles)
#
#     mol = Chem.MolFromSmiles(base_smiles)
#     if mol is None:
#         continue
#
#     # 每个分子生成 5 个随机 SMILES
#     random_smiles_list = [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(5)]
#     all_smiles = [base_smiles] + random_smiles_list
#
#     print("随机 SMILES 增强:")
#     for s in all_smiles:
#         print(s)
#
#     # -------------------
#     # 计算 embedding
#     # -------------------
#     inputs = tokenizer(all_smiles, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] embedding
#
#     # -------------------
#     # 计算相似度矩阵
#     # -------------------
#     sim_matrix = cosine_similarity(embeddings)
#     print("\n余弦相似度矩阵:")
#     print(sim_matrix)
#     print("\n")


# import torch
# from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity
# import random
#
# # -----------------------------
# # 1. MolFormer 初始化
# # -----------------------------
# molformer_dir = "/data0/liushujia/MoLFormer/"
# tokenizer = AutoTokenizer.from_pretrained(molformer_dir, max_length=512, trust_remote_code=True)
# model = AutoModel.from_pretrained(molformer_dir, trust_remote_code=True)
# model.eval()

# # -----------------------------
# # 3. 示例 SMILES 数据
# # -----------------------------
# with open("/data0/liushujia/omics_new/high_missing_smiles.pkl", "rb") as f:
#     aug_smiles_set = pickle.load(f)
# smiles_list = list(aug_smiles_set) # demo
#
# # -----------------------------
# # 4. 对每个分子生成增强 SMILES + MolFormer embedding
# # -----------------------------
# for idx, base_smiles in enumerate(smiles_list):
#     print(f"===== 第 {idx + 1} 个分子 =====")
#     print("原始 SMILES:", base_smiles)
#
#     # 生成增强 SMILES
#     aug_smiles = augment_smiles_noise(base_smiles, n_aug=5, mode="fusion")
#     all_smiles = [base_smiles] + aug_smiles
#     print("增强 SMILES:")
#     for s in all_smiles:
#         print(" ", s)
#
#     # -----------------------------
#     # 4.1 Tokenizer + MolFormer embedding
#     # -----------------------------
#     inputs = tokenizer(all_smiles, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] embedding
#
#     # -----------------------------
#     # 4.2 计算余弦相似度矩阵
#     # -----------------------------
#     sim_matrix = cosine_similarity(embeddings)
#     print("\n余弦相似度矩阵:")
#     for row in sim_matrix:
#         print(["{:.2f}".format(v) for v in row])
#     print("\n")




# with open("/data0/liushujia/omics_new/drugid2smiles.pkl" ,"rb") as f:
#     drugid2smiles = pickle.load(f)
#
# smiles2drugid = {smiles: drug_id for drug_id, smiles in drugid2smiles.items()}
# with open("/data0/liushujia/omics_new/high_missing_smiles.pkl", "rb") as f:
#     aug_smiles_set = pickle.load(f)
#
# aug_drug_ids_str = ["GDSC:" + smiles2drugid[s] for s in aug_smiles_set if s in smiles2drugid]
# aug_drug_ids = [int(smiles2drugid[s]) for s in aug_smiles_set if s in smiles2drugid]
#
# df = pd.read_csv("/data0/liushujia/omics_new/GDSC2_IC50.csv", index_col=0, na_values="NA")
# df = df.loc[aug_drug_ids_str]
# missing_ratio = df.isna().mean(axis=1)
# df_missing = pd.DataFrame({
#     'DrugID': missing_ratio.index,
#     'rate': missing_ratio.values
# })
#
# 1️⃣ 读取两个 CSV
csv1 = pd.read_csv("/home/liushujia/BertCDR/ablation/gdsc2_base_freeze/drug_6.csv")  # 包含 DrugID, PCC, SCC, RMSE
csv1 = csv1[['DrugID', 'RMSE']]  # 正确选择列的方式

csv2 = pd.read_csv("/home/liushujia/BertCDR/ablation/gdsc2_base_finetune/drug_6.csv")
csv2 = csv2[['DrugID', 'RMSE']]

# 2️⃣ 合并两个 DataFrame，并添加后缀以区分
df_compare = pd.merge(csv1, csv2, on='DrugID', suffixes=('_base', '_finetune'))

# 3️⃣ 计算 RMSE 的减小幅度（base - finetune）
df_compare['RMSE_Decrease'] = df_compare['RMSE_base'] - df_compare['RMSE_finetune']

# 4️⃣ 按 RMSE 减小幅度降序排序（减小最多的排在最前面）
df_compare_sorted = df_compare.sort_values(by='RMSE_Decrease', ascending=False)

# 5️⃣ 打印排序后的 DataFrame
print(df_compare_sorted)
# 3️⃣ 只保留 aug_drug_ids 对应的行
# csv1_aug = csv1[csv1['DrugID'].isin(aug_drug_ids)][['DrugID', 'RMSE']].copy()
# csv2_aug = csv2[csv2['DrugID'].isin(aug_drug_ids)][['DrugID', 'RMSE']].copy()

# 4️⃣ 按 DrugID 对齐
#df_compare = pd.merge(csv1_aug, csv2_aug, on='DrugID', suffixes=('', '_aug'))
# #
# print(drugid2smiles["1502"])
# print(drugid2smiles["1243"])
# print(drugid2smiles["2169"])
# print(drugid2smiles["1248"])
# print(drugid2smiles["1031"])


# def get_smiles_missing_rate(csv_path = "/data0/liushujia/omics_new/nest_gdsc2.csv"):
#     df = pd.read_csv(csv_path)
#
#     # 第一列是 SMILES
#     smiles = df.iloc[:, 0]
#
#     values = df.iloc[:, 1:]
#
#     # 计算每一行的缺失率
#     missing_rate = values.isna().sum(axis=1) / values.shape[1]
#
#     # 整合成 dict
#     result = dict(zip(smiles, missing_rate))
#     return result
#
# d = get_smiles_missing_rate()
# with open("./smiles_missing_rate.pkl", "wb") as f:
#     pickle.dump(d, f)
