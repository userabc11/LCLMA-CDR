import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, gexpr_feature, methylation_feature, mutation_feature, miRNA, all_respond, drugid2smiles, do_aug=False, n_aug=5, fp_flip_prob=0.03):
        super().__init__()
        self.gexpr_feature = self._scale_dataframe(gexpr_feature)
        self.mutation_feature = self._scale_dataframe(mutation_feature)
        self.methylation_feature = self._scale_dataframe(methylation_feature)
        self.miRNA = self._scale_dataframe(miRNA)
        with open('/data0/liushujia/omics_new/512dim_copynumber_dict.pkl', 'rb') as f:
            self.copynumber = pickle.load(f)
        # with open("/data0/liushujia/omics_new/v2_copynumber_mae_dict.pkl", 'rb') as f:
        #     self.copynumber = pickle.load(f)
        self.all_respond = all_respond
        self.drugid2smiles = drugid2smiles
        self.smiles2drugid = {value: key for key, value in drugid2smiles.items()}
        self.smiles_morgan_fp_dict = {}
        self.smiles_rdkit_fp_dict = {}
        for s in self.smiles2drugid:
            self.smiles_morgan_fp_dict[s] = self.smiles_to_morgan_fp(s)
            self.smiles_rdkit_fp_dict[s] = self.smiles_to_rdkit_fp(s)
        # with open("/data0/liushujia/smiles2vec.pkl", "rb") as f:
        #     self.smiles_fp_dict = pickle.load(f)

        ########   new  #######
        cell_lines = list(self.gexpr_feature.index)
        self.cellid2type = self.get_cellid_type_dict("/data0/liushujia/omics_new/OmicsProfiles.csv")
        self.cellid2type = {cid: t for cid, t in self.cellid2type.items() if cid in cell_lines}
        all_types = sorted(set(self.cellid2type.values()))  # 去重+排序
        print(f"类别数{len(all_types)}")
        self.type2id = {t: i for i, t in enumerate(all_types)}

        if do_aug:
            print("[INFO]增强前，总反应数：" , len(self.all_respond))
            #self.augment_by_drug()
            #self.augment_by_drug_v2()
            #self.augment_by_drug_rdkit_copy()
            #self.augment_by_copy_v3(0)
            self.augment_by_copy()
            print("[INFO]增强后，总反应数：", len(self.all_respond))

    def _scale_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        scaled_array = scaler.fit_transform(df.values)
        return pd.DataFrame(scaled_array, index=df.index, columns=df.columns)

    def smiles_to_morgan_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string: " + smiles)
            # print("use MorganFingerprint")
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=512)
        fingerprint_array = fingerprint.ToBitString()
        fingerprint_tensor = torch.tensor([int(bit) for bit in fingerprint_array], dtype=torch.float)
        return fingerprint_tensor

    def smiles_to_rdkit_fp(self, smiles, nBits=881):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string: " + smiles)
        # 生成 RDKit 指纹
        fp = RDKFingerprint(mol, maxPath=5, fpSize=nBits)
        fp_str = fp.ToBitString()
        fp_tensor = torch.tensor([int(bit) for bit in fp_str], dtype=torch.float)
        return fp_tensor

    def get_cellid_type_dict(self, csv_path):
        # 读入 CSV
        df = pd.read_csv(csv_path)
        df = df[["ModelID", "Lineage"]]
        df["OncotreeCode"] = df["Lineage"].fillna("Other")
        cellid2type = dict(zip(df["ModelID"], df["Lineage"]))

        return cellid2type

    def augment_by_drug_rdkit(self):
        with open("/data0/liushujia/omics_new/high_missing_smiles.pkl", "rb") as f:
            aug_smiles_set = pickle.load(f)
        smiles2augsmiles = {}
        for s in aug_smiles_set:
            aug_smiles = self.augment_smiles_rdkit(s, n_aug=3)
            smiles2augsmiles[s] = aug_smiles
            for aug_s in aug_smiles:
                self.smiles_morgan_fp_dict[aug_s] = self.smiles_to_morgan_fp(aug_s)
                self.smiles2drugid[aug_s] = "none"

        for r in self.all_respond.copy():
            cell = r[0]
            smiles = r[1]
            ic50 = r[2]
            if smiles in aug_smiles_set:
                aug_smiles = smiles2augsmiles[smiles]
                for aug_s in aug_smiles:
                    self.all_respond.append([cell, aug_s, ic50])

    def augment_by_copy_v3(self, std):

        with open("/home/liushujia/BertCDR/data_process/smiles_missing_rate.pkl", "rb") as f:
            missing_rate_dict = pickle.load(f)
        cell_num = 450
        for smile, missing_rate in missing_rate_dict.items():
            if missing_rate < 0.2:
                continue

            related_records = [r for r in self.all_respond if r[1] == smile]
            if len(related_records) > cell_num:
                continue
            copy_num = int((missing_rate-0.2) * cell_num)
            for _ in range(copy_num):
                # 随机选择一条原始记录
                record = random.choice(related_records)
                cell, smiles_orig, ic50 = record

                # 添加高斯噪声
                noise_std = abs(std * ic50)
                ic50_noisy = ic50 + np.random.normal(0, noise_std)

                # 添加到增强列表
                self.all_respond.append([cell, smiles_orig, ic50_noisy])

    def augment_by_drug_rdkit_copy(self):
        with open("/data0/liushujia/omics_new/high_missing_smiles.pkl", "rb") as f:
            aug_smiles_set = pickle.load(f)
        smiles2augsmiles = {}
        for s in aug_smiles_set:
            aug_smiles = self.augment_smiles_rdkit(s, n_aug=3)
            smiles2augsmiles[s] = aug_smiles
            for aug_s in aug_smiles:
                self.smiles_morgan_fp_dict[aug_s] = self.smiles_to_morgan_fp(aug_s)
                self.smiles2drugid[aug_s] = "none"

        for r in self.all_respond.copy():
            cell = r[0]
            smiles = r[1]
            ic50 = r[2]
            if smiles in aug_smiles_set:
                aug_smiles = smiles2augsmiles[smiles]
                for aug_s in aug_smiles:
                    # ic50_noisy  y值扰动
                    noise_std = abs(0.05 * ic50)
                    ic50_noisy = ic50 + np.random.normal(0, noise_std)
                    self.all_respond.append([cell, aug_s, ic50_noisy])

    def augment_by_drug_v2(self):
        # 生成 3个增强 smiles
        smiles2augsmiles = {}
        # 生成 1个增强 smiles
        smiles2augsmiles2 = {}
        AUGMENTATION_PROBABILITY = 0.3

        with open("/data0/liushujia/omics_new/high_missing_smiles.pkl", "rb") as f:
            aug_smiles_set = pickle.load(f)

        for s in aug_smiles_set:
            aug_smiles = self.augment_smiles(s, n_aug=3)
            smiles2augsmiles[s] = aug_smiles
            for aug_s in aug_smiles:
                self.smiles_morgan_fp_dict[aug_s] = self.smiles_to_morgan_fp(aug_s)
                self.smiles2drugid[aug_s] = "none"

        all_smiles_list = list(self.smiles2drugid.keys())
        for s in all_smiles_list:
            if s not in aug_smiles_set:
                # 只要一个就可以
                aug_smiles = self.augment_smiles(s, n_aug=1)
                smiles2augsmiles2[s] = aug_smiles[0]
                self.smiles_morgan_fp_dict[aug_smiles[0]] = self.smiles_to_morgan_fp(aug_smiles[0])
                self.smiles2drugid[aug_smiles[0]] = "none"

        for r in self.all_respond.copy():
            cell = r[0]
            smiles = r[1]
            ic50 = r[2]
            if smiles in aug_smiles_set:
                aug_smiles = smiles2augsmiles[smiles]
                for aug_s in aug_smiles:
                    self.all_respond.append([cell, aug_s, ic50])
            else:
                if random.random() < AUGMENTATION_PROBABILITY:
                    aug_s = smiles2augsmiles2[smiles]
                    self.all_respond.append([cell, aug_s, ic50])


    def augment_smiles(self, smiles, n_aug=3):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        augmented = []
        for _ in range(n_aug):
            idx = list(range(mol.GetNumAtoms()))
            random.shuffle(idx)  # 打乱原子顺序
            new_mol = Chem.RenumberAtoms(mol, idx)
            new_smiles = Chem.MolToSmiles(new_mol, canonical=False)
            augmented.append(new_smiles)
        return augmented

    def augment_smiles_rdkit(self, smiles, n_aug=3):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        aug_smiles = set()

        while len(aug_smiles) < n_aug:
            rand_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if rand_smiles != smiles:  # 排除原 SMILES
                aug_smiles.add(rand_smiles)

        return list(aug_smiles)

    def conformational_augmentation(self, smiles, n_confs=5):
        """基于构象的SMILES增强（去掉显式氢）"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # 添加氢原子以便构象搜索
        mol_with_h = Chem.AddHs(mol)

        augmented_smiles = []
        for i in range(n_confs):
            try:
                # 生成不同构象
                AllChem.EmbedMolecule(mol_with_h, randomSeed=i + 42)  # 每次不同随机种子
                mol_no_h = Chem.RemoveHs(mol_with_h)
                new_smiles = Chem.MolToSmiles(mol_no_h, canonical=False, isomericSmiles=True)
                augmented_smiles.append(new_smiles)
            except:
                continue

        return augmented_smiles

    def __len__(self):
        return len(self.all_respond)

    def __getitem__(self, idx):
        cell_line = self.all_respond[idx][0]
        smiles = self.all_respond[idx][1]
        y = torch.tensor([self.all_respond[idx][2]], dtype=torch.float)
        drugId = self.smiles2drugid[smiles]

        label_str = self.cellid2type[cell_line]  # e.g. "COAD"
        label = self.type2id[label_str]  # 转成 int

        return {
            "smiles": smiles,
            "morgan": self.smiles_morgan_fp_dict[smiles],
            "rdkit": self.smiles_rdkit_fp_dict[smiles],
            "gexpr": torch.tensor(self.gexpr_feature.loc[cell_line].tolist(), dtype=torch.float),
            "methylation": torch.tensor(self.methylation_feature.loc[cell_line].tolist(), dtype=torch.float),
            "mutation": torch.tensor(self.mutation_feature.loc[cell_line].tolist(), dtype=torch.float),
            "miRNA": torch.tensor(self.miRNA.loc[cell_line].tolist(), dtype=torch.float),
            "copynumber": torch.tensor(self.copynumber[cell_line].tolist(), dtype=torch.float),
            "drug_id": drugId,
            "cell_id": cell_line,
            "y": y,
            "type_label": torch.tensor(label, dtype=torch.long),  # 多分类标签
        }





