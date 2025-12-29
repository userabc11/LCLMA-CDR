import csv
import os
import pickle
import warnings
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# omics data
Cell_list_file = "/data0/liushujia/omics_new/cell_line_list.txt"

Gene_expression_ori_file = "/data0/liushujia/omics_new/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
Gene_expression_filter_file = "/data0/liushujia/omics_new/gene_expression_739dim.csv"

Methylation_ori_file = "/data0/liushujia/omics_new/DNA_methylation_20192dim.csv"
Methylation_filter_file = "/data0/liushujia/omics_new/DNA_methylation_comics_627dim.csv"

Gene_mutation_ori_file = "/data0/liushujia/omics_new/mutation_20113dim.csv"
Gene_mutation_filter_file = "/data0/liushujia/omics_new/mutation_hotspot_1076dim.csv"

MiRNA_file = "/data0/liushujia/omics_new/miRNA_735dim.csv"

Copy_number_file = "/data0/liushujia/omics_new/dna_copynumber_38590dim.csv"

# drug data
Drug_info_file = "/data0/liushujia/csv/PANCANCER_IC_Sat Jun 21 05_40_24 2025.csv"
#drug_smiles_file = '%s/223drugs_pubchem_smiles.txt'
Smiles_dict_file = "/data0/liushujia/omics_new/drugid2smiles.pkl"

# respond data
GDSC1_respond_file = "/data0/liushujia/omics_new/GDSC1_IC50.csv"
GDSC2_respond_file = "/data0/liushujia/omics_new/GDSC2_IC50.csv"
SANGER_respond_file = "/data0/liushujia/omics_new/sanger-response.csv"

LOAD_CONFIG_GDSC2 = {
    "Gene_expression_ori_file": False,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": False,
    "Methylation_filter_file": False,
    "Gene_mutation_ori_file": False,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": False,
    "Copy_number_file": False,
    "GDSC1": False,
    "GDSC2": True,
    "NAN_GDSC2":False,
    "SANGER_RESP":False
}

LOAD_CONFIG_GDSC1 = {
    "Gene_expression_ori_file": False,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": False,
    "Methylation_filter_file": False,
    "Gene_mutation_ori_file": False,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": False,
    "Copy_number_file": False,
    "GDSC1": True,
    "GDSC2": False,
    "NAN_GDSC2":False,
    "SANGER_RESP":False
}

NAN_CONFIG = {
    "Gene_expression_ori_file": False,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": False,
    "Methylation_filter_file": False,
    "Gene_mutation_ori_file": False,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": False,
    "Copy_number_file": False,
    "GDSC1": False,
    "GDSC2": False,
    "NAN_GDSC2": True,
    "SANGER_RESP":False
}

SANGER_CONFIG = {
    "Gene_expression_ori_file": False,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": False,
    "Methylation_filter_file": False,
    "Gene_mutation_ori_file": False,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": False,
    "Copy_number_file": False,
    "GDSC1": False,
    "GDSC2": False,
    "NAN_GDSC2": False,
    "SANGER_RESP":True
}
"""
    盲测加载数据的思想是把所有的数据都load进去，
    下游的方法再决定用哪些数据
"""
BLD_LOAD_CONFIG_GDSC2 = {
    "Gene_expression_ori_file": True,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": True,
    "Methylation_filter_file": True,
    "Gene_mutation_ori_file": True,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": True,
    "Copy_number_file": False,
    "GDSC1": False,
    "GDSC2": True
}

BLD_LOAD_CONFIG_GDSC1 = {
    "Gene_expression_ori_file": True,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": True,
    "Methylation_filter_file": True,
    "Gene_mutation_ori_file": True,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": True,
    "Copy_number_file": False,
    "GDSC1": True,
    "GDSC2": False
}

def loadCellLines():
    with open(Cell_list_file, "r") as f:
        cell_lines = [line.strip() for line in f if line.strip()]
    return cell_lines

def save_respond_csv(all_respond, output_file):
    # 获取所有药物和细胞系列表
    drugs = sorted(list(set([item[1] for item in all_respond])))
    cell_lines = sorted(list(set([item[0] for item in all_respond])))

    df = pd.DataFrame("NA", index=drugs, columns=cell_lines)

    # 填充数据
    for cell_line, smiles, ln_IC50 in all_respond:
        df.loc[smiles, cell_line] = ln_IC50

    df.to_csv(output_file)
    print(f"保存完成：{output_file}, shape={df.shape}")
    return df

def RespondFromCSV(drugid2smiles, Cancer_response_file, all_cellines, is_raw = False):
    # Cancer_response_exp_file中记录药物在细胞系中的IC50值
    respond_data = pd.read_csv(Cancer_response_file, sep=',', header=0, index_col=[0])
    print("orign respond_data：",respond_data.shape)
    """
    # 筛选掉 NaN 值超过一半的行
    threshold = int(respond_data.shape[1]*0.5)  # 列数的一半
    respond_data = respond_data.dropna(thresh=threshold, axis=0)
    print("去除NAN后，respond_data：", respond_data.shape)
    """
    # 过滤experiment_data
    drug_match_list = [item for item in respond_data.index if item.split(':')[1] in drugid2smiles.keys()]
    filtered_respond_data = respond_data.loc[drug_match_list]    #filter by drug list
    cell_match_list = [item for item in filtered_respond_data.columns if item in all_cellines]
    filtered_respond_data = filtered_respond_data[cell_match_list]  #filter by cell-line list

    all_respond = []
    for each_drug in filtered_respond_data.index:
        for each_cellline in filtered_respond_data.columns:
            smiles = drugid2smiles[each_drug.split(':')[-1]]
            ln_IC50 = float(filtered_respond_data.loc[each_drug,each_cellline])
            if is_raw and not np.isnan(ln_IC50):
                ln_IC50 = np.log(ln_IC50)
                all_respond.append([each_cellline, smiles, ln_IC50])
                continue
            if not np.isnan(ln_IC50):
                all_respond.append([each_cellline,smiles,ln_IC50])

    nb_celllines = len(set([item[0] for item in all_respond]))
    nb_drugs = len(set([item[1] for item in all_respond]))
    print('产生了%d 对反应包括 %d 个细胞系和 %d 种药物.' % (len(all_respond), nb_celllines, nb_drugs))
    return all_respond

def NARespondFromCSV(drugid2smiles, Cancer_response_file, all_cellines):
    # 读取数据
    respond_data = pd.read_csv(Cancer_response_file, sep=',', header=0, index_col=[0], na_values=["NA", "NaN", "N/A", ""])
    print("原始 respond_data:", respond_data.shape)

    # ① 过滤 drug list
    drug_match_list = [item for item in respond_data.index if item.split(':')[-1] in drugid2smiles.keys()]
    filtered_respond_data = respond_data.loc[drug_match_list]

    # ② 过滤 cell-line list
    cell_match_list = [item for item in filtered_respond_data.columns if item in all_cellines]
    filtered_respond_data = filtered_respond_data[cell_match_list]
    print("筛选后 respond_data:", filtered_respond_data.shape)

    # ③ 去除缺失率 > 50% 的药物行
    valid_drug_mask = filtered_respond_data.isna().mean(axis=1) >= 0.5
    filtered_respond_data = filtered_respond_data.loc[valid_drug_mask]
    print("高缺失反应：:", filtered_respond_data.shape)

    # ④ 提取所有 NA 的反应
    all_respond = []
    for each_drug in filtered_respond_data.index:
        smiles = drugid2smiles[each_drug.split(':')[-1]]
        for each_cellline in filtered_respond_data.columns:
            ln_IC50 = filtered_respond_data.loc[each_drug, each_cellline]
            if pd.isna(ln_IC50):
                all_respond.append([each_cellline, smiles, -999])  # 缺失值标记为0

    nb_celllines = len(set([item[0] for item in all_respond]))
    nb_drugs = len(set([item[1] for item in all_respond]))
    print('产生了 %d 对 NA 反应，包括 %d 个细胞系和 %d 种药物.' % (len(all_respond), nb_celllines, nb_drugs))

    return all_respond


def loadDataFromFiles(config):
    SAVE_CSV = False
    cell_line_list = loadCellLines()
    results = {}
    # 读取多组学数据，数据对齐
    if(config["Gene_expression_ori_file"]):
        gexpr_ori = pd.read_csv(Gene_expression_ori_file, index_col=0)
        gexpr_ori = gexpr_ori.loc[cell_line_list]
        print("load gexpr_ori:",gexpr_ori.shape)
        results["Gene_expression_ori"] = gexpr_ori

    if (config["Gene_expression_filter_file"]):
        gexpr_filter = pd.read_csv(Gene_expression_filter_file, index_col=0)
        gexpr_filter = gexpr_filter.loc[cell_line_list]
        print("load gexpr_filter:", gexpr_filter.shape)
        results["Gene_expression_filter"] = gexpr_filter

    if (config["Methylation_ori_file"]):
        methy_ori = pd.read_csv(Methylation_ori_file,index_col=0)
        methy_ori = methy_ori.loc[cell_line_list]
        print("load methy_ori:", methy_ori.shape)
        results["Methylation_ori"] = methy_ori

    if (config["Methylation_filter_file"]):
        methy_filter = pd.read_csv(Methylation_filter_file,index_col=0)
        methy_filter = methy_filter.loc[cell_line_list]
        print("load methy_filter:", methy_filter.shape)
        results["Methylation_filter"] = methy_filter

    if (config["Gene_mutation_ori_file"]):
        mut_ori = pd.read_csv(Gene_mutation_ori_file,index_col=0)
        mut_ori = mut_ori.loc[cell_line_list]
        print("load mut_ori:", mut_ori.shape)
        results["Gene_mutation_ori"] = mut_ori

    if (config["Gene_mutation_filter_file"]):
        mut_filter = pd.read_csv(Gene_mutation_filter_file,index_col=0)
        mut_filter = mut_filter.loc[cell_line_list]
        print("load mut_filter:", mut_filter.shape)
        results["Gene_mutation_filter"] = mut_filter

    if (config["MiRNA_file"]):
        mi_rna = pd.read_csv(MiRNA_file,index_col=0)
        mi_rna = mi_rna.loc[cell_line_list]
        print("load mi_rna:", mi_rna.shape)
        results["MiRNA"] = mi_rna

    if (config["Copy_number_file"]):
        copy_number = pd.read_csv(Copy_number_file,index_col=0)
        copy_number = copy_number.loc[cell_line_list]
        print("load copy_number:", copy_number.shape)
        results["Copy_number"] = copy_number

    # #common = set(gexpr_filter.index) & set(methy_filter.index) & set(mut_filter.index) & set(mi_rna.index) & set(copy_number.index)
    # common = set(gexpr_filter.index) & set(methy_filter.index) & set(mut_filter.index) & set(mi_rna.index)
    # print(len(common))
    # # 保存到文件
    # with open("/data0/liushujia/omics_new/cell_line_list.txt", "w") as f:
    #     for cid in common:
    #         f.write(f"{cid}\n")

    # 从drug-cellline反应的csv文件中提取反应
    with open(Smiles_dict_file, "rb") as f:  # 注意这里是 "rb"
        drugid2smiles = pickle.load(f)
        drugid2smiles = {str(k): v for k, v in drugid2smiles.items()}
        results["Drugid2smiles"] = drugid2smiles

    # 1️⃣ 统计选择的模式数量
    selected = [config.get("GDSC1", False), config.get("GDSC2", False), config.get("NAN_GDSC2", False), config.get("SANGER_RESP", False)]
    selected_count = sum(selected)
    # 2️⃣ 检查选择合法性
    if selected_count > 1:
        raise Exception("❌ Config error: only one of [GDSC1, GDSC2, NAN_GDSC2, SANGER_RESP] can be True.")
    if selected_count == 0:
        warnings.warn("⚠️ No response file selected — please enable one of [GDSC1, GDSC2, NAN_GDSC2, SANGER_RESP].")
        return results

    if(config["GDSC1"]):
        print("load respond from GDSC1")
        all_respond = RespondFromCSV(drugid2smiles, GDSC1_respond_file, cell_line_list)
        results["Responds"] = all_respond
        if SAVE_CSV:
            save_respond_csv(all_respond, "/data0/liushujia/omics_new/nest_gdsc1.csv")
    if (config["GDSC2"]):
        print("load respond from GDSC2")
        all_respond = RespondFromCSV(drugid2smiles, GDSC2_respond_file, cell_line_list)
        if SAVE_CSV:
            save_respond_csv(all_respond, "/data0/liushujia/omics_new/nest_gdsc2.csv")
        results["Responds"] = all_respond
    if (config["NAN_GDSC2"]):
        print("load nan respond from GDSC2")
        all_respond = NARespondFromCSV(drugid2smiles, GDSC2_respond_file, cell_line_list)
        results["Responds"] = all_respond
    if (config["SANGER_RESP"]):
        print("load sanger respond")
        results["Responds"] = RespondFromCSV(drugid2smiles, SANGER_respond_file, cell_line_list, is_raw=True)
    return results


# 生成盲测 5折数据，并存入指定目录下
def loadDrugBldData5Fold_and_save(config, save_dir = "/data0/liushujia/gdsc2-drugbld/", seed=42, n_splits=5):
    os.makedirs(save_dir, exist_ok=True)

    cell_line_list = loadCellLines()
    results = {}

    # -------------------------
    # 1. 读取多组学数据
    # -------------------------
    if (config["Gene_expression_ori_file"]):
        gexpr_ori = pd.read_csv(Gene_expression_ori_file, index_col=0)
        gexpr_ori = gexpr_ori.loc[cell_line_list]
        print("load gexpr_ori:", gexpr_ori.shape)
        results["Gene_expression_ori"] = gexpr_ori

    if (config["Gene_expression_filter_file"]):
        gexpr_filter = pd.read_csv(Gene_expression_filter_file, index_col=0)
        gexpr_filter = gexpr_filter.loc[cell_line_list]
        print("load gexpr_filter:", gexpr_filter.shape)
        results["Gene_expression_filter"] = gexpr_filter

    if (config["Methylation_ori_file"]):
        methy_ori = pd.read_csv(Methylation_ori_file, index_col=0)
        methy_ori = methy_ori.loc[cell_line_list]
        print("load methy_ori:", methy_ori.shape)
        results["Methylation_ori"] = methy_ori

    if (config["Methylation_filter_file"]):
        methy_filter = pd.read_csv(Methylation_filter_file, index_col=0)
        methy_filter = methy_filter.loc[cell_line_list]
        print("load methy_filter:", methy_filter.shape)
        results["Methylation_filter"] = methy_filter

    if (config["Gene_mutation_ori_file"]):
        mut_ori = pd.read_csv(Gene_mutation_ori_file, index_col=0)
        mut_ori = mut_ori.loc[cell_line_list]
        print("load mut_ori:", mut_ori.shape)
        results["Gene_mutation_ori"] = mut_ori

    if (config["Gene_mutation_filter_file"]):
        mut_filter = pd.read_csv(Gene_mutation_filter_file, index_col=0)
        mut_filter = mut_filter.loc[cell_line_list]
        print("load mut_filter:", mut_filter.shape)
        results["Gene_mutation_filter"] = mut_filter

    if (config["MiRNA_file"]):
        mi_rna = pd.read_csv(MiRNA_file, index_col=0)
        mi_rna = mi_rna.loc[cell_line_list]
        print("load mi_rna:", mi_rna.shape)
        results["MiRNA"] = mi_rna

    if (config["Copy_number_file"]):
        copy_number = pd.read_csv(Copy_number_file, index_col=0)
        copy_number = copy_number.loc[cell_line_list]
        print("load copy_number:", copy_number.shape)
        results["Copy_number"] = copy_number

    # 2. 读取 drug smiles
    with open(Smiles_dict_file, "rb") as f:  # 注意这里是 "rb"
        drugid2smiles = pickle.load(f)
        drugid2smiles = {str(k): v for k, v in drugid2smiles.items()}
        results["Drugid2smiles"] = drugid2smiles

    # -------------------------
    # 3. 读取 drug-cell line 反应
    # -------------------------
    if config["GDSC1"] and config["GDSC2"]:
        raise Exception("Cannot use both GDSC1 and GDSC2, use one!")
    if not config["GDSC1"] and not config["GDSC2"]:
        raise Warning("No response file is selected")

    if config["GDSC1"]:
        response_file = GDSC1_respond_file
    else:
        response_file = GDSC2_respond_file
    respond_data = pd.read_csv(response_file, sep=',', header=0, index_col=0)
    print("原始 respond_data：", respond_data.shape)

    # 过滤 drugs & cell-lines
    drug_match_list = [item for item in respond_data.index if item.split(':')[-1] in drugid2smiles.keys()]
    filtered_respond_data = respond_data.loc[drug_match_list]
    cell_match_list = [item for item in filtered_respond_data.columns if item in cell_line_list]
    filtered_respond_data = filtered_respond_data[cell_match_list]

    drug_list = list(filtered_respond_data.index)
    random.seed(seed)
    random.shuffle(drug_list)

    # -------------------------
    # 4. 5-fold 划分
    # -------------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(drug_list), 1):
        train_drugs = [drug_list[i] for i in train_idx]
        test_drugs = [drug_list[i] for i in test_idx]

        # validation 从 train_drugs 随机抽 10%
        n_valid = max(1, int(len(train_drugs) * 0.1))
        valid_drugs = random.sample(train_drugs, n_valid)
        train_drugs = list(set(train_drugs) - set(valid_drugs))

        def build_respond(drug_set):
            res = []
            for each_drug in drug_set:
                for each_cellline in filtered_respond_data.columns:
                    ln_IC50 = filtered_respond_data.loc[each_drug, each_cellline]
                    if not np.isnan(ln_IC50):
                        smiles = drugid2smiles[each_drug.split(':')[-1]]
                        res.append([each_cellline, smiles, float(ln_IC50)])
            return res

        all_respond_train = build_respond(train_drugs)
        all_respond_valid = build_respond(valid_drugs)
        all_respond_test = build_respond(test_drugs)

        results["Train"] = all_respond_train
        results["Val"] = all_respond_valid
        results["Test"] = all_respond_test


        # 保存到磁盘
        with open(os.path.join(save_dir, f"results_fold{fold_idx}.pkl"), "wb") as f:
            pickle.dump(results, f)
        print(f"fold{fold_idx} 数据已保存到 {save_dir}")


# 生成盲测 5折数据，并存入指定目录下
def loadCellBldData5Fold_and_save(config, save_dir = "/data0/liushujia/gdsc2-cellbld/", seed=42, n_splits=5):
    os.makedirs(save_dir, exist_ok=True)

    cell_line_list = loadCellLines()
    results = {}

    # -------------------------
    # 1. 读取多组学数据
    # -------------------------
    if (config["Gene_expression_ori_file"]):
        gexpr_ori = pd.read_csv(Gene_expression_ori_file, index_col=0)
        gexpr_ori = gexpr_ori.loc[cell_line_list]
        print("load gexpr_ori:", gexpr_ori.shape)
        results["Gene_expression_ori"] = gexpr_ori

    if (config["Gene_expression_filter_file"]):
        gexpr_filter = pd.read_csv(Gene_expression_filter_file, index_col=0)
        gexpr_filter = gexpr_filter.loc[cell_line_list]
        print("load gexpr_filter:", gexpr_filter.shape)
        results["Gene_expression_filter"] = gexpr_filter

    if (config["Methylation_ori_file"]):
        methy_ori = pd.read_csv(Methylation_ori_file, index_col=0)
        methy_ori = methy_ori.loc[cell_line_list]
        print("load methy_ori:", methy_ori.shape)
        results["Methylation_ori"] = methy_ori

    if (config["Methylation_filter_file"]):
        methy_filter = pd.read_csv(Methylation_filter_file, index_col=0)
        methy_filter = methy_filter.loc[cell_line_list]
        print("load methy_filter:", methy_filter.shape)
        results["Methylation_filter"] = methy_filter

    if (config["Gene_mutation_ori_file"]):
        mut_ori = pd.read_csv(Gene_mutation_ori_file, index_col=0)
        mut_ori = mut_ori.loc[cell_line_list]
        print("load mut_ori:", mut_ori.shape)
        results["Gene_mutation_ori"] = mut_ori

    if (config["Gene_mutation_filter_file"]):
        mut_filter = pd.read_csv(Gene_mutation_filter_file, index_col=0)
        mut_filter = mut_filter.loc[cell_line_list]
        print("load mut_filter:", mut_filter.shape)
        results["Gene_mutation_filter"] = mut_filter

    if (config["MiRNA_file"]):
        mi_rna = pd.read_csv(MiRNA_file, index_col=0)
        mi_rna = mi_rna.loc[cell_line_list]
        print("load mi_rna:", mi_rna.shape)
        results["MiRNA"] = mi_rna

    if (config["Copy_number_file"]):
        copy_number = pd.read_csv(Copy_number_file, index_col=0)
        copy_number = copy_number.loc[cell_line_list]
        print("load copy_number:", copy_number.shape)
        results["Copy_number"] = copy_number

    # 2. 读取 drug smiles
    with open(Smiles_dict_file, "rb") as f:  # 注意这里是 "rb"
        drugid2smiles = pickle.load(f)
        drugid2smiles = {str(k): v for k, v in drugid2smiles.items()}
        results["Drugid2smiles"] = drugid2smiles

    # -------------------------
    # 3. 读取 drug-cell line 反应
    # -------------------------
    if config["GDSC1"] and config["GDSC2"]:
        raise Exception("Cannot use both GDSC1 and GDSC2, use one!")
    if not config["GDSC1"] and not config["GDSC2"]:
        raise Warning("No response file is selected")

    if config["GDSC1"]:
        response_file = GDSC1_respond_file
    else:
        response_file = GDSC2_respond_file
    respond_data = pd.read_csv(response_file, sep=',', header=0, index_col=0)
    print("原始 respond_data：", respond_data.shape)

    # 过滤 drugs & cell-lines
    drug_match_list = [item for item in respond_data.index if item.split(':')[-1] in drugid2smiles.keys()]
    filtered_respond_data = respond_data.loc[drug_match_list]
    cell_match_list = [item for item in filtered_respond_data.columns if item in cell_line_list]
    filtered_respond_data = filtered_respond_data[cell_match_list]

    # -------------------------
    # 4. 基于 Cell 的 5-fold 划分
    # -------------------------
    cell_list = list(filtered_respond_data.columns)
    random.seed(seed)
    random.shuffle(cell_list)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(cell_list), 1):
        train_cells = [cell_list[i] for i in train_idx]
        test_cells = [cell_list[i] for i in test_idx]

        # validation 从 train_cells 随机抽 10%
        n_valid = max(1, int(len(train_cells) * 0.1))
        valid_cells = random.sample(train_cells, n_valid)
        train_cells = list(set(train_cells) - set(valid_cells))

        def build_respond(cell_set):
            res = []
            for each_cellline in cell_set:
                for each_drug in filtered_respond_data.index:
                    ln_IC50 = filtered_respond_data.loc[each_drug, each_cellline]
                    if not np.isnan(ln_IC50):
                        smiles = drugid2smiles[each_drug.split(':')[-1]]
                        res.append([each_cellline, smiles, float(ln_IC50)])
            return res

        all_respond_train = build_respond(train_cells)
        all_respond_valid = build_respond(valid_cells)
        all_respond_test = build_respond(test_cells)

        results["Train"] = all_respond_train
        results["Val"] = all_respond_valid
        results["Test"] = all_respond_test

        # 保存到磁盘
        with open(os.path.join(save_dir, f"results_fold{fold_idx}.pkl"), "wb") as f:
            pickle.dump(results, f)
        print(f"Cell-based fold{fold_idx} 数据已保存到 {save_dir}")


def respondCCLE_to_csv(output_csv="/data0/liushujia/omics_new/sanger-response-cleaned.csv"):
    # === 3. 读取 CCLE 反应数据
    df = pd.read_csv("/data0/liushujia/omics_new/sanger-response.csv")
    df = df[["ARXSPAN_ID", "DRUG_ID", "IC50_PUBLISHED"]].dropna().copy()
    df["DRUG_ID"] = "GDSC:" + df["DRUG_ID"].astype(str)

    # === 7. 生成 drug × cell line 的矩阵
    matrix = df.pivot_table(
        index="DRUG_ID",
        columns="ARXSPAN_ID",
        values="IC50_PUBLISHED",
        aggfunc="mean"
    )
    matrix.to_csv(output_csv, na_rep="NA")
    print(f"矩阵已保存到：{output_csv}")

#respondCCLE_to_csv()
#r1 = loadDataFromFiles(LOAD_CONFIG_GDSC1)
#r2 = loadDataFromFiles(LOAD_CONFIG_GDSC2)
#r_nan= loadDataFromFiles(NAN_CONFIG)
#print("**")

#loadCellBldData5Fold_and_save(BLD_LOAD_CONFIG_GDSC2)

# GDSC1 盲测数据生成
# loadDrugBldData5Fold_and_save(BLD_LOAD_CONFIG_GDSC1, save_dir = "/data0/liushujia/gdsc1-drugbld/", seed=42, n_splits=5)
# loadCellBldData5Fold_and_save(BLD_LOAD_CONFIG_GDSC1, save_dir = "/data0/liushujia/gdsc1-cellbld/", seed=42, n_splits=5)

# with open(Smiles_dict_file, "rb") as f:  # 注意这里是 "rb"
#     drugid2smiles = pickle.load(f)

#loadDataFromFiles(LOAD_CONFIG)
# s = loadCellLines()
# q = pd.read_csv(Gene_mutation_ori_file, index_col=0)
# print(len(s & set(q.index)))

# cell_line_list = loadCellLines()
# respond_data = pd.read_csv(respond_file, sep=',', header=0, index_col=[0])
# r_cells = set(respond_data.columns)
# gexpr_filter = pd.read_csv(Gene_expression_filter_file, index_col=0)
# print(len(r_cells & set(gexpr_filter.index)))
#
# methy_filter = pd.read_csv(Methylation_filter_file, index_col=0)
# print(len(r_cells & set(methy_filter.index)))
#
# mut_filter = pd.read_csv(Gene_mutation_filter_file, index_col=0)
# print(len(r_cells & set(mut_filter.index)))
#
# mirna_filter = pd.read_csv(MiRNA_file, index_col=0)
# print(len(r_cells & set(mirna_filter.index)))
#
# copy_filter = pd.read_csv(Copy_number_file, index_col=0)
# print(len(r_cells & set(copy_filter.index)))

# gexpr_filter = pd.read_csv(Gene_expression_filter_file, index_col=0)
# columns = gexpr_filter.columns
# for c in columns:
#     print(c)
# print(gexpr_filter.shape)
# methy_filter = pd.read_csv(Methylation_filter_file, index_col=0)
# print(methy_filter.shape)
# mut_filter = pd.read_csv(Gene_mutation_filter_file, index_col=0)
# print(mut_filter.shape)
# mirna_filter = pd.read_csv(MiRNA_file, index_col=0)
# print(mirna_filter.shape)
# copy_filter = pd.read_csv(Copy_number_file, index_col=0)
# print(copy_filter.shape)
# print(len(set(gexpr_filter.index) & set(methy_filter.index) & set(mut_filter.index) & set(mirna_filter.index) & set(copy_filter.index)))

# respond_data = pd.read_csv(GDSC2_respond_file, sep=',', header=0, index_col=[0])
# columns = respond_data.columns
# for c in columns:
#     print(c)