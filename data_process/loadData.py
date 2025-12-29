import csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DPATH = "/home/liushujia/Graphormer-master/data_process/data/"

# omics data
Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'%DPATH
Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'%DPATH
Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv'%DPATH

# drug data
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
drug_smiles_file = '%s/223drugs_pubchem_smiles.txt'%DPATH

# respond data
Cancer_response_file = '%s/CCLE/GDSC_IC50.csv'%DPATH
adequent_test_file = '%s/CCLE/adequentTest.csv'%DPATH
blind_test_file = '%s/CCLE/blindTest.csv'%DPATH

# drug blind
drug_bld_train_file = './drug_bld_train.csv'
drug_bld_test_file = './drug_bld_test.csv'
drug_bld_val_file = './drug_bld_val.csv'

# cell blind
cell_bld_train_file = '/data0/liushujia/cell_blind/cell_bld_train.csv'
cell_bld_test_file = '/data0/liushujia/cell_blind/cell_bld_test.csv'
cell_bld_val_file = '/data0/liushujia/cell_blind/cell_bld_val.csv'

all_respond_data = '%s/CCLE/GDSC_IC50.csv'%DPATH

def DrugidToSmiles():
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}
    pubchemid2smiles = {item.split('\t')[0]: item.split('\t')[1].strip() for item in open(drug_smiles_file).readlines()}
    drugid2smiles = {}
    for id in drugid2pubchemid.keys():
        if pubchemid2smiles.get(drugid2pubchemid[id]) is not None:
            drugid2smiles[id] = pubchemid2smiles[drugid2pubchemid[id]]

    return drugid2smiles

def RespondFromCSV(drugid2smiles, Cancer_response_file, all_cellines):
    # Cancer_response_exp_file中记录药物在细胞系中的IC50值
    respond_data = pd.read_csv(Cancer_response_file, sep=',', header=0, index_col=[0])
    print("respond_data：",respond_data.shape)
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
            if not np.isnan(ln_IC50):
                all_respond.append([each_cellline,smiles,ln_IC50])

    nb_celllines = len(set([item[0] for item in all_respond]))
    nb_drugs = len(set([item[1] for item in all_respond]))
    print('产生了%d 对反应包括 %d 个细胞系和 %d 种药物.' % (len(all_respond), nb_celllines, nb_drugs))
    return all_respond

def NARespondFromCSV(drugid2smiles, Cancer_response_file, all_cellines):
    # Cancer_response_exp_file中记录药物在细胞系中的IC50值
    respond_data = pd.read_csv(Cancer_response_file, sep=',', header=0, index_col=[0])
    print("respond_data：",respond_data.shape)
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
            if np.isnan(ln_IC50):
                all_respond.append([each_cellline,smiles,0])

    nb_celllines = len(set([item[0] for item in all_respond]))
    nb_drugs = len(set([item[1] for item in all_respond]))
    print('产生了%d 对反应包括 %d 个细胞系和 %d 种药物.' % (len(all_respond), nb_celllines, nb_drugs))
    return all_respond

def loadDataFromFiles():
    all_cellines = []
    # 读取多组学数据，数据对齐
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])

    # “对齐”多组学数据
    all_cellines = set(list(gexpr_feature.index)) & set(list(mutation_feature.index)) & set(list(methylation_feature.index))
    all_cellines = list(all_cellines)
    print("对齐后，剩余的细胞系数量为：",len(all_cellines))
    gexpr_feature = gexpr_feature.loc[all_cellines]
    mutation_feature = mutation_feature.loc[all_cellines]
    methylation_feature = methylation_feature.loc[all_cellines]
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    # 归一化处理
    scaler = StandardScaler()
    gexpr_feature_std = scaler.fit_transform(gexpr_feature)
    methylation_feature_std = scaler.fit_transform(methylation_feature)
    gexpr_feature = pd.DataFrame(gexpr_feature_std, index=gexpr_feature.index, columns=gexpr_feature.columns)
    methylation_feature = pd.DataFrame(methylation_feature_std,index=methylation_feature.index, columns=methylation_feature.columns)

    mutation_feature = mutation_feature.sum(axis=1)
    print(mutation_feature)
    l = mutation_feature.tolist()
    l.sort()
    print("mutation_feature分布情况 ",l)

    # drugid2smiles(字典) 建立drugid -> smiles的映射
    drugid2smiles = DrugidToSmiles()

    # 从drug-cellline反应的csv文件中提取反应
    #all_respond = []
    #all_respond = RespondFromCSV(drugid2smiles, all_cellines)

    # 分别读取 adequent_test和 blind_test反应数据
    adeq_resspond = RespondFromCSV(drugid2smiles, adequent_test_file, all_cellines)
    print(f"CDR test中生成了{len(adeq_resspond)}对反应")

    blind_resspond = RespondFromCSV(drugid2smiles, blind_test_file, all_cellines)
    print(f"Blind test中生成了{len(blind_resspond)}对反应")

    return gexpr_feature, mutation_feature, methylation_feature \
        , adeq_resspond, blind_resspond, drugid2smiles

def loadDrugBlindTestDataFromFiles():
    # 读取多组学数据，数据对齐
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])

    # “对齐”多组学数据
    all_cellines = set(list(gexpr_feature.index)) & set(list(mutation_feature.index)) & set(list(methylation_feature.index))
    all_cellines = list(all_cellines)
    print("对齐后，剩余的细胞系数量为：",len(all_cellines))
    gexpr_feature = gexpr_feature.loc[all_cellines]
    mutation_feature = mutation_feature.loc[all_cellines]
    methylation_feature = methylation_feature.loc[all_cellines]
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    # 归一化处理
    scaler = StandardScaler()
    gexpr_feature_std = scaler.fit_transform(gexpr_feature)
    methylation_feature_std = scaler.fit_transform(methylation_feature)
    gexpr_feature = pd.DataFrame(gexpr_feature_std, index=gexpr_feature.index, columns=gexpr_feature.columns)
    methylation_feature = pd.DataFrame(methylation_feature_std,index=methylation_feature.index, columns=methylation_feature.columns)

    mutation_feature = mutation_feature.sum(axis=1)
    print(mutation_feature)
    l = mutation_feature.tolist()
    l.sort()
    print("mutation_feature分布情况 ",l)

    # drugid2smiles(字典) 建立drugid -> smiles的映射
    drugid2smiles = DrugidToSmiles()

    # 分别读取 adequent_test和 blind_test反应数据
    print("加载药物盲测数据")
    train_resspond = RespondFromCSV(drugid2smiles, drug_bld_train_file, all_cellines)
    print(f"盲测训练集中生成了{len(train_resspond)}对反应")

    test_resspond = RespondFromCSV(drugid2smiles, drug_bld_test_file, all_cellines)
    print(f"盲测测试集中生成了{len(test_resspond)}对反应")

    val_resspond = RespondFromCSV(drugid2smiles, drug_bld_val_file, all_cellines)
    print(f"盲测验证集中生成了{len(val_resspond)}对反应")
    return gexpr_feature, mutation_feature, methylation_feature \
        , train_resspond, test_resspond, val_resspond, drugid2smiles

def loadCellBlindTestDataFromFiles():
    # 读取多组学数据，数据对齐
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])

    # “对齐”多组学数据
    all_cellines = set(list(gexpr_feature.index)) & set(list(mutation_feature.index)) & set(list(methylation_feature.index))
    all_cellines = list(all_cellines)
    print("对齐后，剩余的细胞系数量为：",len(all_cellines))
    gexpr_feature = gexpr_feature.loc[all_cellines]
    mutation_feature = mutation_feature.loc[all_cellines]
    methylation_feature = methylation_feature.loc[all_cellines]
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    # 归一化处理
    scaler = StandardScaler()
    gexpr_feature_std = scaler.fit_transform(gexpr_feature)
    methylation_feature_std = scaler.fit_transform(methylation_feature)
    gexpr_feature = pd.DataFrame(gexpr_feature_std, index=gexpr_feature.index, columns=gexpr_feature.columns)
    methylation_feature = pd.DataFrame(methylation_feature_std,index=methylation_feature.index, columns=methylation_feature.columns)

    mutation_feature = mutation_feature.sum(axis=1)
    print(mutation_feature)
    l = mutation_feature.tolist()
    l.sort()
    print("mutation_feature分布情况 ",l)

    # drugid2smiles(字典) 建立drugid -> smiles的映射
    drugid2smiles = DrugidToSmiles()

    # 分别读取 adequent_test和 blind_test反应数据
    print("加载细胞系盲测数据")
    train_resspond = RespondFromCSV(drugid2smiles, cell_bld_train_file, all_cellines)
    print(f"细胞系盲测训练集中生成了{len(train_resspond)}对反应")

    test_resspond = RespondFromCSV(drugid2smiles, cell_bld_test_file, all_cellines)
    print(f"细胞系盲测测试集中生成了{len(test_resspond)}对反应")

    val_resspond = RespondFromCSV(drugid2smiles, cell_bld_val_file, all_cellines)
    print(f"细胞系盲测验证集中生成了{len(val_resspond)}对反应")
    return gexpr_feature, mutation_feature, methylation_feature \
        , train_resspond, test_resspond, val_resspond, drugid2smiles

def loadExitAndNaDataFromFile():
    all_cellines = []
    # 读取多组学数据，数据对齐
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])

    # “对齐”多组学数据
    all_cellines = set(list(gexpr_feature.index)) & set(list(mutation_feature.index)) & set(list(methylation_feature.index))
    all_cellines = list(all_cellines)
    print("对齐后，剩余的细胞系数量为：",len(all_cellines))
    gexpr_feature = gexpr_feature.loc[all_cellines]
    mutation_feature = mutation_feature.loc[all_cellines]
    methylation_feature = methylation_feature.loc[all_cellines]
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    # 归一化处理
    scaler = StandardScaler()
    gexpr_feature_std = scaler.fit_transform(gexpr_feature)
    methylation_feature_std = scaler.fit_transform(methylation_feature)
    gexpr_feature = pd.DataFrame(gexpr_feature_std, index=gexpr_feature.index, columns=gexpr_feature.columns)
    methylation_feature = pd.DataFrame(methylation_feature_std,index=methylation_feature.index, columns=methylation_feature.columns)

    mutation_feature = mutation_feature.sum(axis=1)

    # drugid2smiles(字典) 建立drugid -> smiles的映射
    drugid2smiles = DrugidToSmiles()

    # 从drug-cellline反应的csv文件中提取反应
    exit_respond = RespondFromCSV(drugid2smiles, all_respond_data, all_cellines)
    print(f"已知反应中生成了{len(exit_respond)}对反应")

    nan_respond = NARespondFromCSV(drugid2smiles, all_respond_data, all_cellines)
    print(f"未知反应中生成了{len(nan_respond)}对反应")

    return gexpr_feature, mutation_feature, methylation_feature \
        , exit_respond, nan_respond, drugid2smiles
