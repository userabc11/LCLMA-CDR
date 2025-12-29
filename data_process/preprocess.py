import csv
import pickle
import random

import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from tqdm import tqdm

def count_matching_cell_lines(csv1_path, csv2_path):
    # 读入两个CSV
    df1 = pd.read_csv(csv1_path, sep=",")
    df2 = pd.read_csv(csv2_path, sep=",")

    # 取出 cell line 名称
    gdsc_cell_lines = df1["Cell Line Name"].str.strip().str.upper()
    depmap_cell_lines = df2["CellLineName"].str.strip().str.upper()
    depmap_stripped = df2["StrippedCellLineName"].str.strip().str.upper()

    # 匹配：先看直接能匹配的
    direct_matches = gdsc_cell_lines[gdsc_cell_lines.isin(depmap_cell_lines)]

    # 再看 stripped 能匹配的
    stripped_matches = gdsc_cell_lines[gdsc_cell_lines.isin(depmap_stripped)]

    # 统计总 unique 匹配数量
    total_matches = pd.concat([direct_matches, stripped_matches]).nunique()

    return {
        "direct_match_count": direct_matches.nunique(),
        "stripped_match_count": stripped_matches.nunique(),
        "total_unique_matches": total_matches
    }


def count_drugs_and_cells(csv_path):
    # 读取csv
    df = pd.read_csv(csv_path)

    num_drugs = df['Drug ID'].nunique()
    num_cells = df['Cosmic ID'].nunique()

    return {
        "drug_num": num_drugs,
        "cell_line_num": num_cells
    }

def filter_expression_by_cgc(expr_path, cgc_path, save_path=None):
    # 读入表达矩阵 (行=cell line，列=gene symbol (name + id))
    expr = pd.read_csv(expr_path, index_col=0)

    # 处理表达矩阵的基因名，去掉括号和空格
    expr.columns = expr.columns.str.split(" ").str[0]

    # 读入CGC
    cgc = pd.read_csv(cgc_path, sep="\t")
    genes = cgc['GENE_SYMBOL'].astype(str)

    # 取交集
    common_genes = expr.columns.intersection(genes)
    filtered_expr = expr[common_genes]
    print("filtered_expr:",filtered_expr.shape)
    if save_path:
        filtered_expr.to_csv(save_path, index=True, header=True)

    print(f"原始基因数: {expr.shape[1]}, CGC基因数: {len(genes)}, 交集: {len(common_genes)}")


def binarize_mutation(mutation_path, save_path=None):
    mut = pd.read_csv(mutation_path)

    # 只保留我们需要的字段
    mut = mut[['ModelID', 'HugoSymbol']].dropna()

    # 去掉没基因名字的情况
    mut = mut[mut['HugoSymbol'] != ""]

    # 建立二值矩阵
    binary_mut = (
        mut
        .assign(value=1)
        .drop_duplicates()  # 避免重复的 ModelID-基因 对
        .pivot(index='ModelID', columns='HugoSymbol', values='value')
        .fillna(0)
        .astype(int)
    )

    # 保存
    if save_path:
        binary_mut.to_csv(save_path, index=True, header=True)

    print("生成的二值化矩阵:",binary_mut.shape)


def binarize_mutation_on_hotspot(mutation_path, save_path=None):
    mut = pd.read_csv(mutation_path)

    # 只保留需要的字段
    required_cols = ['ModelID', 'HugoSymbol', 'Hotspot', 'OncogeneHighImpact', 'TumorSuppressorHighImpact']
    mut = mut[required_cols].dropna(subset=['ModelID', 'HugoSymbol'])

    for col in ['Hotspot', 'OncogeneHighImpact', 'TumorSuppressorHighImpact']:
        mut[col] = mut[col].astype(str).str.upper().map({'TRUE': True, 'FALSE': False})

    # 只保留至少一个高置信度突变标志为 TRUE 的记录
    mut = mut[
        (mut['Hotspot'] == True) |
        (mut['OncogeneHighImpact'] == True) |
        (mut['TumorSuppressorHighImpact'] == True)
    ]

    # 建立二值矩阵
    binary_mut = (
        mut[['ModelID', 'HugoSymbol']]
        .assign(value=1)
        .drop_duplicates()  # 避免重复 ModelID-基因 对
        .pivot(index='ModelID', columns='HugoSymbol', values='value')
        .fillna(0)
        .astype(int)
    )

    # 保存
    if save_path:
        binary_mut.to_csv(save_path, index=True, header=True)

    print("生成的二值化矩阵:", binary_mut.shape)


def filter_mut_binary_by_cgc(mutation_file, cgc_file, output_file=None):

    # 读取二值化 mutation 矩阵
    mut = pd.read_csv(mutation_file, index_col=0)

    # 读取 CGC
    cgc = pd.read_csv(cgc_file, sep="\t")

    # 取交集，避免 mutation 里不存在的基因报错
    cgc_genes = cgc['GENE_SYMBOL'].unique()
    selected_genes = [g for g in cgc_genes if g in mut.columns]
    mut_filtered = mut[selected_genes]

    # 保存结果
    if output_file:
        mut_filtered.to_csv(output_file)

    print(f"原始基因数: {mut.shape[1]}, CGC基因数: {len(cgc_genes)}, 交集: {len(selected_genes)}")


def count_cosmic_flags(path, flags=None):
    """
    :return: {'Hotspot': 4397, 'OncogeneHighImpact': 5391, 'TumorSuppressorHighImpact': 10208}
    """
    df = pd.read_csv(path)
    if flags is None:
        flags = ['Hotspot', 'OncogeneHighImpact', 'TumorSuppressorHighImpact']

    result = {}
    for flag in flags:
        if flag in df.columns:
            result[flag] = (df[flag] == True).sum()
        else:
            result[flag] = None  # 列不存在
    return result


def process_methylation(input_file, model_file, output_file):
    # 读取甲基化矩阵
    df = pd.read_csv(input_file, sep="\t")
    df = df.replace(r'^\s*NA\s*$', 0, regex=True)
    gene_col = df.iloc[:,1]  # 一般是 gene 或 TSS 标识
    df = df.set_index(gene_col)

    feature_cols = ["TSS_id", "gene", "chr", "fpos", "tpos", "strand", "avg_coverage"]
    df = df.drop(columns=feature_cols)
    # 转置，得到列 = CCLE cell line 名称
    df_t = df.T

    # 读取映射表 (model.csv)
    model_df = pd.read_csv(model_file)
    # 注意 CCLE 数据里的列名可能是 StrippedCellLineName 或 CellLineName
    mapping = dict(zip(model_df["CCLEName"], model_df["ModelID"]))

    # 映射成 ACH-xxx
    df_t.index = df_t.index.map(mapping)

    # 去掉没有映射上的细胞系
    df_t = df_t.dropna(axis=0, how="any")
    df_t.index.name = "DepMap_ID"

    # 保存
    df_t.to_csv(output_file, index=True, header=True)
    print(f"✅ 处理完成", df_t.shape)


def extract_cosmic_methylation_with_synonyms(methy_file, cosmic_file, output_file):
    # 读取 methylation 矩阵
    df = pd.read_csv(methy_file, index_col=0)

    # 读取 COSMIC 基因列表
    cosmic_df = pd.read_csv(cosmic_file, sep="\t")

    # 所有 COSMIC 名字集合，包括 GENE_SYMBOL 和 SYNONYMS
    cosmic_genes = set(cosmic_df['GENE_SYMBOL'])
    for syn_list in cosmic_df['SYNONYMS'].dropna():
        # SYNONYMS 可能是逗号分隔的字符串
        for name in syn_list.split(','):
            cosmic_genes.add(name.strip())

    # 筛选列
    cosmic_cols = [col for col in df.columns if col in cosmic_genes]
    df_cosmic = df[cosmic_cols]

    # NA 填 0
    df_cosmic = df_cosmic.fillna(0)

    # 保存
    df_cosmic.to_csv(output_file)
    print(f"✅ 已筛选 {len(cosmic_cols)} 个 COSMIC 基因及其别名，输出保存到 {output_file}")
    print(df_cosmic.shape)

def process_gdsc_ic50(raw_file, model_file, output_file):
     # 读取原始 GDSC 数据
    df = pd.read_csv(raw_file)
    df = df[["Drug ID", "Cell Line Name", "IC50"]]

    # 读取 model.csv
    model_df = pd.read_csv(model_file)

    # 建立 cell line name -> ACH ID 的映射
    cell_name2ach = dict(zip(model_df["CellLineName"], model_df["ModelID"]))

    # 映射成 ACH-XXXX
    df["DepMap_ID"] = df["Cell Line Name"].map(cell_name2ach)

    # 去掉没有匹配的
    df = df.dropna(subset=["DepMap_ID"])

    # 转换成 Drug × CellLine 矩阵
    df_pivot = df.pivot_table(
        index=df["Drug ID"].map(lambda x: f"GDSC:{x}"),
        columns="DepMap_ID",
        values="IC50",
        aggfunc="mean"  # 如果重复，取均值
    )

    # 缺失值填充 NA
    df_pivot = df_pivot.fillna("NA")

    # 统计
    n_drugs = df_pivot.shape[0]
    n_cells = df_pivot.shape[1]
    n_responses = (df_pivot != "NA").sum().sum()

    print(f"✅ 处理完成: {df_pivot.shape}, 保存到 {output_file}")
    print(f"  - 药物数: {n_drugs}")
    print(f"  - 细胞系数: {n_cells}")
    print(f"  - 总反应数: {n_responses}")

    # 保存
    df_pivot.to_csv(output_file)

    return df_pivot

def DrugidToSmiles(drug_info_file, drug_smiles_file, smile_inchi_file, output_pkl):
    # --- 来源1: drug_info + pubchem mapping ---
    reader = csv.reader(open(drug_info_file, 'r'))
    rows = [item for item in reader]

    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    pubchemid2smiles = {
        item.split('\t')[0]: item.split('\t')[1].strip()
        for item in open(drug_smiles_file).readlines()
    }

    drugid2smiles = {}
    for id in drugid2pubchemid.keys():
        if pubchemid2smiles.get(drugid2pubchemid[id]) is not None:
            drugid2smiles[id] = pubchemid2smiles[drugid2pubchemid[id]]

    # --- 来源2: smile_inchi.csv ---
    df = pd.read_csv(smile_inchi_file)
    df = df.dropna(subset=["smiles"])
    drugid2smiles2 = dict(zip(df["drug_id"].astype(str), df["smiles"]))

    # --- 来源3: 你的 csv 第一列和第三列 ---
    df_csv = pd.read_csv("/data0/liushujia/omics_new/drugid2smiles.csv", header=None)  # 这里替换成实际 csv 文件
    df_csv = df_csv.dropna(subset=[2])
    drugid2smiles3 = dict(zip(df_csv[0].astype(str), df_csv[2]))

    # --- 合并所有字典（后者覆盖前者） ---
    merged = {**drugid2smiles, **drugid2smiles2, **drugid2smiles3}

    # --- 保存为 pkl ---
    with open(output_pkl, "wb") as f:
        pickle.dump(merged, f)

    print(f"✅ 完成: {len(merged)} 个 drug_id -> SMILES 映射, 保存到 {output_pkl}")
    return merged



def smiles_dict_to_csv(id_smiles_dict, save_path):
    """
    将 {id: smiles} 字典保存为 CSV 文件
    """
    # 转换为 DataFrame
    df = pd.DataFrame(list(id_smiles_dict.items()), columns=["id", "smiles"])

    # 保存为 CSV，不带 index
    df.to_csv(save_path, index=False)

# 统计 GDSC2数据集中药物和cell line的数量，共 243467对反应
# r = count_drugs_and_cells("/data0/liushujia/csv/PANCANCER_IC_Sat Jun 21 05_40_24 2025.csv")
# r = {'drug_num': 297, 'cell_line_num': 969}

# gdcs 与 model 根据 cell line name进行数据对齐
#result = count_matching_cell_lines("/data0/liushujia/csv/PANCANCER_IC_Sat Jun 21 05_40_24 2025.csv", "/data0/liushujia/omics_new/Model.csv")
#result = {'direct_match_count': 825, 'stripped_match_count': 233, 'total_unique_matches': 864}

#从原始基因表达数据中筛选 Cosmic基因，并存入 csv文件
# filter_expression_by_cgc(  #原始基因数: 19205, CGC基因数: 758, 交集: 739
#     "/data0/liushujia/omics_new/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
#     "/data0/liushujia/omics_new/Cosmic_gene_expression.tsv",
#     "/data0/liushujia/omics_new/gene_expression_739dim.csv"
# )

# 将基因突变数据（格式复杂）转为二值化的 csv
# binarize_mutation(
#     "/data0/liushujia/omics_new/OmicsSomaticMutations.csv",
#     "/data0/liushujia/omics_new/mutation.csv"
# )

# 将基因突变数据（格式复杂）转为二值化的 csv,并根据HotSpot进行筛选
# binarize_mutation_on_hotspot(
#     "/data0/liushujia/omics_new/OmicsSomaticMutations.csv",
#     "/data0/liushujia/omics_new/mutation_hotspot_1076dim.csv"
# )

# 将基因突变数据（格式复杂）转为二值化的 csv,并根据Cosmic中的基因进行筛选，这样可以和 gene expression相对应
# filter_mut_binary_by_cgc(
#     "/data0/liushujia/omics_new/mutation_20113dim.csv",
#     "/data0/liushujia/omics_new/Cosmic_gene_expression.tsv",
#     "/data0/liushujia/omics_new/mutation_cosmic.csv"
# )

#处理甲基化数据
# process_methylation(
#     "/data0/liushujia/omics_new/CCLE_RRBS_TSS_1kb_20180614.txt",
#     "/data0/liushujia/omics_new/Model.csv",
#     "/data0/liushujia/omics_new/DNA_methylation_20192dim.csv"
# )

# extract_cosmic_methylation_with_synonyms(
#     "/data0/liushujia/omics_new/DNA_methylation_20192dim.csv",
#     "/data0/liushujia/omics_new/Cosmic_gene_expression.tsv",
#     "/data0/liushujia/omics_new/DNA_methylation_comics.csv"
# )


# #药物数: 403, 细胞系数: 805, 总反应数: 276397
# process_gdsc_ic50(
#     "/data0/liushujia/csv/PANCANCER_IC_Sat Jun 21 05_40_24 2025.csv",
#     "/data0/liushujia/omics_new/Model.csv",
#     "/data0/liushujia/omics_new/GDSC_IC50.csv"
# )

#药物数: 297, 细胞系数: 804, 总反应数: 201359
# process_gdsc_ic50(
#     "/data0/liushujia/omics_new/PANCANCER_GDSC1.csv",
#     "/data0/liushujia/omics_new/Model.csv",
#     "/data0/liushujia/omics_new/GDSC1_IC50.csv"
# )

# DrugidToSmiles(
#     "/home/liushujia/Graphormer-master/data_process/data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv",
#     "/home/liushujia/Graphormer-master/data_process/data/223drugs_pubchem_smiles.txt",
#     "/data0/liushujia/omics_new/smile_inchi.csv",
#     "/data0/liushujia/omics_new/drugid2smiles.pkl"
# )

# with open("/data0/liushujia/omics_new/drugid2smiles.pkl", "rb") as f:
#     id2smiles = pickle.load(f)
# smiles_dict_to_csv(
#     id2smiles,
#     "/data0/liushujia/molmvc/drugid2smiles.csv"
# )


# df1 = pd.read_csv("/data0/liushujia/omics_new/PANCANCER_GDSC1.csv")
# df2 = pd.read_csv("/data0/liushujia/csv/PANCANCER_IC_Sat Jun 21 05_40_24 2025.csv")
# mapping1 = df1[['Drug ID', 'Drug Name']].drop_duplicates()
# mapping2 = df2[['Drug ID', 'Drug Name']].drop_duplicates()
# dict1 = dict(zip(mapping1['Drug ID'], mapping1['Drug Name']))
# dict2 = dict(zip(mapping2['Drug ID'], mapping2['Drug Name']))
# for drug_id, drug_name in dict2.items():
#     try:
#         n2 = dict1[drug_id]
#         if n2 != drug_name:
#             print(f"冲突：{drug_id}  {drug_name}  {n2}")
#         else:
#             print("匹配")
#     except KeyError:
#         print(f"警告：Drug ID {drug_id} 只存在于文件1中")
# print("*")

# df1 = pd.read_csv("/data0/liushujia/omics_new/GDSC1_IC50.csv", index_col=0, na_values="NA")
# df2 = pd.read_csv("/data0/liushujia/omics_new/GDSC2_IC50.csv", index_col=0, na_values="NA")
# # 对齐行和列，只保留两边都存在的部分
# common_rows = df1.index.intersection(df2.index)
# common_cols = df1.columns.intersection(df2.columns)
#
# sub1 = df1.loc[common_rows, common_cols]
# sub2 = df2.loc[common_rows, common_cols]
#
# mask = (~sub1.isna()) & (~sub2.isna()) & (sub1 != sub2)
#
# conflicts = mask.stack()[mask.stack()]
# print(f"冲突总数: {len(conflicts)}")

# with open("/data0/liushujia/omics_new/drugid2smiles.pkl", "rb") as f:  # 注意这里是 "rb"
#     drugid2smiles = pickle.load(f)
#
# df = pd.read_csv("/data0/liushujia/omics_new/drugid2smiles.csv", header=None, names=['GDSC_ID', 'Empty', 'SMILES'])
# gdsc_to_smiles = dict(zip(df['GDSC_ID'], df['SMILES']))
# for drug_id, drug_name in gdsc_to_smiles.items():
#     drugid2smiles[drug_id] = drug_name
# with open("/data0/liushujia/omics_new/drugid2smiles.pkl", "wb") as f:  # 注意是"wb"模式
#     pickle.dump(drugid2smiles, f)

# with open("/data0/liushujia/omics_new/drugid2smiles.pkl", "rb") as f:  # 注意这里是 "rb"
#     drugid2smiles = pickle.load(f)
# with open("/data0/liushujia/molmvc/drugid2smiles.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     # 不写 header
#     for drug_id, smiles in drugid2smiles.items():
#         writer.writerow([drug_id, smiles])

# sample_ratio=0.1
# with open("/data0/liushujia/pretrain/pubchem-10m.txt", 'r') as f:
#     lines = f.readlines()
#
# total_lines = len(lines)
# sample_size = int(total_lines * sample_ratio)
# sampled_lines = random.sample(lines, sample_size)
#
# # 写入新文件
# with open("/data0/liushujia/pretrain/pubchem-1m.txt", 'w') as f:
#     f.writelines(sampled_lines)
#
# print(f"从 {total_lines} 行中随机抽取了 {sample_size} 行")

# def get_high_missing_smiles(csv_file, out_pkl="/data0/liushujia/omics_new/high_missing_smiles.pkl", threshold=0.4):
#     # 读取 CSV
#     df = pd.read_csv(csv_file, index_col=0, na_values="NA")
#
#     # 计算每行缺失比例
#     missing_ratio = df.isna().mean(axis=1)  # 每个药物的缺失比例
#
#     high_missing_smiles = missing_ratio[missing_ratio > threshold].index.tolist()
#     print(f"共有 {len(high_missing_smiles)} 个药物缺失比例大于 {threshold}")
#
#     aug_smiles_set = set(high_missing_smiles)
#     # 保存为 pkl
#     with open(out_pkl, "wb") as f:
#         pickle.dump(aug_smiles_set, f)
#
#     return aug_smiles_set

# 调用示例
#high_missing_smiles = get_high_missing_smiles("/data0/liushujia/omics_new/nest_gdsc2.csv")
