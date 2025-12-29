from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import pickle
from types import SimpleNamespace
from torch.utils.data import DataLoader

from data_process.dataSet import *
from data_process.loadData00 import *

def loadData(args, config, ratio=1.0):
    print("start loading data from ./data ...")
    data = loadDataFromFiles(config)
    gexpr,methylation,mutation,miRNA, response, drugid2smiles= \
            data["Gene_expression_filter"], \
            data["Methylation_filter"], \
            data["Gene_mutation_filter"], \
            data["MiRNA"], \
            data["Responds"], \
            data["Drugid2smiles"]

    n = len(response)
    indices = list(range(n))

    # use numpy random
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    train_end = int(n * 0.8)

    val_end = train_end + int(n * 0.1)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # 按 ratio 截取训练集
    if ratio < 1.0:
        keep_n = int(len(train_idx) * ratio)
        train_idx = train_idx[:keep_n]

    train = [response[i] for i in train_idx]
    val = [response[i] for i in val_idx]
    test = [response[i] for i in test_idx]

    # dataset
    train_dataset = MyDataset(gexpr, methylation, mutation, miRNA, train, drugid2smiles, do_aug=False)
    val_dataset = MyDataset(gexpr, methylation, mutation, miRNA, val, drugid2smiles, do_aug=False)
    test_dataset = MyDataset(gexpr, methylation, mutation, miRNA, test, drugid2smiles, do_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


def loadNaNData(args, config):
    print("start loading NAN response data from ./data ...")
    data = loadDataFromFiles(config)
    gexpr,methylation,mutation,miRNA, response, drugid2smiles= \
            data["Gene_expression_filter"], \
            data["Methylation_filter"], \
            data["Gene_mutation_filter"], \
            data["MiRNA"], \
            data["Responds"], \
            data["Drugid2smiles"]

    dataset = MyDataset(gexpr, methylation, mutation, miRNA, response, drugid2smiles, do_aug=False)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader


def getDrugBldFoldLoader(args, fold):
    print(f"开始加载第 {fold}折药物盲测数据...")
    assert fold in [1, 2, 3, 4, 5]
    path = f"/data0/liushujia/gdsc1-drugbld/results_fold{fold}.pkl"
    #path = f"/data0/liushujia/gdsc2-drugbld/results_fold{fold}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    gexpr, methylation, mutation, miRNA, drugid2smiles, train, val, test, = \
            data["Gene_expression_filter"], \
            data["Methylation_filter"], \
            data["Gene_mutation_filter"], \
            data["MiRNA"], \
            data["Drugid2smiles"], \
            data["Train"], \
            data["Val"], \
            data["Test"]

    train_dataset = MyDataset(gexpr, methylation, mutation, miRNA, train, drugid2smiles, do_aug=False)
    val_dataset = MyDataset(gexpr, methylation, mutation, miRNA, val, drugid2smiles, do_aug=False)
    test_dataset = MyDataset(gexpr, methylation, mutation, miRNA, test, drugid2smiles, do_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


def getCellBldFoldLoader(args, fold):
    print(f"开始加载第 {fold}折细胞系盲测数据...")
    assert fold in [1, 2, 3, 4, 5]
    path = f"/data0/liushujia/gdsc1-cellbld/results_fold{fold}.pkl"
    #path = f"/data0/liushujia/gdsc2-cellbld/results_fold{fold}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    gexpr, methylation, mutation, miRNA, drugid2smiles, train, val, test, = \
            data["Gene_expression_filter"], \
            data["Methylation_filter"], \
            data["Gene_mutation_filter"], \
            data["MiRNA"], \
            data["Drugid2smiles"], \
            data["Train"], \
            data["Val"], \
            data["Test"]

    train_dataset = MyDataset(gexpr, methylation, mutation, miRNA, train, drugid2smiles, do_aug=False)
    val_dataset = MyDataset(gexpr, methylation, mutation, miRNA, val, drugid2smiles, do_aug=False)
    test_dataset = MyDataset(gexpr, methylation, mutation, miRNA, test, drugid2smiles, do_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader



# args = {
#     "seed":42,
#     "train_batch_size": 512,
#     "test_batch_size": 512,
#     "finetune_layers": 3
# }
# LOAD_CONFIG = {
#     "Gene_expression_ori_file": False,
#     "Gene_expression_filter_file": True,
#     "Methylation_ori_file": False,
#     "Methylation_filter_file": True,
#     "Gene_mutation_ori_file": False,
#     "Gene_mutation_filter_file": True,
#     "MiRNA_file": True,
#     "Copy_number_file": False,
#     "GDSC1": False,
#     "GDSC2": True
# }
# args_obj = SimpleNamespace(**args)
# loader, _, _ = loadData(args_obj, LOAD_CONFIG)
# for i, batch in enumerate(loader):
#     l = len(batch["smiles"])
#     print(f"epoch {i}  {l}")