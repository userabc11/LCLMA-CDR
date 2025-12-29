import numpy as np
from rdkit import Chem
from tqdm import tqdm
from collections import defaultdict
import os, torch, random, copy, csv
import random

def calculate_pcc(x, y):
    # Pearson correlation coefficient (PCC)
    corr, _ = pearsonr(x, y)
    return corr

def calculate_scc(x, y):
    # Spearman rank correlation coefficient (SCC)
    corr, _ = spearmanr(x, y)
    return corr

def calculate_rmse(y_true, y_pred):
    # 计算均方根误差 RMSE
    mse = np.mean((y_true - y_pred) ** 2)  # 计算均方误差 MSE
    rmse = np.sqrt(mse)  # 计算 RMSE
    return rmse

def get_valuation(y_true,y_pred):
    pcc = calculate_pcc(y_true,y_pred)
    scc = calculate_scc(y_true,y_pred)
    rmse = calculate_rmse(y_true,y_pred)
    return pcc,scc,rmse

def augment_smiles_rdkit(smiles_list):
    aug_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # 跳过无效的 SMILES

        max_attempts = 5
        attempts = 0
        found = False

        while attempts < max_attempts:
            rand_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if rand_smiles != smiles:  # 排除原 SMILES
                aug_list.append(rand_smiles)
                found = True
                break
            attempts += 1

        # 如果尝试 max_attempts 次后仍未找到不同的 SMILES，添加原始 SMILES
        if not found:
            aug_list.append(smiles)

    return aug_list


def augment_smiles(smiles_list, n_aug=1, mask_prob=0.1, delete_prob=0.1, mode="fusion"):
    """
    对 SMILES 列表做噪声增强（mask/delete/fusion）。
    """
    aug_list = []

    for smiles in smiles_list:
        for _ in range(n_aug):
            s = list(smiles)
            if mode == "mask":
                for i in range(len(s)):
                    if random.random() < mask_prob:
                        s[i] = "[MASK]"
                aug_list.append("".join(s))
            elif mode == "delete":
                s = [c for c in s if random.random() > delete_prob]
                aug_list.append("".join(s))
            elif mode == "fusion":
                choice = random.choice(["mask", "delete"])
                # 递归调用单个 SMILES 增强
                s_aug = augment_smiles([smiles], n_aug=1, mode=choice, mask_prob=mask_prob, delete_prob=delete_prob)[0]
                aug_list.append(s_aug)

    return aug_list

def soft_cross_entropy(pred, soft_targets):
    log_prob = torch.log_softmax(pred, dim=1)
    loss = -(soft_targets * log_prob).sum(dim=1).mean()
    return loss

def train(train_loader, model, optimizer, criterion, scheduler, device, epoch, max_epoch):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader),
                         desc="Train_Loader", leave=False, mininterval=60):

        smiles = batch["smiles"]
        morgan = batch["morgan"].to(device)
        rdkit = batch["rdkit"].to(device)
        gexpr = batch["gexpr"].to(device)
        methylation = batch["methylation"].to(device)
        mutation = batch["mutation"].to(device)
        miRNA = batch["miRNA"].to(device)
        copynumber = batch["copynumber"].to(device)
        y = batch["y"].to(device)

        # ----------------------------- # Mixup # ----------------------------------
        alpha = 0.5
        lam = np.random.beta(alpha, alpha)

        batch_size = y.size(0)
        drug_id_list = batch["drug_id"]  # 这里是 list[str]，不用转 tensor
        index = list(range(batch_size))  # 默认自己和自己 mix

        # 为每个 drug_id 建立索引表
        drug_to_indices = {}
        for i, d in enumerate(drug_id_list):
            drug_to_indices.setdefault(d, []).append(i)

        # 遍历 batch，给每个样本找同 drug 的 partner
        for b in range(batch_size):
            candidates = drug_to_indices[drug_id_list[b]]
            if len(candidates) > 1:
                j = random.choice(candidates)
                index[b] = j
            else:
                index[b] = b  # 没有 partner，不做 mix

        optimizer.zero_grad()
        index = torch.tensor(index, device=device)
        gexpr_mixed = lam * gexpr + (1 - lam) * gexpr[index]
        methylation_mixed = lam * methylation + (1 - lam) * methylation[index]
        mutation_mixed = lam * mutation + (1 - lam) * mutation[index]
        miRNA_mixed = lam * miRNA + (1 - lam) * miRNA[index]
        copynumber_mixed = lam * copynumber + (1 - lam) * copynumber[index]
        y_mix = lam * y + (1 - lam) * y[index]
        # 增强 smiles
        if(i % 2 == 0):
            smiles = augment_smiles(smiles_list=smiles)
        outputs = model({
            "smiles": smiles,
            "morgan": morgan,
            "rdkit": rdkit,
            "gexpr": gexpr_mixed,
            "methylation": methylation_mixed,
            "mutation": mutation_mixed,
            "miRNA": miRNA_mixed,
            "copynumber": copynumber_mixed
        })
        loss = criterion(outputs, y_mix)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy().flatten())
        #all_labels.append(y_mix.detach().cpu().numpy().flatten())
        all_labels.append(y.detach().cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pcc, scc, rmse = get_valuation(all_preds, all_labels)

    return train_loss / len(train_loader.dataset), pcc, scc, rmse


def test(test_loader, model, criterion, device, save_path=None):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test_Loader", leave=False,
                         mininterval=60 * 2):
        with torch.no_grad():
            smiles = batch["smiles"]  # list[str]
            morgan = batch["morgan"].to(device)
            rdkit = batch["rdkit"].to(device)
            gexpr = batch["gexpr"].to(device)  # tensor [B, gexpr_dim]
            methylation = batch["methylation"].to(device)  # tensor [B, methyl_dim]
            mutation = batch["mutation"].to(device)  # tensor [B, methyl_dim]
            miRNA = batch["miRNA"].to(device)
            copynumber = batch["copynumber"].to(device)
            y = batch["y"].to(device)  # tensor [B, 1]

            outputs = model({
                "smiles": smiles,
                "morgan": morgan,
                "rdkit": rdkit,
                "gexpr": gexpr,
                "methylation": methylation,
                "mutation": mutation,
                "miRNA": miRNA,
                "copynumber": copynumber
            })
            loss = criterion(outputs, y)
            test_loss += loss.item()

        all_preds.append(outputs.detach().cpu().numpy().flatten())
        all_labels.append(y.detach().cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pcc, scc, rmse = get_valuation(all_preds, all_labels)
    return test_loss / len(test_loader.dataset), pcc, scc, rmse


def test_and_write(test_loader, model, criterion, device, args, fold, topic):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    output_path = f"./{topic}/{args.exp_name}/all_results_{fold}.csv"

    # 打开 CSV 文件，写入表头
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["DrugID", "CellID", "pred", "label"])

        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test_Loader", leave=False,
                            mininterval=60 * 2):
            with torch.no_grad():
                smiles = batch["smiles"]  # list[str]
                morgan = batch["morgan"].to(device)
                rdkit = batch["rdkit"].to(device)
                gexpr = batch["gexpr"].to(device)  # tensor [B, gexpr_dim]
                methylation = batch["methylation"].to(device)  # tensor [B, methyl_dim]
                mutation = batch["mutation"].to(device)  # tensor [B, methyl_dim]
                miRNA = batch["miRNA"].to(device)
                copynumber = batch["copynumber"].to(device)
                y = batch["y"].to(device)  # tensor [B, 1]
                cellId_list = batch["cell_id"]
                drugId_list = batch["drug_id"]

                outputs = model({
                    "smiles": smiles,
                    "morgan": morgan,
                    "rdkit": rdkit,
                    "gexpr": gexpr,
                    "methylation": methylation,
                    "mutation": mutation,
                    "miRNA": miRNA,
                    "copynumber": copynumber
                })
                loss = criterion(outputs, y)
                test_loss += loss.item()

                preds = outputs.detach().cpu().numpy().flatten()
                labels = y.detach().cpu().numpy().flatten()

                # 写入每条样本
                for d_id, c_id, p, l in zip(drugId_list, cellId_list, preds, labels):
                    writer.writerow([d_id, c_id, p, l])

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pcc, scc, rmse = get_valuation(all_preds, all_labels)

    return test_loss / len(test_loader.dataset), pcc, scc, rmse
