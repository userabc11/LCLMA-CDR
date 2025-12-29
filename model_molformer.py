import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionFusion(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, features):  # [B, M, D]
        fused, _ = self.attn(features, features, features)  # [B, M, D]
        fused = fused.mean(dim=1)  # 或者加权池化
        return fused


class CrossAttention(nn.Module):
    def __init__(self, dim = 384, hid_dim = 384):
        super().__init__()
        self.attn = nn.MultiheadAttention(hid_dim, num_heads=4, batch_first=True)
        self.attn1 = nn.MultiheadAttention(hid_dim, num_heads=4, batch_first=True)
        self.proj_drug = nn.Linear(dim, hid_dim)
        self.proj_omics = nn.Linear(dim, hid_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, drug, omics):
        drug_proj = self.proj_drug(drug).unsqueeze(1)
        omics_proj = self.proj_omics(omics).unsqueeze(1)

        # 让 drug query omics key/value
        attn_output, _ = self.attn(query=drug_proj, key=omics_proj, value=omics_proj)
        attn_output1, _ = self.attn1(query=omics_proj, key=drug_proj, value=drug_proj)

        attn_output = self.dropout(attn_output.squeeze(1))  # (batch, hid_dim)
        attn_output1 = self.dropout(attn_output1.squeeze(1))  # (batch, hid_dim)

        return torch.cat([drug + attn_output, omics + attn_output1], dim=1)


class SelfAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.r = 8
        self.gamma = 0.5
        self.clf = nn.Sequential(
            nn.Linear(dim, dim // self.r),
            nn.ReLU(),
            nn.Linear(dim // self.r, dim)
        )
        self.clf.apply(self.xavier_init)

    def forward(self, x):
        att = self.clf(x)
        att = torch.sigmoid(att)
        return self.gamma * x * att + x

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):  # features: [B, M, D], M=模态数
        scores = self.attn(features)  # [B, M, 1]
        weights = self.softmax(scores)  # [B, M, 1]
        fused = torch.sum(weights * features, dim=1)  # [B, D]
        return fused, weights



class OmicsEncoderLight(nn.Module):
    def __init__(self, gexpr_dim=739, methy_dim=627, mut_dim=1076, copynuber_dim=512, rna_dim=735 ,
                 n_filters=4, hid_dim=512 ,output_dim=384):
        super().__init__()
        # miRNA
        self.miRNA_encoder1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8),
            nn.BatchNorm1d(n_filters * 4),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )
        self.miRNA_encoder2 = nn.Sequential(
            nn.Linear(368, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # copynumber
        self.copynumber_encoder = nn.Sequential(
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.gexpr_encoder = nn.Sequential(
            nn.Linear(gexpr_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.gate = GatedAttention(output_dim)

    def forward(self, gexpr, methy, mut, miRNA, copynumber):
        miRNA = miRNA[:, None, :]
        miRNA = self.miRNA_encoder1(miRNA)
        miRNA = miRNA.view(-1, miRNA.shape[1] * miRNA.shape[2])
        miRNA = self.miRNA_encoder2(miRNA)

        copynumber = self.copynumber_encoder(copynumber)
        gexpr = self.gexpr_encoder(gexpr)

        features = torch.stack([miRNA, copynumber, gexpr], dim=1)
        fused, _ = self.gate(features)
        fused = copynumber
        return fused


class CLSAttentionFusion(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features):  # [B, M, D]
        B = features.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, features], dim=1)   # [B, 1+M, D]

        out, _ = self.attn(x, x, x)
        out = self.norm(out)

        return out[:, 0]  # CLS 向量


class HierarchicalAttentionFusion(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.intra_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features):  
        """
        features: [B, M, T, D]
        M = 模态数, T = 每个模态的 token 数
        """
        B, M, T, D = features.shape
        x = features.view(B * M, T, D)

        intra, _ = self.intra_attn(x, x, x)
        intra = intra.mean(dim=1)            # [B*M, D]
        intra = intra.view(B, M, D)          

        inter, _ = self.inter_attn(intra, intra, intra)
        fused = inter.mean(dim=1)

        return self.norm(fused)


class MolFormerCDR(nn.Module):
    def __init__(self, args, do_down_stream=False):
        super().__init__()
        # 加载 MoLFormer 模型 & tokenizer
        molformer_dir = "/data0/liushujia/MoLFormer/"
        self.finetune_layers = args.finetune_layers
        self.Tokenizer = AutoTokenizer.from_pretrained(molformer_dir, max_length=512, trust_remote_code=True)
        self.MolFormer = AutoModel.from_pretrained(molformer_dir, trust_remote_code=True)
        self.do_down_stream = do_down_stream

        # MoLFormer 的 hidden size = 768
        self.L1 = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256)
        )

        self.morgan_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128)
        )

        self.rdkit_encoder = nn.Sequential(
            nn.Linear(881, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128)
        )

        self.OmicsEncoder = OmicsEncoderLight()
        state_dict = torch.load("./transfer_pretrain.pth")
        self.OmicsEncoder.load_state_dict(state_dict)

        self.Predictor = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self.att = SelfAtt(768)

        self.set_trainable_params()

    def set_trainable_params(self):
        # 先冻结 MoLFormer 所有层
        for param in self.MolFormer.parameters():
            param.requires_grad = False

        # 解冻最后 finetune_layersfinetune_layers 层 Transformer block
        if self.finetune_layers > 0:
            for layer in self.MolFormer.encoder.layer[-self.finetune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # 解冻 L1、OmicsEncoder、Predictor
        for param in self.L1.parameters():
            param.requires_grad = True
        for param in self.OmicsEncoder.parameters():
            param.requires_grad = True
        for param in self.Predictor.parameters():
            param.requires_grad = True
        for param in self.morgan_encoder.parameters():
            param.requires_grad = True
        for param in self.rdkit_encoder.parameters():
            param.requires_grad = True
        for param in self.att.parameters():
            param.requires_grad = True

    def get_param_group(self):
        param_groups = []
        lr = 1e-4
        # 仅当 finetune_layers > 0 时才添加 MoLFormer 参数组
        if self.finetune_layers > 0:
            param_groups.append({
                "params": self.MolFormer.encoder.layer[-self.finetune_layers:].parameters(),
                "lr": 1e-4,
                "name": "Molformer"
            })

        param_groups += [
            {"params": self.L1.parameters(), "lr": lr, "name": "L1"},
            {"params": self.OmicsEncoder.parameters(), "lr": lr, "name": "OmicsEncoder"},
            {"params": self.Predictor.parameters(), "lr": lr, "name": "Predictor"},
            {"params": self.morgan_encoder.parameters(), "lr": lr, "name": "morgan_encoder"},
            {"params": self.rdkit_encoder.parameters(), "lr": lr, "name": "rdkit_encoder"},
            {"params": self.att.parameters(), "lr": lr, "name": "att"}
        ]

        return param_groups


    def forward(self, data):
        smiles_token = self.Tokenizer(
            data["smiles"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.MolFormer.device)

        outputs = self.MolFormer(**smiles_token)

        # CLS 向量
        cls_embed = outputs.last_hidden_state[:, 0, :]
        drug_embedding = self.L1(cls_embed)

        morgan = self.morgan_encoder(data["morgan"])
        rdkit = self.rdkit_encoder(data["rdkit"])
        fp_embedding = morgan + rdkit
        #fp_embedding = self.fp_encoder(F.normalize(data["fp"], dim=-1))

        # omics
        cell_embedding = self.OmicsEncoder(data["gexpr"], data["methylation"], data["mutation"], data["miRNA"], data["copynumber"])

        fused = torch.cat([drug_embedding, fp_embedding, cell_embedding], dim=1)
        fused = self.att(fused)
        output = self.Predictor(fused)
        return output

        # if self.train:
        #     fused, y_a, y_b, lam = self.manifold_mixup(fused, data["label"])
        #     output = self.Predictor(fused)
        #     return output, y_a, y_b, lam
        # else:
        #     output = self.Predictor(fused)
        #     return output


        # if self.train:
        #     fused_mix, y_mix = self.manifold_mixup(fused, data["label"])
        #     output = self.Predictor(fused_mix)
        #     return output, y_mix
        # else:
        #     output = self.Predictor(fused)
        #     return output

        # output = self.Predictor(self.att(torch.cat([drug_embedding, fp_embedding], dim=1), cell_embedding))
        # return output

    def manifold_mixup(self, fused, y, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        batch_size = fused.size(0)
        index = torch.randperm(batch_size).to(fused.device)
        fused_mix = lam * fused + (1 - lam) * fused[index]
        y_mix = lam * y + (1 - lam) * y[index]
        return fused_mix, y_mix

    def forward_drug_only(self, smiles_list):
        smiles_token = self.Tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.MolFormer.device)

        outputs = self.MolFormer(**smiles_token, output_attentions=True)
        return outputs.last_hidden_state[:, 0, :], outputs.attentions, smiles_token

    def get_drug_embedding(self, smiles_list):
        smiles_token = self.Tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.MolFormer.device)

        outputs = self.MolFormer(**smiles_token, output_attentions=True)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        drug_embedding = self.L1(cls_embed)

        return drug_embedding

    def forward_with_mid(self, data):
        smiles_token = self.Tokenizer(
            data["smiles"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.MolFormer.device)
        morgan = self.morgan_encoder(data["morgan"])
        rdkit = self.rdkit_encoder(data["rdkit"])

        outputs = self.MolFormer(**smiles_token)

        # CLS 向量
        cls_embed = outputs.last_hidden_state[:, 0, :]
        x0 = torch.cat([cls_embed, morgan, rdkit, data["gexpr"], data["miRNA"], data["copynumber"]], dim=1)
        drug_embedding = self.L1(cls_embed)


        fp_embedding = morgan + rdkit

        # omics
        cell_embedding = self.OmicsEncoder(data["gexpr"], data["methylation"], data["mutation"], data["miRNA"], data["copynumber"])

        fused = torch.cat([drug_embedding, fp_embedding, cell_embedding], dim=1)
        x1 = fused
        fused = self.att(fused)
        x2 = self.Predictor[:-1](fused)
        output = self.Predictor(fused)
        return output, x0, x1, x2

