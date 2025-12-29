from data_process.dataLoader import loadData, getDrugBldFoldLoader, getCellBldFoldLoader
from tqdm import tqdm, trange
import torch.nn as nn
from model_molformer import MolFormerCDR
from drugbld_molformer import BLDMolFormer
from molformer_ablidation import AblationMolFormerCDR
from parameter import parse_args, IOStream, table_printer
from main_helper import *
import torch.nn.functional as F

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
    'NAN_GDSC2':False,
    "SANGER_RESP":False
}

LOAD_CONFIG_GDSC1 = {
    "Gene_expression_ori_file": False,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": False,
    "Methylation_filter_file": True,
    "Gene_mutation_ori_file": False,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": True,
    "Copy_number_file": False,
    "GDSC1": True,
    "GDSC2": False,
    'NAN_GDSC2':False,
    "SANGER_RESP":False
}

LOAD_CONFIG_SANGER = {
    "Gene_expression_ori_file": False,
    "Gene_expression_filter_file": True,
    "Methylation_ori_file": False,
    "Methylation_filter_file": True,
    "Gene_mutation_ori_file": False,
    "Gene_mutation_filter_file": True,
    "MiRNA_file": True,
    "Copy_number_file": False,
    "GDSC1": False,
    "GDSC2": False,
    'NAN_GDSC2':False,
    "SANGER_RESP":True
}

class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, true):
        alpha = 0.5
        rmse = torch.sqrt(self.mse(pred, true))
        mae = self.mae(pred, true)
        return alpha * rmse + (1 - alpha) * mae


def apply_mask(x, mask_ratio=0.15):
    if x.dim() != 2:
        raise ValueError("只支持 [batch, dim] 的输入")

    batch_size, dim = x.shape
    num_mask = int(dim * mask_ratio)

    rand = torch.rand(batch_size, dim, device=x.device)

    # 取每行的前 num_mask 个最小值作为 mask
    thresh = torch.topk(rand, num_mask, dim=1, largest=False).values.max(dim=1, keepdim=True).values
    mask = (rand <= thresh).float()  # [batch, dim] 0/1 mask

    return x * (1 - mask)


def exp_init(topic):
    """实验初始化"""
    if not os.path.exists(topic):
        os.mkdir(topic)
    if not os.path.exists(topic + '/' +args.exp_name):
        os.mkdir(topic + '/' + args.exp_name)

    # 对训练文件留 backup
    output_dir = f"{topic}/{args.exp_name}"
    os.system(f'cp main.py {output_dir}/main.py.backup')
    os.system(f'cp main_helper.py {output_dir}/main_helper.py.backup')
    os.system(f'cp model.py {output_dir}/model.py.backup')
    os.system(f'cp model_molformer.py {output_dir}/model_molformer.py.backup')
    os.system(f'cp parameter.py {output_dir}/parameter.py.backup')
    # /data_process
    os.system(f'cp data_process/dataSet.py {output_dir}/dataSet.py.backup')
    os.system(f'cp data_process/dataLoader.py {output_dir}/dataLoader.py.backup')
    os.system(f'cp data_process/loadData.py {output_dir}/loadData.py.backup')


def drugbld_exp_init():
    """实验初始化"""
    if not os.path.exists('drugbld'):
        os.mkdir('drugbld')
    if not os.path.exists('drugbld/' + args.exp_name):
        os.mkdir('drugbld/' + args.exp_name)

    # 对训练文件留 backup
    output_dir = f"drugbld/{args.exp_name}"
    os.system(f'cp main.py {output_dir}/main.py.backup')
    os.system(f'cp main_helper.py {output_dir}/main_helper.py.backup')
    os.system(f'cp model.py {output_dir}/model.py.backup')
    os.system(f'cp model_molformer.py {output_dir}/model_molformer.py.backup')
    os.system(f'cp parameter.py {output_dir}/parameter.py.backup')
    # /data_process
    os.system(f'cp data_process/dataSet.py {output_dir}/dataSet.py.backup')
    os.system(f'cp data_process/dataLoader.py {output_dir}/dataLoader.py.backup')
    os.system(f'cp data_process/loadData.py {output_dir}/loadData.py.backup')


def main(args, config, topic='outputs', ratio=1.0, start_val_epoch=100, is_ablation=False):
    model_path = "model.pth"
    TRAIN_FROM_CHEKPOINT = False
    best_model_dict = None
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    patience = 30
    counter = 0
    best_loss = 999

    IO = IOStream(f'{topic}/' + args.exp_name + '/run.log')
    IO.cprint("--------------------------------------------------------------")
    IO.cprint(str(table_printer(args)))  # 参数可视化
    """
        实验描述
    """
    device = f'cuda:{args.gpu_index}'
    print(f"Using GPU {args.gpu_index}")

    # data
    train_loader, val_loader, test_loader = loadData(args, config, ratio=ratio)
    IO.cprint(f"[INFO]Training set size: {len(train_loader.dataset)} samples")
    IO.cprint(f"[INFO]Validation set size: {len(val_loader.dataset)} samples")
    IO.cprint(f"[INFO]Test set size: {len(test_loader.dataset)} samples")
    IO.cprint(str(config))

    # model
    if is_ablation:
        model = AblationMolFormerCDR(args)
    else:
        model = MolFormerCDR(args)
    if TRAIN_FROM_CHEKPOINT:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        IO.cprint("train from check point")
    model = model.to(device)
    IO.cprint(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint('Model Parameter: {}'.format(total_params))

    # optimizer
    criterion = ComboLoss()
    param_groups = model.get_param_group()
    optimizer = torch.optim.AdamW(param_groups)
    for i, group in enumerate(optimizer.param_groups):
        IO.cprint(f"Param group {i}: lr = {group['lr']}, name = {group['name']}, num_params = {len(group['params'])}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    IO.cprint('Using AdamW...')

    losses, test_losses = [], []
    IO.cprint('Using MSELoss...')

    # train
    epochs = trange(args.epochs, leave=True, desc="Epochs")
    for epoch in epochs:
        train_loss, pcc, scc, rmse = train(train_loader, model, optimizer, criterion, scheduler, device, epoch, args.epochs)
        losses.append(train_loss)
        IO.cprint(
            'Epoch #{:03d},Train_Loss:{:.4f},pcc{:.3f},scc{:.3f},rmse{:.3f}'.format(epoch, train_loss, pcc, scc, rmse))

        if (epoch + 1) > start_val_epoch:
            # do val
            test_loss, pcc, scc, rmse = test(val_loader, model, criterion, device)
            test_losses.append(test_loss)
            IO.cprint(
                '[##VAL##]Epoch #{:03d},Test_Loss:{:.4f},pcc:{:.3f},scc:{:.3f},rmse:{:.3f}'.format(epoch, test_loss,
                                                                                                   pcc, scc, rmse))
            # 早停机制
            if test_loss < best_loss:
                best_model_dict = copy.deepcopy(model.state_dict())
                best_loss = test_loss
                counter = 0
            else:
                counter = counter + 1
            if counter == patience:
                IO.cprint("[info] **** early stop ****")
                break

    model.load_state_dict(best_model_dict)
    torch.save(model.state_dict(), f'{topic}/{args.exp_name}/model_{args.seed}.pth')
    IO.cprint(f'model saved in: {topic}/{args.exp_name}/model_{args.seed}.pth')

    IO.cprint("\n******* on test dataset *******")
    #predict_nan_response(test_loader, model, args, device)

    test_loss, pcc, scc, rmse = test(test_loader, model, criterion, device)
    IO.cprint(f"test pcc: {pcc}")
    IO.cprint(f"test scc: {scc}")
    IO.cprint(f"test rmse: {rmse}")

    test_and_write(test_loader, model, criterion, device, args, args.seed, topic)
    return 0, 0, 0


def main_fold(args, fold, task):
    model_path = "/home/liushujia/Graphormer-master/outputs/Prediction_final_nobn/model.pth"
    TRAIN_FROM_CHEKPOINT = False
    best_model_dict = None
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    patience = 10
    counter = 0
    best_loss = 999

    IO = IOStream(f'{task}/' + args.exp_name + '/run.log')
    IO.cprint("--------------------------------------------------------------")
    IO.cprint(str(table_printer(args)))  # 参数可视化

    device = f'cuda:{args.gpu_index}'
    print(f"Using GPU {args.gpu_index}")

    # data
    if (task == "drugbld"):
        train_loader, val_loader, test_loader = getDrugBldFoldLoader(args, fold)
        #model = BLDMolFormer(args)
        model = MolFormerCDR(args)
    elif (task == "cellbld"):
        train_loader, val_loader, test_loader = getCellBldFoldLoader(args, fold)
        model = MolFormerCDR(args)
    else:
        raise Exception("task mast be drugbld or cellbld")

    IO.cprint(f"[INFO]Training set size: {len(train_loader.dataset)} samples")
    IO.cprint(f"[INFO]Validation set size: {len(val_loader.dataset)} samples")
    IO.cprint(f"[INFO]Test set size: {len(test_loader.dataset)} samples")

    if TRAIN_FROM_CHEKPOINT:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        IO.cprint("train from check point")
    model = model.to(device)
    IO.cprint(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint('Model Parameter: {}'.format(total_params))

    # optimizer
    criterion = ComboLoss()
    param_groups = model.get_param_group()
    optimizer = torch.optim.AdamW(param_groups)
    for i, group in enumerate(optimizer.param_groups):
        IO.cprint(f"Param group {i}: lr = {group['lr']}, name = {group['name']}, num_params = {len(group['params'])}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    IO.cprint('Using AdamW...')

    # loss
    losses, test_losses = [], []
    IO.cprint('Using Comboloss...')

    # train
    epochs = trange(args.epochs, leave=True, desc="Epochs")
    for epoch in epochs:
        train_loss, pcc, scc, rmse = train(train_loader, model, optimizer, criterion, scheduler, device, epoch, args.epochs)
        losses.append(train_loss)
        IO.cprint(
            'Epoch #{:03d},Train_Loss:{:.4f},pcc{:.3f},scc{:.3f},rmse{:.3f}'.format(epoch, train_loss, pcc, scc, rmse))

        if (epoch + 1) % 3 == 0:
            # do val
            test_loss, pcc, scc, rmse = test(val_loader, model, criterion, device)
            test_losses.append(test_loss)
            IO.cprint(
                '[##VAL##]Epoch #{:03d},Test_Loss:{:.4f},pcc:{:.3f},scc:{:.3f},rmse:{:.3f}'.format(epoch, test_loss,
                                                                                                   pcc, scc, rmse))

            # do test
            test_loss, pcc, scc, rmse = test(test_loader, model, criterion, device)
            IO.cprint(
                '[##TEST##]Epoch #{:03d},Test_Loss:{:.4f},pcc:{:.3f},scc:{:.3f},rmse:{:.3f}'.format(epoch, test_loss,
                                                                                                   pcc, scc, rmse))
            # 早停机制
            if test_loss < best_loss:
                best_model_dict = copy.deepcopy(model.state_dict())
                best_loss = test_loss
                counter = 0
            else:
                counter = counter + 1
            if counter == patience and (epoch+1) > 50:
                IO.cprint("[info] **** early stop ****")
                break

    model.load_state_dict(best_model_dict)

    IO.cprint("\n******* on test dataset *******")
    test_loss, pcc, scc, rmse = test_and_write(test_loader, model, criterion, device, args, fold, task)
    IO.cprint(f"use test_and_write")
    IO.cprint(f"test pcc: {pcc}")
    IO.cprint(f"test scc: {scc}")
    IO.cprint(f"test rmse: {rmse}")

    return 0, 0, 0


if __name__ == '__main__':
    args = parse_args()
    topic = "mix_test"
    ratio = 1.0
    start_val_epoch = 100
    exp_init(topic)
    pcc, scc, rmse = main(args, LOAD_CONFIG_GDSC2, topic=topic, ratio=ratio, start_val_epoch=start_val_epoch)
        
    # drugbld test
    # args = parse_args()
    # args.epochs = 150
    # args.seed = 10
    # task = "drugbld"
    # drugbld_exp_init()
    # folds = [1,2,3,4,5]
    #
    # for f in folds:
    #     pcc, scc, rmse = main_fold(args, f, task)

    # cellbld` test
    # args = parse_args()
    # task = "cellbld"
    # args.epochs = 200
    # exp_init(task)
    # folds = [1, 2, 3, 4, 5]
    #
    # for f in folds:
    #     pcc, scc, rmse = main_fold(args, f, task)



