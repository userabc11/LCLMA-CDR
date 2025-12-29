import argparse
from texttable import Texttable
import os

def parse_args():
    parser = argparse.ArgumentParser()  # 参数解析器对象

    parser.add_argument('--seed', type=int, default=26, help='Random seed of the experiment')
    parser.add_argument('--exp_name', type=str, default="sanger_response_v2", help='Name of the experiment')
    parser.add_argument('--train_batch_size', type=int, default=512, help='Size of the training batch')
    parser.add_argument('--test_batch_size', type=int, default=512, help='Size of the testing batch')
    parser.add_argument('--gpu_index', type=int, default=2, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument('--finetune_layers', type=int, default=1, help='Finetune layers num')
    parser.add_argument('--epochs', type=int, default=250, help='Maximum number of epochs')
    args = parser.parse_args()  # 解析命令行参数

    return args

class IOStream():
    """训练日志文件"""
    def __init__(self, path):
        self.file = open(path, 'a') # 附加模式：用于在文件末尾添加内容，如果文件不存在则创建新文件

    def cprint(self, text):
        print(text)
        self.file.write(text + '\n')
        self.file.flush() # 确保将写入的内容刷新到文件中，以防止数据在缓冲中滞留

    def close(self):
        self.file.close()


def table_printer(args):
    """绘制参数表格"""
    args = vars(args) # 转成字典类型
    keys = sorted(args.keys()) # 按照字母顺序进行排序
    table = Texttable()
    table.set_cols_dtype(['t', 't']) # 列的类型都为文本(str)
    rows = [["Parameter", "Value"]] # 设置表头
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])]) # 下划线替换成空格，首字母大写
    table.add_rows(rows)
    return table.draw()

