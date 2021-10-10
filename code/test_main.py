# 基本依赖包
import os
import sys
import time
import json
import traceback
import numpy as np
from glob import glob
from tqdm import tqdm
from tools import parse, py_op


# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


# 自定义文件
import loss
import models
import function
import loaddata
# import framework
from loaddata import dataloader
from models import lstm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

args = parse.args
args.split_nn = args.split_num + args.split_nor * 3
args.vocab_size = args.split_nn * 145 + 1
print('vocab_size', args.vocab_size)
args.task = "task2"

def main():
    patient_time_record_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_time_record_dict.json'))
    patient_master_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_master_dict.json'))
    patient_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_label_dict.json'))
    patient_test = list(json.load(open(os.path.join(args.file_dir, args.task, 'test.json'))))

    test_dataset = dataloader.DataSet(
                patient_test, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='test')
    test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)  

    for i, data in enumerate(tqdm(test_loader)):
        data = [ Variable(x.to(device)) for x in data ]
        visits, values, mask, master, labels, times, trends  = data
        print(data)


if __name__ == '__main__':
    main()