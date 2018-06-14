# -*- coding: utf-8 -*-
from __future__ import print_function
from utils import Data
import os
import pandas as pd
import numpy as np
from skimage import io


# 读入所有的数据集
def read_datasets(datasets):
    assert(len(datasets) > 0)  # 确保数据集路径不为空

    res = read_dataset(datasets[0])  # 读入第一个数据集
    for i in range(1, len(datasets)):
        data = read_dataset(datasets[i])
        res.dataset += '/'+data.dataset
        res.data.update(data.data)
    return res

# 读入一个数据集
def read_dataset(dataset):
    """
    data format:
      data = <
        dataset: 'otb'/'vot2013'/'vot2014'/'vot2015'
        data: {'seq': <gts: np.array, frames: [string]>}     
        >
    """
    # create the container
    data = Data()  # 空数据
    data.dataset = dataset  # data的dataset成员

    # find the current path
    cur_dir = os.path.dirname(os.path.realpath(__file__))  # 获取当前路径

    if dataset[:3] == 'vot':  # vot数据集的读入方式
        # find sequences
        dir_path = os.path.join(cur_dir, 'data', 'vot', dataset)  # 获取路径
        seqs = [path for path in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, path))]  # 获取目录下所有视频名字
        # load sequences
        data.data = {}
        for seq in seqs:
            print('loading...', dataset, seq)
            gt_path = os.path.join(dir_path, seq, 'groundtruth.txt')
            gt = pd.read_csv(gt_path, header=None).as_matrix()  # 读入groud truth的文件数据
            if gt.shape[1] >= 6:
                x = np.amin(gt[:, ::2], axis=1)
                y = np.amin(gt[:, 1::2], axis=1)
                width = np.amax(gt[:, ::2], axis=1) - x
                height = np.amax(gt[:, 1::2], axis=1) - y
                gt = np.c_[x, y, width, height]  # 获取每帧的真值

            data.data[seq] = Data()
            data.data[seq].gts = gt  # 保存每帧的真值
            data.data[seq].frames = [os.path.join(dir_path, seq, '%08d.jpg'%(i+1)) for i in range(gt.shape[0])]  # 保存每帧的路径

    elif dataset[:3] == 'otb':  # otb数据集的读入方式
        # find sequences
        dir_path = os.path.join(cur_dir, 'data', 'otb')  # 保存otb数据集的目录
        seqs = [path for path in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path,path))
                and path != 'Jogging' and path != 'Skating2']  # 获取目录下所有视频名字
        seqs = seqs + ['Jogging-1', 'Jogging-2', 'Skating2-1', 'Skating2-2']  # 由于这些视频的真值保存具有特殊性，所以特殊处理
        # load sequences
        data.data = {}
        for seq in seqs:
            print('loading...', dataset, seq)  # 载入指定数据集的指定视频
            if seq == 'Jogging-1' or seq == 'Jogging-2' or seq == 'Skating2-1' or seq == 'Skating2-2':  # 特殊视频处理
                gt_path = os.path.join(dir_path, seq[:-2], 'groundtruth_rect.'+seq[-1]+'.txt')  # groundtruth单独一份
                img_dir_path = os.path.join(dir_path, seq[:-2], 'img')  # 图片共享一份
            elif seq == 'Human4':  # 特殊数据集
                gt_path = os.path.join(dir_path, seq, 'groundtruth_rect.2.txt')
                img_dir_path = os.path.join(dir_path, seq, 'img')
            else:
                gt_path = os.path.join(dir_path, seq, 'groundtruth_rect.txt')  # 普遍视频的groundtruth文件格式
                img_dir_path = os.path.join(dir_path, seq, 'img')  # 普通视频的图片前缀
            gt = pd.read_csv(gt_path, header=None, sep='[\s*\,*]+').as_matrix()  # 读入真值

            data.data[seq] = Data()
            data.data[seq].gts = gt  # 保存真值
            if seq == 'Board':
                data.data[seq].frames = [os.path.join(img_dir_path, '%05d.jpg'%(i+1)) for i in range(gt.shape[0])]
            elif seq == 'BlurCar1':
                data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+247)) for i in range(gt.shape[0])]
            elif seq == 'BlurCar3':
                data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+3)) for i in range(gt.shape[0])]
            elif seq == 'BlurCar4':
                data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+18)) for i in range(gt.shape[0])]
            else:
                data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+1)) for i in range(gt.shape[0])]

            for i in range(len(data.data[seq].frames)):
                if gt[i][2] < 1 or gt[i][3] < 1:
                  print(str(i))
                  exit(0)

            # test for the existence of the first frame
            print('testing...',dataset,seq)
            io.imread(data.data[seq].frames[0])
    return data

def read_seq(dataset, seq):
  
  # create the container
  data = Data()
  data.dataset = dataset

  # find the current path
  cur_dir = os.path.dirname(os.path.realpath(__file__))

  print('loading...',dataset,seq)
  if dataset[:3] == 'vot':
    dir_path = os.path.join(cur_dir, 'data', 'vot', dataset)

    data.data = {}
    gt_path = os.path.join(dir_path,seq,'groundtruth.txt')
    gt = pd.read_csv(gt_path, header=None).as_matrix()
    if gt.shape[1] >= 6:
      x = np.amin(gt[:, ::2], axis=1)
      y = np.amin(gt[:, 1::2], axis=1)
      width = np.amax(gt[:, ::2], axis=1) - x
      height = np.amax(gt[:, 1::2], axis=1) - y
      gt = np.c_[x, y, width, height]

    data.data[seq] = Data()
    data.data[seq].gts = gt
    data.data[seq].frames = [os.path.join(dir_path, seq, '%08d.jpg'%(i+1)) for i in range(gt.shape[0])]
  elif dataset[:3] == 'otb':
    dir_path = os.path.join(cur_dir, 'data', 'otb')

    data.data = {}
    if seq == 'Jogging-1' or seq == 'Jogging-2' or seq == 'Skating2-1' or seq == 'Skating2-2':
      gt_path = os.path.join(dir_path,seq[:-2],'groundtruth_rect.'+seq[-1]+'.txt')
      img_dir_path = os.path.join(dir_path,seq[:-2],'img')
    elif seq == 'Human4':
      gt_path = os.path.join(dir_path,seq,'groundtruth_rect.2.txt')
      img_dir_path = os.path.join(dir_path,seq,'img')
    else:
      gt_path = os.path.join(dir_path,seq,'groundtruth_rect.txt')
      img_dir_path = os.path.join(dir_path,seq,'img')
    gt = pd.read_csv(gt_path, header=None, sep='[\s*\,*]+').as_matrix()

    data.data[seq] = Data()
    data.data[seq].gts = gt
    if seq == 'Board':
      data.data[seq].frames = [os.path.join(img_dir_path, '%05d.jpg'%(i+1)) for i in range(gt.shape[0])]
    elif seq == 'BlurCar1':
      data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+247)) for i in range(gt.shape[0])]
    elif seq == 'BlurCar3':
      data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+3)) for i in range(gt.shape[0])]
    elif seq == 'BlurCar4':
      data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+18)) for i in range(gt.shape[0])]
    else:
      data.data[seq].frames = [os.path.join(img_dir_path, '%04d.jpg'%(i+1)) for i in range(gt.shape[0])]

    for i in range(len(data.data[seq].frames)):
      if gt[i][2] < 1 or gt[i][3] < 1:
        print(str(i))
        exit(0)

    # test for the existence of the first frame
    print('testing...',dataset,seq)
    io.imread(data.data[seq].frames[0])
  return data

if __name__ == '__main__':
    print(read_dataset('otb'))
    print(read_dataset('vot2013'))
    print(read_dataset('vot2014'))
    print(read_dataset('vot2015'))
