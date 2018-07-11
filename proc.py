# -*- coding:utf-8 -*-
from utils import Data
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import io, transform

###########################################################################
#                            load_patch                                   #
###########################################################################
def load_box(inp, bboxes, img_input=False, norm=False):
  n = bboxes.shape[0]
  if img_input:
    im = inp
  else:
    im = load_image(inp, norm)

  res = np.zeros([n, 117, 117, 3])
  for i in range(n):
    img_crop = im_crop(im, bboxes[i])
    img_resize = transform.resize(img_crop, [117, 117])
    res[i] = img_resize
  return res

###########################################################################
#                            load_patch                                   #
###########################################################################
def load_patch(paths, bboxes, norm=False):
  n = len(paths)

  res = np.zeros([n, 117, 117, 3])
  for i in range(n):
    path = paths[i]
    bbox = bboxes[i]
    img_crop = im_crop(load_image(path, norm=norm), bbox)
    img_resize = transform.resize(img_crop, [117, 117])
    res[i] = img_resize
  return res

###########################################################################
#                            load_image                                   #
###########################################################################
def load_image(path, norm=False):
  # load image
  img = io.imread(path)
  if len(img.shape) == 2:
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
  if norm:
    img = img.astype(np.float32) / 255.0 - 0.5
  return img

###########################################################################
#                            im_crop                                      #
###########################################################################
def im_crop(im, bbox):
  bbox = np.around(bbox).astype(np.int)
  h, w, c = im.shape

  h_bot = np.maximum(0, bbox[1])
  h_top = np.minimum(h, bbox[1]+bbox[3])
  w_bot = np.maximum(0, bbox[0])
  w_top = np.minimum(w, bbox[0]+bbox[2])
  im_cropped = im[h_bot:h_top, w_bot:w_top]
  return im_cropped

###########################################################################
#                          seq2roidb                                      #
# 处理好roi
###########################################################################
def seq2roidb(seq_data, config):
    gts = seq_data.gts  # 每一帧的真值
    frames = seq_data.frames  # 每一帧的路径
    im = io.imread(frames[0])  # 读入第一张图像
    im_size = im.shape[:2]  # 获取图像的size
    return sample_rois(frames, gts, im_size, config)  # 采样ROI

# 对矩形框进行采样
def sample_rois(frames, gts, im_size, config):
    '''
     rois = [ <img_path: string, pos_boxes: np.array, neg_boxes: np.array> ]
    '''
    rois = []  # 保存正负样本

    for i in range(len(frames)):  # 对每张图像进行处理
        #print(str(i))
        target = gts[i]  # 获取真值的矩形框
    
        # sample postive boxes
        verbose = False  # 控制是否采样结束的变量
        pos_examples = np.array([]).reshape([0, 4])  # 保存正样本
        while len(pos_examples) < config.posPerFrame-1:  # 根据采集的正样本数目决定是否跳出
            pos = sample(target, config.posPerFrame*5, im_size, config.scale_factor, 0.1, 5, False, verbose)  # 采样
            r = overlap_ratio(pos, target)  # 计算正样本的IOU值
            pos = pos[np.logical_and(r > config.pos_range[0], r <= config.pos_range[1])]  # 获取满足IOU要求的矩形框

            if verbose:
                exit(0)

            if pos.shape[0] == 0:  # 当一个样本都没有采集成功时就退出
                verbose = True
                continue
            index = np.arange(pos.shape[0])
            np.random.shuffle(index)
            index = index[:min(pos.shape[0],config.posPerFrame-pos_examples.shape[0]-1)]  # 防止采集数量过多
            pos_examples = np.r_[pos_examples,pos[index]]  # 记录所有的正样本

        # sample negative boxes
        verbose = False
        neg_examples = np.array([]).reshape([0, 4])
        while len(neg_examples) < config.negPerFrame :
            neg = sample(target, config.negPerFrame*2, im_size, config.scale_factor, 2, 10, True, verbose)  # 采样矩形框
            r = overlap_ratio(neg, target)  #
            neg = neg[np.logical_and(r>=config.neg_range[0],r<config.neg_range[1])]

            if verbose:
                exit(0)

            if neg.shape[0] == 0:
                verbose = True
                continue
            index = np.arange(neg.shape[0])
            np.random.shuffle(index)
            index = index[:min(neg.shape[0],config.negPerFrame-neg_examples.shape[0])]
            neg_examples = np.r_[neg_examples,neg[index]]

        # pack into rois
        rois.append(Data())
        rois[-1].img_path = frames[i]
        rois[-1].pos_boxes = np.r_[pos_examples,target.reshape(1,-1)]
        rois[-1].neg_boxes = neg_examples
    return rois

# 采样的函数
def sample(gt, n, im_size, scale_factor, transfer_range, scale_range, valid, verbose=False):
    '''
    gt: 矩形框真值
    n: 采集的总数
    im_size: 图像的大小
    scale_factor: 尺度因子
    transfer_range: 平移
    scale_range: 尺度
    valid: 是否有效
    verbose: 是否详细显示
    '''
    samp = np.array([gt[0]+gt[2]/2.0, gt[1]+gt[3]/2.0, gt[2], gt[3]])
    samples = np.repeat(np.reshape(samp, [1, -1]), n, axis=0)  # 创建二维矩阵保存样本
    h, w = im_size  # 记录图像的高和宽

    if verbose:
      print(w, h)
      print(gt)
      print(samp)
      print(transfer_range)
      print(scale_range)

    samples[:, 0] = np.add(samples[:, 0], transfer_range*samp[2]*(np.random.rand(n)*2-1))  # 中心点的x坐标平移
    samples[:, 1] = np.add(samples[:, 1], transfer_range*samp[3]*(np.random.rand(n)*2-1))  # 中心点的y坐标平移
    samples[:, 2:] = np.multiply(samples[:, 2:], np.power(scale_factor, scale_range*np.repeat(np.random.rand(n,1)*2-1,2,axis=1)))  # 产生不同宽高的样本
    samples[:, 2] = np.maximum(0, np.minimum(w-5, samples[:,2]))  # 截断宽度
    samples[:, 3] = np.maximum(0, np.minimum(h-5, samples[:,3]))  # 截断高度
  
    if verbose:
      print(samples[0])

    samples = np.c_[samples[:,0]-samples[:,2]/2, samples[:,1]-samples[:,3]/2, samples[:,2], samples[:,3]]
  
    if verbose:
      print(samples[0])

    # 确保左上角的点在图像内
    if valid:
      samples[:,0] = np.maximum(0,np.minimum(w-samples[:,2],samples[:,0]))
      samples[:,1] = np.maximum(0,np.minimum(h-samples[:,3],samples[:,1]))
    else:
      samples[:,0] = np.maximum(0-samples[:,2]/2,np.minimum(w-samples[:,2]/2,samples[:,0]))
      samples[:,1] = np.maximum(0-samples[:,3]/2,np.minimum(h-samples[:,3]/2,samples[:,1]))
  
    if verbose:
      print(samples[0])
    return samples

###########################################################################
#                          overlap_ratio                                  #
###########################################################################
'''
计算IOU的函数
'''
def overlap_ratio(boxes1, boxes2):
    # find intersection bbox
    x_int_bot = np.maximum(boxes1[:, 0], boxes2[0])
    x_int_top = np.minimum(boxes1[:, 0] + boxes1[:, 2], boxes2[0] + boxes2[2])
    y_int_bot = np.maximum(boxes1[:, 1], boxes2[1])
    y_int_top = np.minimum(boxes1[:, 1] + boxes1[:, 3], boxes2[1] + boxes2[3])

    # find intersection area
    dx = x_int_top - x_int_bot
    dy = y_int_top - y_int_bot
    area_int = np.where(np.logical_and(dx>0, dy>0), dx * dy, np.zeros_like(dx))

    # find union
    area_union = boxes1[:,2] * boxes1[:,3] + boxes2[2] * boxes2[3] - area_int

    # find overlap ratio
    ratio = np.where(area_union > 0, area_int/area_union, np.zeros_like(area_int))
    return ratio


###########################################################################
#                          overlap_ratio of two bboxes                    #
###########################################################################
def overlap_ratio_pair(boxes1, boxes2):
  # find intersection bbox
  x_int_bot = np.maximum(boxes1[:, 0], boxes2[:, 0])
  x_int_top = np.minimum(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
  y_int_bot = np.maximum(boxes1[:, 1], boxes2[:, 1])
  y_int_top = np.minimum(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])

  # find intersection area
  dx = x_int_top - x_int_bot
  dy = y_int_top - y_int_bot
  area_int = np.where(np.logical_and(dx>0, dy>0), dx * dy, np.zeros_like(dx))

  # find union
  area_union = boxes1[:,2] * boxes1[:,3] + boxes2[:, 2] * boxes2[:, 3] - area_int

  # find overlap ratio
  ratio = np.where(area_union > 0, area_int/area_union, np.zeros_like(area_int))
  return ratio
