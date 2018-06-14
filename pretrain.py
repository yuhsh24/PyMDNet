# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from models import MDNet
import reader
import proc
import os
import argparse


class Config(object):
    momentum = 0.9  # 冲量
    weight_decay = 0.0005  # 权重衰减
    lr_rate = 0.0001  # 基础学习率
    lr_rates = {'conv': 1.0, 'bias': 2.0, 'fc6-conv': 10.0, 'fc6-bias': 20.0}  # 额外学习率

    batch_frames = 8  #
    batch_size = 128  # 总体样本的数目
    batch_pos = 32  # 正样本的数目
    batch_neg = 96  # 负样本的数目
    num_cycle = 100  #

    posPerFrame = 50
    negPerFrame = 200
    scale_factor = 1.05
    input_size = 107

    pos_range = [0.7, 1]
    neg_range = [0, 0.5]


# 预训练mdnet模型
def pretrain_mdnet(datasets, init_model_path, result_dir, load_path=None, shuffle=True,
                   norm=False, dropout=True, regularization=True):
    config = Config()  # 获取配置的参数

    # print parameters
    print('shuffle', shuffle)  # 是否shuffle
    print('norm', norm)  # 是否归一化
    print('dropout', dropout)  # 是否使用dropout层
    print('regularization', regularization)  # 是否使用正则化
    print('init_model_path', init_model_path)  # 初始化网络的路径
    print('result_dir', result_dir)  # 模型保存路径

    # create directory
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)  # 创建模型保存路径的目录

    # load sequences
    train_data = reader.read_datasets(datasets)  # 获取训练数据
    K = len(train_data.data)  # 获取数据集的视频数目

    # create session and saver
    gpu_config = tf.ConfigProto(allow_soft_placement=True)  # tensorflow网络配置
    sess = tf.InteractiveSession(config=gpu_config)  # 创建session

    # load model, weights
    model = MDNet(config)  # 定义MDNet对象
    model.build_trainer(K, config.batch_size, dropout=dropout, regularization=regularization)  # 创建trainer网络
    tf.global_variables_initializer().run()  # 变量初始化
    model.load(init_model_path,sess)  # load网络
    sess.run(model.lr_rate.assign(config.lr_rate))  # 学习率赋值

    # create saver
    saver = tf.train.Saver([v for v in tf.global_variables() if 'fc6' not in v.name])  # 除了第6层以外都保存

    # restore from model
    if load_path is not None:
        saver.restore(sess, load_path)  # 加载网络

    # prepare roidb and frame list
    train_loss_file = open(os.path.join(result_dir, 'train_loss.txt'), 'w')  # 写train_loss的文件
    n_frames = config.batch_frames*config.num_cycle  # 一共使用的帧数
    for i in range(config.num_cycle):
        loss_total = 0  # loss清空
        print('###### training cycle '+str(i)+'/'+str(config.num_cycle)+'...')  # 第几轮training

        seq_i = 0  # 记录当前轮次使用的序列数目
        for seq, seq_data in train_data.data.iteritems():
            print('### training video "'+seq+'"...')  # 训练的视频名字
            seq_n_frames = len(seq_data.frames)  #
    
            ## prepare roidb
            print('- preparing roidb...')
            seq_data.rois = proc.seq2roidb(seq_data, config)

            ## prepare frame list
            print('- shuffle frames...')
            seq_data.frame_lists = []
            while len(seq_data.frame_lists) < n_frames:
              seq_data.frame_lists = np.r_[seq_data.frame_lists, np.random.permutation(seq_n_frames)]
            seq_data.frame_lists = seq_data.frame_lists[:n_frames]

            ## start training
            # extract batch_size frames
            frame_inds = seq_data.frame_lists[config.batch_frames * i: config.batch_frames * (i+1)].astype(np.int)

            # sample boxes
            pos_boxes = np.concatenate([seq_data.rois[frame_ind].pos_boxes for frame_ind in frame_inds], axis=0)
            neg_boxes = np.concatenate([seq_data.rois[frame_ind].neg_boxes for frame_ind in frame_inds], axis=0)
            pos_inds = np.random.permutation(config.posPerFrame * config.batch_frames)[:config.batch_pos]
            neg_inds = np.random.permutation(config.negPerFrame * config.batch_frames)[:config.batch_neg]
      
            # pack as boxes, paths
            pos_boxes = pos_boxes[pos_inds]
            neg_boxes = neg_boxes[neg_inds]
            boxes = np.r_[pos_boxes, neg_boxes]

            box_relinds = np.r_[pos_inds // config.posPerFrame, neg_inds // config.negPerFrame]
            paths = [seq_data.frames[ind] for ind in frame_inds[box_relinds]]
            gts = np.repeat(np.identity(2), [config.batch_pos, config.batch_neg], axis=0)
            patches = proc.load_patch(paths, boxes, norm=False)

            # shuffle
            if shuffle:
              inds = np.random.permutation(config.batch_size)
              patches = patches[inds]
              gts = gts[inds]

            # training
            _, loss, score, weight, bias = sess.run([model.trainable[seq_i],
                                                     model.losses['loss-'+str(seq_i)],
                                                     model.layers['fc6-'+str(seq_i)],
                                                     model.weights['fc6-'+str(seq_i)],
                                                     model.biases['fc6-'+str(seq_i)]],
                                                     feed_dict={model.layers['input']: patches,
                                                                model.layers['y-'+str(seq_i)]: gts})
            print(seq_i)
            print(score.reshape(-1, 2)[:5])
            print(gts[:5])
            print(np.mean(loss))
            print(weight)
            print(bias)
            loss_total += np.mean(loss)

            # update seq_i
            seq_i += 1

        ## save the model
        train_loss_file.write('Epoch '+str(i)+', Loss: '+str(np.mean(loss)))
        saver.save(sess, os.path.join(result_dir, 'model_e'+str(i)+'.ckpt'), global_step=i+1)
    train_loss_file.close()

# 配置函数
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_shuffle', action='store_true', help='disable shuffling frames')  # 是否shuffle帧数
    parser.add_argument('--norm', action='store_true', help='normalize input image')  # 是否对图像进行归一化
    parser.add_argument('--no_dropout', action='store_true', help='disable dropout')  # 是否加入dropout层数
    parser.add_argument('--no_regularization', action='store_true', help='disable regularization')  # 是否使用正则化
    parser.add_argument('--result_dir', help='places to store the pretrained model')  # 保存模型的目录
    parser.add_argument('--dataset', choices=['otb', 'vot', 'otb_vot'], help='choose pretrained dataset: [vot/otb/otb_vot]')  # 数据集选择
    parser.add_argument('--init_model_path', help='initial model path')  # 初始化模型的地址
    parser.add_argument('--load_path', default=None, help='initial model path')  # load path
    return parser.parse_args()

# 主函数
def main():
    params = get_params()  # 获取配置
    if params.dataset == 'otb':
        datasets = ['otb']  # otb数据集
    elif params.dataset == 'vot':
        datasets = ['vot2013', 'vot2014', 'vot2015']  # vot数据集
    elif params.dataset == 'otb_vot':
        datasets = ['otb', 'vot2013', 'vot2014', 'vot2015']  # otb和vot数据集
    #预训练mdnet函数
    pretrain_mdnet(datasets, load_path=params.load_path, init_model_path=params.init_model_path, result_dir=params.result_dir,
                   shuffle=(not params.no_shuffle), norm=params.norm, dropout=(not params.no_dropout), regularization=(not params.no_regularization))

if __name__ == '__main__':
    main()
