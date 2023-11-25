import torch
import transforms
import model
from data import BSDS_500
import yaml
import utils
import time
import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torchvision
if __name__ == '__main__':
    fild_id = open('./cfgs.yaml')
    cfgs = yaml.load(fild_id, yaml.FullLoader)
    fild_id.close()
    if not os.path.exists('./validation/'):
        os.makedirs('./validation/')
    if not os.path.exists('./checkpoint/'):  #存放模型的文件夹
        os.makedirs('./checkpoint/')



    trans = transforms.Compose([
        # transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 数据加载，用师兄的当然最快（白嫖最香）
    dataset = BSDS_500(root=cfgs['dataset'], VOC=True, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=6)
    net = model.Net().train()
    criterion = utils.Cross_Entropy()

    # optimal
    if cfgs['method'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': criterion.parameters()}],
                                     lr=cfgs['lr'])
    elif cfgs['method'] == 'SGD':
        optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': criterion.parameters()}],
                                    lr=cfgs['lr'], momentum=cfgs['momentum'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device is :', torch.cuda.get_device_name(device))

    net.to(device)
    criterion.to(device)


    for epoch in range(cfgs['max_epoch']):
        # utils.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'] )
        running_loss = 0.0
        if epoch == 5:
            optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': criterion.parameters()}],
                                              lr=cfgs['lr']*0.1)
        # if epoch == 7:
        #     optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': criterion.parameters()}],
        #                                  lr=cfgs['lr'] * 0.01)
        for i, data in enumerate(dataloader):
            start_time = time.time()
            optimizer.zero_grad()
            # forward + backward + optimize
            images = data['images'].to(device)
            labels = data['labels'].to(device)

            prediction= net(images)
            loss, dp, dn = criterion(prediction, labels)

            loss.backward() # 更新梯度
            optimizer.step()# 根据当前学习率去优化参数
            duration = time.time() - start_time
            print_epoch = 100
            running_loss += loss.item()
            if i % print_epoch == print_epoch - 1:  # print every 10 mini-batches  窗口打印训练过程中的 loss
                examples_per_sec = cfgs['batch_size'] / duration  # 每秒训练多少个样本
                sec_per_batch = float(duration)  # 一个batch多少时间
                format_str = '%s: step [%d, %5d/%4d], lr = %e, loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), epoch + 1, i + 1, len(dataloader), optimizer.param_groups[0]['lr'],
                                    running_loss / print_epoch, examples_per_sec, sec_per_batch))
                running_loss = 0.0

                # validation 保存训练过程中的图片，主要是用来观察出来的结果对不对
            validation_epoch = 100
            if not os.path.exists('./validation/'):
                os.makedirs('./validation/')
            if i % validation_epoch == validation_epoch - 1:  # 第10个batch的预测做记录

                prediction = net(images)
                prediction = prediction.cpu().detach().numpy().transpose((0, 2, 3, 1)) # 装换成 batch * n * n * 通道
                # B, C, H, W = prediction[0].shape
                # results_all = torch.zeros((len(prediction), 1, H, W))
                # for i in range(len(prediction)):
                #     results_all[i, 0, :, :] = prediction[i]
                # if not os.path.exists(os.path.join('./', "val")):
                #     os.makedirs(os.path.join('./', "val"))
                # torchvision.utils.save_image(1 - results_all,
                #                              os.path.join('./', "val", "epoch%s.jpg" % epoch))
                for j in range(prediction.shape[0]): # 将每个batch 的图像取出来 让后写近图片
                    cv2.imwrite('./validation/' + str(j) + '.png', prediction[j] * 255)

                #  Print the ratio of outline pixels to background pixels 轮廓像素与背景像素的比例
                ax = plt.subplot(1, 2, 1)  # 第一幅图是正比例
                data_ = dp.cpu().detach().numpy()  # 正类预测
                ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))  # 绘制正类预测的直方图

                ax = plt.subplot(1, 2, 2)  # 绘制负类的直方图
                data_ = dn.cpu().detach().numpy()
                ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                plt.savefig('./validation/test' + str(epoch) + '.png')
                plt.close('all')

                # save  保存所训练的网络模型(保存参数数据)
        save_epoch = str(epoch)
        torch.save(net.state_dict(), './checkpoint/' + save_epoch + cfgs['save_name'])

    print('Finished Training')

