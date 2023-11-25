import numpy as np
import torch
import torch.nn as nn

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes[1],
                        number_of_classes[0],
                        filter_size,
                        filter_size,), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes[1]):
        for j in range(number_of_classes[0]):
            weights[i,j, :, :] = upsample_kernel
    return torch.Tensor(weights)

def get_padding(output_size, input_size, factor):
    TH = output_size[2] - ((input_size[2]-1)*factor) - (factor*2)
    TW = output_size[3] - ((input_size[3]-1)*factor) - (factor*2)
    padding_H = int(np.ceil(TH / (-2)))
    out_padding_H = TH - padding_H*(-2)

    padding_W = int(np.ceil(TW / (-2)))
    out_padding_W = TW - padding_W*(-2)
    return (padding_H, padding_W), (out_padding_H, out_padding_W)

def cfgs2name(cfgs):
    name = '%s_%s_%s(%s,%s,%s)' % \
            (cfgs['dataset'], cfgs['backbone'], cfgs['loss'], cfgs['a'], cfgs['b'],cfgs['c'])
    if 'MultiCue' in cfgs['dataset']:
        name = name + '_' + str(cfgs['multicue_seq'])
    return name

#交叉熵
class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        # self.weight1 = nn.Parameter(torch.Tensor([1.]))
        # self.weight2 = nn.Parameter(torch.Tensor([1.]))

    def forward(self, pred, labels, side_output=None):
        # def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]

        total_loss = cross_entropy_per_image(pred, labels)+ 0.00 * 0.1 * dice_loss_per_image(pred, labels)
        if side_output is not None:
            for s in side_output:
                total_loss += cross_entropy_per_image(s, labels) / len(side_output)

        # total_loss = cross_entropy_per_image(pred, labels)
        # total_loss = dice_loss_per_image(pred, labels)
        # total_loss = 1.00 * cross_entropy_per_image(pred, labels) + \
        # 0.00 * 0.1 * dice_loss_per_image(pred, labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        return total_loss, (1-pred_pos).abs(), pred_neg

def dice(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    dice = ((logits * labels).sum() * 2 + eps) / (logits.sum() + labels.sum() + eps)
    dice_loss = dice.pow(-1)
    return dice_loss


def dice_loss_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += dice(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_with_weight(_logit, _label)
    return total_loss / len(logits)
def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]
    # weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
    #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy

def cross_entropy_orignal(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels >= 0.5].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)

    weight_pos, weight_neg = get_weight(labels, labels, 0.4, 1.5)

    cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
                            (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy

def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]
    # weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
    #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy

def get_weight(src, mask, threshold, weight):
    count_pos = src[mask >= threshold].size()[0]
    count_neg = src[mask == 0.0].size()[0]
    total = count_neg + count_pos
    weight_pos = count_neg / total
    weight_neg = (count_pos / total) * weight
    return weight_pos, weight_neg

def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)