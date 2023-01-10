# coding:utf8
import torch
import torch.nn as nn
from torch.autograd import Variable

class TestDiceLoss(nn.Module):
    def __init__(self, n_class):
        super(TestDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, pred, label, show=False):
        smooth = 0.00001
        batch_size = pred.size(0)
        pred = torch.max(pred, 1)[1]
        pred = self.one_hot_encoder(pred).contiguous().view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        inter = torch.sum(torch.sum(pred * label, 2), 0) + smooth
        union1 = torch.sum(torch.sum(pred, 2), 0) + smooth
        union2 = torch.sum(torch.sum(label, 2), 0) + smooth

        andU = 2.0 * inter / (union1 + union2)
        score = andU

        return score.float()

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()  # torch.sparse.torch.eye
                                             

    def forward(self, X_in):
        '''
        :param X_in: batch*depth*length*height or batch*length*height
        :return: batch*class*depth*length*height or batch*calss*length*height
        '''
        n_dim = X_in.dim()  
        output_size = X_in.size() + torch.Size([self.depth])  
        num_element = X_in.numel()  
        X_in = X_in.data.long().view(num_element)  
        out1 = Variable(self.ones.index_select(0, X_in))
        out = out1.view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class SoftDiceLoss(nn.Module):
    def __init__(self, n_class):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        smooth = 1
        batch_size = pred.size(0)
        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        inter = torch.sum(pred * label, 2) + smooth
        union1 = torch.sum(pred, 2) + smooth
        union2 = torch.sum(label, 2) + smooth

        andU = torch.sum(2.0 * inter/(union1 + union2))
        score = 1 - andU/(batch_size*self.n_class)

        return score


class CrossEntropy_and_Dice_Loss(nn.Module):
    def __init__(self, n_class, lamda=1):
        super(CrossEntropy_and_Dice_Loss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.lamda = lamda
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.SoftDiceloss = SoftDiceLoss(n_class)

    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        score = self.lamda*self.CrossEntropyLoss(pred, label)+self.n_class*self.SoftDiceloss(pred, label)
        return score