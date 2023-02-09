from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d(p=0.2)      # 0.5
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d(p=0.2)          # 0.5
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d(p=0.2)      # 0.5
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=1)

        self.conv3 = nn.Conv3d(inChans, 16, kernel_size=1)
        self.relu3 = ELUCons(elu, 16)

        self.conv5 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.bn5 = ContBatchNorm3d(16)
        self.relu5 = ELUCons(elu, 16)

        self.conv6 = nn.Conv3d(16, outChans, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):

        out_2 = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        out_2 = self.dropout(out_2)

        out_3 = self.relu3(self.conv3(out_2))

        out_6 = self.conv6(self.relu5(self.bn5(self.conv5(out_3))))

        return out_6



class OutputTransition_Edge(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(OutputTransition_Edge, self).__init__()
        self.conv1 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=1)

        self.conv3 = nn.Conv3d(inChans, 16, kernel_size=1)
        self.relu3 = ELUCons(elu, 16)

        self.conv5 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.bn5 = ContBatchNorm3d(16)
        self.relu5 = ELUCons(elu, 16)

        self.conv6 = nn.Conv3d(16, outChans, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):

        out_2 = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        out_2 = self.dropout(out_2)     # added

        out_3 = self.relu3(self.conv3(out_2))

        out_6 = self.conv6(self.relu5(self.bn5(self.conv5(out_3))))

        return out_6



class OutputTransition_Skele(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(OutputTransition_Skele, self).__init__()
        self.conv1 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=1)

        self.conv3 = nn.Conv3d(inChans, 16, kernel_size=1)

        self.conv5 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.bn5 = ContBatchNorm3d(16)

        self.conv6 = nn.Conv3d(16, outChans, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):

        out_2 = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        out_2 = self.dropout(out_2)     # added

        out_3 = self.conv3(out_2)

        out_6 = self.conv6(self.bn5(self.conv5(out_3)))

        return out_6


class MetaTransition_In(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(MetaTransition_In, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

        self.conv2 = nn.Conv3d(outChans, outChans // 2, kernel_size=3, padding=1)
        self.bn2 = ContBatchNorm3d(outChans // 2)
        self.relu2 = ELUCons(elu, outChans // 2)

        self.conv3 = nn.Conv3d(outChans//2, outChans // 2, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):

        out_1 = self.relu1(self.bn1(self.conv1(x)))
        out_2 = self.dropout(self.relu2(self.bn2(self.conv2(out_1))))
        out_3 = self.conv3(out_2)

        return out_3


class MetaTransition_Fused(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(MetaTransition_Fused, self).__init__()
        self.conv1 = nn.Conv3d(inChans, inChans, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(inChans)
        self.relu1 = ELUCons(elu, inChans)

        self.conv2 = nn.Conv3d(inChans, inChans // 2, kernel_size=3, padding=1)
        self.bn2 = ContBatchNorm3d(inChans // 2)
        self.relu2 = ELUCons(elu, inChans // 2)

        self.conv3 = nn.Conv3d(inChans // 2, outChans, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):

        out_1 = self.relu1(self.bn1(self.conv1(x)))
        out_1 = self.dropout(out_1)
        out_2 = self.relu2(self.bn2(self.conv2(out_1)))
        out_3 = self.conv3(out_2)

        return out_3


class BowelNet(nn.Module):
    def __init__(self, elu=True):
        super(BowelNet, self).__init__()

        num_baseC = 4   # 8

        self.in_tr = InputTransition(1, 2 * num_baseC, elu)

        # skeleton segmentor
        #########################################################################
        self.down_tr32_skele = DownTransition(2 * num_baseC, 1, elu)
        self.down_tr64_skele = DownTransition(4 * num_baseC, 2, elu)
        self.down_tr128_skele = DownTransition(8 * num_baseC, 2, elu, dropout=True)
        self.down_tr256_skele = DownTransition(16 * num_baseC, 2, elu, dropout=True)
        self.up_tr256_skele = UpTransition(32 * num_baseC, 32 * num_baseC, 2, elu, dropout=True)
        self.up_tr128_skele = UpTransition(32 * num_baseC, 16 * num_baseC, 2, elu, dropout=True)
        self.up_tr64_skele = UpTransition(16 * num_baseC, 8 * num_baseC, 1, elu)
        self.up_tr32_skele = UpTransition(8 * num_baseC, 4 * num_baseC, 1, elu)
        self.out_tr_skele = OutputTransition_Skele(4 * num_baseC, 3, elu)   # no relu for negative skeleton flux value
        self.out_tr_skele_mask = OutputTransition(4 * num_baseC, 2, elu)

        # boundary segmentor
        #########################################################################
        self.down_tr32_edge = DownTransition(2 * num_baseC, 1, elu)
        self.down_tr64_edge = DownTransition(4 * num_baseC, 2, elu)
        self.down_tr128_edge = DownTransition(8 * num_baseC, 2, elu, dropout=True)
        self.down_tr256_edge = DownTransition(16 * num_baseC, 2, elu, dropout=True)
        self.up_tr256_edge = UpTransition(32 * num_baseC, 32 * num_baseC, 2, elu, dropout=True)
        self.up_tr128_edge = UpTransition(32 * num_baseC, 16 * num_baseC, 2, elu, dropout=True)
        self.up_tr64_edge = UpTransition(16 * num_baseC, 8 * num_baseC, 1, elu)
        self.up_tr32_edge = UpTransition(8 * num_baseC, 4 * num_baseC, 1, elu)

        self.out_tr_edge = OutputTransition_Edge(4 * num_baseC, 1, elu)
        self.out_tr_edge_mask = OutputTransition(4 * num_baseC, 2, elu)

        # meta segmentor
        #########################################################################
        self.in_seg_meta = MetaTransition_In(1, 4 * num_baseC, elu)
        self.in_img_meta = MetaTransition_In(1, 4 * num_baseC, elu)
        self.fused_mask_meta = MetaTransition_Fused(4 * num_baseC, 2, elu)


    def forward(self, x, train_segmentor):

        assert train_segmentor in ['base', 'meta']

        out16 = self.in_tr(x)

        out32_skele = self.down_tr32_skele(out16)
        out64_skele = self.down_tr64_skele(out32_skele)
        out128_skele = self.down_tr128_skele(out64_skele)
        out256_skele = self.down_tr256_skele(out128_skele)
        out_skele = self.up_tr256_skele(out256_skele, out128_skele)
        out_skele = self.up_tr128_skele(out_skele, out64_skele)
        out_skele = self.up_tr64_skele(out_skele, out32_skele)
        out_skele = self.up_tr32_skele(out_skele, out16)  # same resolution as input
        out_skele_seg = self.out_tr_skele(out_skele)
        out_skele_mask_seg = self.out_tr_skele_mask(out_skele)

        out32_edge = self.down_tr32_edge(out16)
        out64_edge = self.down_tr64_edge(out32_edge)
        out128_edge = self.down_tr128_edge(out64_edge)
        out256_edge = self.down_tr256_edge(out128_edge)
        out_edge = self.up_tr256_edge(out256_edge, out128_edge)
        out_edge = self.up_tr128_edge(out_edge, out64_edge)
        out_edge = self.up_tr64_edge(out_edge, out32_edge)
        out_edge = self.up_tr32_edge(out_edge, out16)     # same resolution as input
        out_edge_seg = self.out_tr_edge(out_edge)
        out_edge_mask_seg = self.out_tr_edge_mask(out_edge)

        if train_segmentor == 'base':
            return out_skele_seg, out_skele_mask_seg, out_edge_seg, out_edge_mask_seg
        else:
            # in_seg_avg = 1.0*torch.argmax(0.5 * (F.softmax(out_edge_mask_seg, dim=1) + F.softmax(out_skele_mask_seg, dim=1)), dim=1).unsqueeze(1)
            in_seg_avg = 0.5 * (torch.argmax(out_edge_mask_seg, dim=1).unsqueeze(1) + torch.argmax(out_skele_mask_seg, dim=1).unsqueeze(1))
            seg_feat = self.in_seg_meta(in_seg_avg)
            img_feat = self.in_img_meta(x)
            comb_feat = torch.cat((img_feat, seg_feat), dim=1)
            out = self.fused_mask_meta(comb_feat)
            return out, out_edge_mask_seg, out_skele_mask_seg