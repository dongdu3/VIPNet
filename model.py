import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from layers import (CResnetBlockConv1d, CBatchNorm1d, CBatchNorm1d_legacy)
import resnet
import numpy as np

class VoxelDecoder64(nn.Module):
    ''' Voxel64 Decoder with batch normalization (BN) class.
        Args:
            z_dim (int): input feature z dimension
            gf_dim (int): dimension of feature channel
    '''
    def __init__(self, z_dim=256, gf_dim=256):
        super(VoxelDecoder64, self).__init__()

        self.z_dim = z_dim
        self.gf_dim = gf_dim

        self.fc_z = nn.Sequential(
            nn.Conv1d(self.z_dim, self.gf_dim * 2 * 2 * 2, 1),
            nn.BatchNorm1d(self.gf_dim * 2 * 2 * 2),
            nn.LeakyReLU(0.2)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim, self.gf_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.gf_dim),
            nn.LeakyReLU(0.2)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim, self.gf_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.gf_dim//2),
            nn.LeakyReLU(0.2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim//2, self.gf_dim//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.gf_dim//4),
            nn.LeakyReLU(0.2)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim//4, self.gf_dim//8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.gf_dim//8),
            nn.LeakyReLU(0.2)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim//8, self.gf_dim//16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.gf_dim//16),
            nn.LeakyReLU(0.2)
        )

        self.deconv6 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim//16, 1, kernel_size=1, stride=1, padding=0)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = z.contiguous().view(-1, self.z_dim, 1)
        net = self.fc_z(z)
        net = net.contiguous().view(-1, self.gf_dim, 2, 2, 2)
        # print(net.size())  # torch.Size([-1, 256, 2, 2, 2])
        net = self.deconv1(net)
        # print(net.size())  # torch.Size([-1, 256, 4, 4, 4])
        net = self.deconv2(net)
        # print(net.size())  # torch.Size([-1, 128, 8, 8, 8])
        net = self.deconv3(net)
        # print(net.size())  # torch.Size([-1, 64, 16, 16, 16])
        net = self.deconv4(net)
        # print(net.size())  # torch.Size([-1, 32, 32, 32, 32])
        net = self.deconv5(net)
        # print(net.size())  # torch.Size([-1, 16, 64, 64, 64])
        out = self.deconv6(net)
        # print(out.size())  # torch.Size([-1, 1, 64, 64, 64])
        out_sigmoid = self.sigmoid(out)

        return out, out_sigmoid

    def provide_middle_feature(self, z):    # provide
        z = z.contiguous().view(-1, self.z_dim, 1)
        feat = self.fc_z(z)
        feat = feat.contiguous().view(-1, self.gf_dim, 2, 2, 2)
        # print(net.size())  # torch.Size([-1, 256, 2, 2, 2])
        feat = self.deconv1(feat)
        # print(net.size())  # torch.Size([-1, 256, 4, 4, 4])
        feat = self.deconv2(feat)
        # print(net.size())  # torch.Size([-1, 128, 8, 8, 8])
        feat = self.deconv3(feat)
        # print(net.size())  # torch.Size([-1, 64, 16, 16, 16])

        return feat     # return global feature for patch high-resolution

    def predict_with_middle_feature(self, z):
        z = z.contiguous().view(-1, self.z_dim, 1)
        net = self.fc_z(z)
        net = net.contiguous().view(-1, self.gf_dim, 2, 2, 2)
        # print(net.size())  # torch.Size([-1, 256, 2, 2, 2])
        net = self.deconv1(net)
        # print(net.size())  # torch.Size([-1, 256, 4, 4, 4])
        net = self.deconv2(net)
        # print(net.size())  # torch.Size([-1, 128, 8, 8, 8])
        feat = self.deconv3(net)
        # print(net.size())  # torch.Size([-1, 64, 16, 16, 16])
        net = self.deconv4(feat)
        # print(net.size())  # torch.Size([-1, 32, 32, 32, 32])
        net = self.deconv5(net)
        # print(net.size())  # torch.Size([-1, 16, 64, 64, 64])
        out = self.deconv6(net)
        # print(out.size())  # torch.Size([-1, 1, 64, 64, 64])
        out_sigmoid = self.sigmoid(out)

        return feat, out_sigmoid

class PointDecoder(nn.Module):
    ''' Sample Point Decoder (PointSetGenerationV1).
        Args:
            z_dim (int): input feature z dimension
    '''
    def __init__(self, z_dim=256, npc=512):
        super(PointDecoder, self).__init__()

        self.z_dim = z_dim
        self.pc_num = npc

        self.fc1 = nn.Conv1d(self.z_dim, 512, 1)
        self.fc2 = nn.Conv1d(512, 1024, 1)
        self.fc3 = nn.Conv1d(1024, 1024, 1)
        self.fc4 = nn.Conv1d(1024, self.pc_num*3, 1)

        self.relu = F.relu

    def forward(self, z):
        z = z.contiguous().view(-1, self.z_dim, 1)
        net = self.relu(self.fc1(z))
        net = self.relu(self.fc2(net))
        net = self.relu(self.fc3(net))
        out = torch.tanh(self.fc4(net))/2.    # output val: (-0.5, 0.5)
        out = out.contiguous().view(-1, self.pc_num, 3)

        return out

class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super(DecoderCBatchNorm, self).__init__()

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)
        # batch_size, D, T = p.size()
        net = self.fc_p(p)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


"""
##########################################define network##########################################
"""

class VoxelNetwork64(nn.Module):
    def __init__(self, z_dim=256, gf_dim=256):
        super(VoxelNetwork64, self).__init__()
        self.z_dim = z_dim
        self.encoder = resnet.Resnet18(self.z_dim)
        self.decoder = VoxelDecoder64(z_dim=self.z_dim, gf_dim=gf_dim)

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        y, y_sigmoid = self.decoder(x)

        return y, y_sigmoid

    def provide_middle_feature(self, img):
        z = self.encoder(img)
        return self.decoder.provide_middle_feature(z)


class VoxelRefineNetwork(nn.Module):
    def __init__(self, channel_dim=64, vox_size=64):
        super(VoxelRefineNetwork, self).__init__()
        self.channel_dim = channel_dim

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, self.channel_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.channel_dim),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.channel_dim, self.channel_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.channel_dim//2),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.channel_dim // 2, self.channel_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.channel_dim // 4),
            nn.LeakyReLU(0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(self.channel_dim // 4, 1, kernel_size=1, stride=1, padding=0)
        )

        self.sigmoid = nn.Sigmoid()
        self.vox_size = vox_size

    def voxel_updater(self, vox_sigmoid, p, p_occ_sigmoid, occ_lambda=0.9):     # vox_shape: [bs, 1, vs, vs, vs]
        # pick volumetric positions/coordinates corresponding to points
        p = (p+0.5)*(self.vox_size-0.001)    # exclude the 1*self.vox_size
        p_coord = p.type(torch.LongTensor)  # shape: [bs, n, 3]
        d = p.floor() + 0.5 - p   # shape: [bs, n, 3]
        w = occ_lambda - torch.sum(d*d, dim=-1)  # shape: [bs, n]

        # update occupancy values
        vox_update = vox_sigmoid.clone()
        bs = p_coord.size()[0]
        for i in range(0, bs):
            vox_update[i, 0, p_coord[i, :, 0], p_coord[i, :, 1], p_coord[i, :, 2]] = \
                vox_sigmoid[i, 0, p_coord[i, :, 0], p_coord[i, :, 1], p_coord[i, :, 2]]*(1.-w[i]) + \
                p_occ_sigmoid[i, :]*w[i]

        return vox_update

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        out = self.conv4(net)

        out_sigmoid = self.sigmoid(out)

        return out, out_sigmoid

    def predict_with_local_feature(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        voxel = self.conv4(feat)

        voxel = self.sigmoid(voxel)

        return feat, voxel

class VoxelSuperResNetwork_16_64(nn.Module):    # generate voxel from 16*16*16 to 64*64*64
    def __init__(self, input_dim=1, gf_dim=128):
        super(VoxelSuperResNetwork_16_64, self).__init__()
        self.input_dim = input_dim
        self.gf_dim = gf_dim

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(self.input_dim, self.gf_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.gf_dim),
            nn.LeakyReLU(0.2)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim, self.gf_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.gf_dim//2),
            nn.LeakyReLU(0.2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim//2, self.gf_dim//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.gf_dim//4),
            nn.LeakyReLU(0.2)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(self.gf_dim//4, 1, kernel_size=1, stride=1, padding=0)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        net = self.deconv1(x)
        # print(net.size())  # torch.Size([-1, gf_dim, 32, 32, 32])
        net = self.deconv2(net)
        # print(net.size())  # torch.Size([-1, gf_dim//2, 64, 64, 64])
        net = self.deconv3(net)
        # print(out.size())  # torch.Size([-1, gf_dim//4, 64, 64, 64])
        out = self.deconv4(net)
        # print(out.size())  # torch.Size([-1, 1, 64, 64, 64])
        out_sigmoid = self.sigmoid(out)

        return out, out_sigmoid


class VoxelPatchSuperResNetwork(nn.Module):
    def __init__(self, input_dim=1, gf_dim=64, w_update=0.7, begin_idx=None):
        super(VoxelPatchSuperResNetwork, self).__init__()
        self.input_dim = input_dim
        self.gf_dim = gf_dim
        self.w_update = w_update
        if begin_idx is not None:
            self.begin_idx = begin_idx
        else:
            begin_idx = []
            for i in range(0, 64, 16):
                for j in range(0, 64, 16):
                    for k in range(0, 64, 16):
                        begin_idx.append([i, j, k])
            self.begin_idx = np.array(begin_idx, dtype=np.int32) * 4

        self.generator = VoxelSuperResNetwork_16_64(input_dim=self.input_dim, gf_dim=self.gf_dim)
        self.refiner = VoxelRefineNetwork(channel_dim=gf_dim, vox_size=64)

    def forward(self, vox_patch, patch_idx, pc, pc_occ, b_return_patch_pc_idx=False):
        patch_size = vox_patch.size()  # [opt.patchNum, 1, 16, 16, 16]

        vox_pre, vox_pre_sigmoid = self.generator(vox_patch)

        pc = pc.contiguous().view(-1, 3) + 0.5    # pc: [0, 1]
        pc_occ = pc.contiguous().view(-1, 1)
        pc_256 = np.array(pc.cpu().data.squeeze().numpy() * 256., dtype=np.int32)
        vox_pre_update = vox_pre_sigmoid.clone()
        pc_idx_patch = []
        for i in range(0, patch_size[0]):
            idx = self.begin_idx[patch_idx[i]]
            pc_idx = np.where((pc_256[:, 0] >= idx[0]) & (pc_256[:, 0] < idx[0] + 64) &
                              (pc_256[:, 1] >= idx[1]) & (pc_256[:, 1] < idx[1] + 64) &
                              (pc_256[:, 2] >= idx[2]) & (pc_256[:, 2] < idx[2] + 64))

            if len(pc_idx) > 0 and len(pc_idx[0]) > 0:
                pc_patch = (pc[pc_idx] - (torch.from_numpy(np.array(idx, dtype=np.float32))/256.).cuda())/0.25 * 63.999
                pc_patch_coord = pc_patch.type(torch.LongTensor)    # shape: [n, 3]
                pc_occ_patch = pc_occ[pc_idx]                       # shape: [n, 1]
                vox_pre_update[i, 0, pc_patch_coord[:, 0], pc_patch_coord[:, 1], pc_patch_coord[:, 2]] = \
                    vox_pre_sigmoid[i, 0, pc_patch_coord[:, 0], pc_patch_coord[:, 1], pc_patch_coord[:, 2]] * \
                    (1.-self.w_update) + pc_occ_patch[:, 0] * self.w_update

                pc_idx_patch.append(pc_patch_coord)
            else:
                pc_idx_patch.append([])

        vox_ref, vox_ref_sigmoid = self.refiner(vox_pre_update)

        if b_return_patch_pc_idx:
            return vox_pre, vox_pre_sigmoid, vox_ref, vox_ref_sigmoid, pc_idx_patch
        else:
            return vox_pre, vox_pre_sigmoid, vox_ref, vox_ref_sigmoid
    

class PointSetGenerationNetwork(nn.Module):
    def __init__(self, z_dim=256, n_pc=512):
        super(PointSetGenerationNetwork, self).__init__()
        self.z_dim = z_dim
        self.n_pc = n_pc
        self.encoder = resnet.Resnet18(self.z_dim)
        self.decoder = PointDecoder(z_dim=self.z_dim, npc=self.n_pc)

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        y = self.decoder(x)

        return y


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
    '''

    def __init__(self, z_dim=256):
        super(OccupancyNetwork, self).__init__()
        self.encoder = resnet.Resnet18(z_dim)
        self.decoder = DecoderCBatchNorm(dim=3, c_dim=256, hidden_size=256)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img, p):
        ''' Performs a forward pass through the network.

        Args:
            img (tensor): input image
            p (tensor): sampled points
        '''
        img = img[:, :3, :, :].contiguous()

        c = self.encoder(img)

        logits = self.decoder(p, c)
        p_occ = dist.Bernoulli(logits=logits).logits
        p_occ_sigmoid = self.sigmoid(p_occ)

        return p_occ, p_occ_sigmoid

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def predict(self, img, pts):
        c = self.encoder(img)

        # print('p_shape', p.size())
        pts_occ = self.decode(pts, c).logits
        pts_occ_sigmoid = self.sigmoid(pts_occ)

        return pts_occ_sigmoid


class VoxelPatchSuperResNetwork_Image(nn.Module):
    def __init__(self, z_dim=256, input_vox_dim=1, gf_dim=128, image_encoder=None):
        super(VoxelPatchSuperResNetwork_Image, self).__init__()
        self.z_dim = z_dim
        self.input_vox_dim = input_vox_dim
        self.gf_dim = gf_dim
        self.encoder = image_encoder
        if image_encoder is None:
            self.encoder = resnet.Resnet18(self.z_dim)

        self.decoder = VoxelSuperResNetwork_16_64(input_dim=1+self.z_dim, gf_dim=self.gf_dim)   # 2: 1+1, include id index

    def forward(self, img, vox_patch):
        img = img[:, :3, :, :].contiguous()
        z = self.encoder(img)
        patch_size = vox_patch.size()   # [opt.patchNum, 2, 16, 16, 16]
        z = z.contiguous().view((1, self.z_dim, 1, 1, 1))
        z = z.repeat(patch_size[0], 1, patch_size[2], patch_size[3], patch_size[4])

        vox, vox_sigmoid = self.decoder(torch.cat((vox_patch, z), 1))

        return vox, vox_sigmoid


class SVRNetworkOccGuide(nn.Module):        # occupancy network without kl-divergence
    def __init__(self, z_dim_vox=128, gf_dim_vox=128, z_dim_pc=256, z_dim_occ=256, n_pc=1024):
        super(SVRNetworkOccGuide, self).__init__()
        self.z_dim_vox = z_dim_vox
        self.gf_dim_vox = gf_dim_vox
        self.z_dim_pc = z_dim_pc
        self.z_dim_occ = z_dim_occ
        self.n_pc = n_pc

        self.img2vox = VoxelNetwork64(z_dim=self.z_dim_vox, gf_dim=self.gf_dim_vox)
        self.img2pc = PointSetGenerationNetwork(z_dim=self.z_dim_pc, n_pc=self.n_pc)
        self.voxrefine = VoxelRefineNetwork()
        self.occnet = OccupancyNetwork(z_dim=self.z_dim_occ)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img, occ_pc):
        # img2vox
        vox_pre, vox_pre_sigmoid = self.img2vox(img)

        # img2pc
        pc_pre = self.img2pc(img)   # here pc_pre value is belong to [-0.5, 0.5], while we use occnet data which mesh vertice value is belong to 10/9*[-0.5, 0.5]

        c = self.occnet.encoder(img)
        # obtain occupancy value corresponding to supervised points
        logits1 = self.occnet.decoder(occ_pc, c)
        occ_pre = dist.Bernoulli(logits=logits1).logits
        occ_pre_sigmoid = self.sigmoid(occ_pre)

        # obtain occupancy value corresponding to predicted points for voxel refinement
        logits = self.occnet.decoder(10./9.*pc_pre, c)
        pc_occ_pre = dist.Bernoulli(logits=logits).logits
        pc_occ_pre_sigmoid = self.sigmoid(pc_occ_pre)  # shape: [bs, n])

        # voxel refinement
        vox_update = self.voxrefine.voxel_updater(vox_pre_sigmoid, pc_pre, pc_occ_pre_sigmoid)
        vox_refine_pre, vox_refine_pre_sigmoid = self.voxrefine(vox_update)

        return vox_pre, vox_pre_sigmoid, pc_pre, occ_pre, occ_pre_sigmoid, vox_refine_pre, vox_refine_pre_sigmoid

    def predict(self, img, b_return_mid=False, b_return_pc=False):
        # img2vox
        vox_pre, vox_pre_sigmoid = self.img2vox(img)

        # img2pc
        pc_pre = self.img2pc(img)   # here pc_pre value is belong to [-0.5, 0.5], while we use occnet data which mesh vertice value is belong to 10/9*[-0.5, 0.5]

        c = self.occnet.encoder(img)
        # obtain occupancy value corresponding to predicted points for voxel refinement
        logits = self.occnet.decoder(10./9.*pc_pre, c)
        pc_occ_pre = dist.Bernoulli(logits=logits).logits
        pc_occ_pre_sigmoid = self.sigmoid(pc_occ_pre)  # shape: [bs, n])

        # voxel refinement
        vox_update = self.voxrefine.voxel_updater(vox_pre_sigmoid, pc_pre, pc_occ_pre_sigmoid)
        vox_refine_pre, vox_refine_pre_sigmoid = self.voxrefine(vox_update)

        if b_return_mid:
            return vox_pre, vox_pre_sigmoid, pc_pre, vox_refine_pre, vox_refine_pre_sigmoid
        if b_return_pc:
            return vox_refine_pre, vox_refine_pre_sigmoid, pc_pre
        else:
            return vox_refine_pre, vox_refine_pre_sigmoid


    def predict_highres_input(self, img, b_pc_guide=False):
        # img2vox
        feat_low, vox_pre_sigmoid = self.img2vox.decoder.predict_with_middle_feature(self.img2vox.encoder(img))

        # img2pc
        pc_pre = self.img2pc(img)

        c = self.occnet.encoder(img)
        # obtain occupancy value corresponding to predicted points for voxel refinement
        logits = self.occnet.decoder(pc_pre, c)
        pc_occ_pre = dist.Bernoulli(logits=logits).logits
        pc_occ_pre_sigmoid = self.sigmoid(pc_occ_pre)  # shape: [bs, n])

        # voxel refinement
        vox_update = self.voxrefine.voxel_updater(vox_pre_sigmoid, pc_pre, pc_occ_pre_sigmoid)
        feat_high, vox_ref_sigmoid = self.voxrefine.predict_with_local_feature(vox_update)

        if b_pc_guide is not True:
            return torch.cat([vox_ref_sigmoid, feat_high], 1), feat_low
        else:
            return torch.cat([vox_ref_sigmoid, feat_high], 1), feat_low, pc_pre, pc_occ_pre_sigmoid
