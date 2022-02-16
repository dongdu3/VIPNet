from __future__ import print_function
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import time,datetime
from tensorboardX import SummaryWriter
from dataset_patch_l2h import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--patchNum', type=int, default=48, help='input patch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--vipnet', type=str, default='vipnet_finetune.pt', help='model path')
parser.add_argument('--log', type=str, default='log_patch_l2h', help='log path')
parser.add_argument('--nepoch', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# create path
model_path = os.path.join(opt.model, opt.cat)
log_path = os.path.join(opt.log, opt.cat)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = SummaryWriter(log_path)

# Creat train/val dataloader
dataset = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                   vox256_root=os.path.join(opt.dataRoot, 'vox256'),
                   filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                   cat=opt.cat, mode='train', patch_num=opt.patchNum, view_pick=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataset_val = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                       vox256_root=os.path.join(opt.dataRoot, 'vox256'),
                       filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                       cat=opt.cat, mode='val', patch_num=opt.patchNum, view_pick=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
len_dataset_val = len(dataset_val)
print('training set num', len_dataset)
print('validation set num', len_dataset_val)

# Create network
vipnet = SVRNetworkOccGuide()
vipnet.cuda()
vipnet.load_state_dict(torch.load(os.path.join(model_path, opt.vipnet)))
vipnet.eval()
print("Previous vipnet weights loaded...")

vox_dim = 1
feat_high_dim = 16
feat_low_dim = 32
input_dim = vox_dim + feat_high_dim + feat_low_dim
network_patch_l2h = VoxelSuperResNetwork_16_64(input_dim=input_dim, gf_dim=64)
network_patch_l2h.cuda()
print('Defined patch_l2h network.')

# Create Loss Module
criterion = nn.BCEWithLogitsLoss()

# Create optimizer, just optimize the parameters of decoder
optimizer = optim.Adam(network_patch_l2h.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

it_step = 0
for epoch in range(1, opt.nepoch + 1):
    # TRAIN MODE
    network_patch_l2h.train()
    for it, data in enumerate(dataloader, 0):
        it_step += 1

        optimizer.zero_grad()
        img, vox64, idx, name, view_id = data
        idx = idx[0]
        img = img.cuda()
        vox64 = vox64.cuda()
        vox64 = vox64.contiguous().view(opt.patchNum, 1, 64, 64, 64)

        vox_ref_cat_feat_high, feat_low = vipnet.predict_highres_input(img)

        vox_in = torch.zeros(opt.patchNum, vox_dim+feat_high_dim, 16, 16, 16).type(torch.FloatTensor).cuda()
        for i in range(0, opt.patchNum):    # here batch_size = 1
            vox_in[i] = vox_ref_cat_feat_high[0, :, dataset.begin_idx64[idx[i]][0]: dataset.begin_idx64[idx[i]][0] + 16,
                            dataset.begin_idx64[idx[i]][1]: dataset.begin_idx64[idx[i]][1] + 16,
                            dataset.begin_idx64[idx[i]][2]: dataset.begin_idx64[idx[i]][2] + 16]

        del img, vox_ref_cat_feat_high
        # torch.cuda.empty_cache()

        feat_low = feat_low.repeat(opt.patchNum, 1, 1, 1, 1)
        vox_in_cat = torch.cat([vox_in, feat_low], 1)

        del feat_low, vox_in
        # torch.cuda.empty_cache()

        vox_pre, vox_pre_sigmoid = network_patch_l2h(vox_in_cat)
        loss = criterion(vox_pre, vox64)

        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            logger.add_scalar('train/loss', loss, it_step)
            print('[%d: %d/%d] train loss:  %f' % (epoch, it, len_dataset / opt.batchSize, loss.item()))

        # del img, vox64, vox_feat, vox_ref, vox_in, vox_in_cat, vox_pre, vox_pre_sigmoid
        # torch.cuda.empty_cache()

    # #VALIDATION
    # network_patch_l2h_img.eval()
    # with torch.no_grad():
    #     for it, data in enumerate(dataloader_val, 0):
    #         vox16, vox64, name = data
    #         vox16 = vox16.cuda()
    #         vox16 = vox16.contiguous().view(opt.patchNum, 1, 16, 16, 16)
    #         vox64 = vox64.cuda()
    #         vox64 = vox64.contiguous().view(opt.patchNum, 1, 64, 64, 64)
    #
    #         vox_pre, vox_pre_sigmoid = network_patch_l2h_img(vox16)
    #         loss = criterion(vox_pre, vox64)
    #         logger.add_scalar('val/loss', loss, it_step)
    #
    #         print('[%d: %d/%d] val loss: %f' % (epoch, it, len_dataset_val / opt.batchSize, loss.item()))
    #         break   # just validate one batch

    torch.save(network_patch_l2h.state_dict(), os.path.join(model_path, 'patch_l2h.pt'))
    print('save model')
print('Training done!')
