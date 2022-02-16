from __future__ import print_function
import sys
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import time,datetime
from tensorboardX import SummaryWriter
from dataset import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default='./checkpoint/', help='model path')
parser.add_argument('--log', type=str, default='./log_vipnet_finetune/', help='log path')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'
# '04090263'(rifle) need less training epochs(10)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Creat train/val dataloader
dataset = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                   vox_root=os.path.join(opt.dataRoot, 'vox64'),
                   pc_root=os.path.join(opt.dataRoot, 'pc'),
                   pc_occ_root=os.path.join(opt.dataRoot, 'pc_occ'),
                   filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                   cat=opt.cat, mode='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataset_val = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                       vox_root=os.path.join(opt.dataRoot, 'vox64'),
                       pc_root=os.path.join(opt.dataRoot, 'pc'),
                       pc_occ_root=os.path.join(opt.dataRoot, 'pc_occ'),
                       filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                       cat=opt.cat, mode='val')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
len_dataset_val = len(dataset_val)
print('training set num', len_dataset)
print('validation set num', len_dataset_val)

# create path
model_path = os.path.join(opt.model, opt.cat)
log_path = os.path.join(opt.log, opt.cat)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = SummaryWriter(log_path)

cudnn.benchmark = True

# Create network
network = SVRNetworkOccGuide()
network.cuda()
# load pre-trained models
network.img2vox.load_state_dict(torch.load(os.path.join(model_path, 'img2vox64.pt')))
print('loaded pre-trained img2vox model.')
network.img2pc.load_state_dict(torch.load(os.path.join(model_path, 'img2pc.pt')))
print('loaded pre-trained img2pc model.')
network.occnet.load_state_dict(torch.load(os.path.join(opt.model, 'occnet.pt')))
print('loaded pre-trained occnet model.')
network.voxrefine.load_state_dict(torch.load(os.path.join(model_path, 'vox64_refine.pt')))
print('loaded pre-trained voxrefine model.')

# Create Loss Module
sys.path.append('./utils/chamferdistance/')
import dist_chamfer as ext
dist_chamfer = ext.chamferDist()

criterion_bce_init = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([5.]).cuda())
criterion_bce_ref = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([2.]).cuda())
criterion_bce = nn.BCEWithLogitsLoss()

# Create optimizer
# optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
optimizer = optim.Adam(network.parameters(), lr=opt.lr)

it_step = 0
for epoch in range(1, opt.nepoch+1):
    # TRAIN MODE
    network.train()
    for i, data in enumerate(dataloader, 0):
        it_step += 1

        optimizer.zero_grad()
        img, vox_gt, pc_gt, occ_data, name, view_id = data
        img = img.cuda()
        vox_gt = vox_gt.cuda()
        pc_gt = pc_gt.cuda()
        occ_pc = occ_data[None].cuda()
        occ_gt = occ_data['occ'].cuda()

        vox_pre, vox_pre_sigmoid, pc_pre, occ_pre, _, vox_refine_pre, vox_refine_pre_sigmoid = network(img, occ_pc)

        # loss
        vox_init_loss = criterion_bce_init(vox_pre, vox_gt)
        dist1, dist2 = dist_chamfer(pc_gt, pc_pre)
        cd_loss = torch.mean(dist1) + torch.mean(dist2)
        occ_loss = criterion_bce(occ_pre, occ_gt)
        vox_ref_loss = criterion_bce_ref(vox_refine_pre, vox_gt)

        loss = vox_init_loss + 0.5 * cd_loss + occ_loss + 3 * vox_ref_loss

        loss.backward()
        optimizer.step()

        if it_step % 10 == 0:
            # logs
            logger.add_scalar('train/vox_init_loss', vox_init_loss, it_step)
            logger.add_scalar('train/cd_loss', cd_loss, it_step)
            logger.add_scalar('train/occ_loss', occ_loss, it_step)
            logger.add_scalar('train/vox_ref_loss', vox_ref_loss, it_step)
            logger.add_scalar('train/total_loss', loss, it_step)

            print('[%d: %d/%d] train vox_init_loss: %f, cd_loss: %f, vox_ref_loss: %f, occ_loss: %f, total_loss: %f' %
                  (epoch, i, len_dataset / opt.batchSize, vox_init_loss.item(), cd_loss.item(), vox_ref_loss.item(),
                   occ_loss.item(), loss.item()))

    # #VALIDATION
    # network.eval()
    # with torch.no_grad():
    #     for i, data in enumerate(dataloader_val, 0):
    #         img, vox, pc, occ_data, name, view_id = data
    #         img = img.cuda()
    #         vox = vox.cuda()
    #         pc = pc.cuda()
    #         occ_p = occ_data[None].cuda()
    #         occ_val = occ_data['occ'].cuda()
    #
    #         vox_pre, vox_pre_sigmoid, p_pre, _, _, occ_val_pre, _, vox_refine_pre, vox_refine_pre_sigmoid = network(img, occ_p)
    #         vox_init_loss = criterion_bce(vox_pre, vox)
    #         dist1, dist2 = dist_chamfer(pc, p_pre)
    #         cd_loss = (torch.mean(dist1) + torch.mean(dist2)) * 10.
    #         occ_loss = criterion_bce(occ_val_pre, occ_val)
    #         vox_ref_loss = criterion_bce(vox_refine_pre, vox)
    #
    #         loss = vox_init_loss + cd_loss + 2 * occ_loss + 2 * vox_ref_loss
    #
    #         logger.add_scalar('val/vox_init_loss', vox_init_loss, it_step)
    #         logger.add_scalar('val/vox_ref_loss', vox_ref_loss, it_step)
    #         logger.add_scalar('val/cd_loss', cd_loss, it_step)
    #         logger.add_scalar('val/occ_loss', occ_loss, it_step)
    #         logger.add_scalar('val/total_loss', loss, it_step)
    #
    #         print('[%d: %d/%d] val vox_init_loss: %f, vox_ref_loss: %f, cd_loss: %f, occ_loss: %f, total_loss: %f' %
    #               (epoch, i, len_dataset / opt.batchSize, vox_init_loss.item(), vox_ref_loss.item(),
    #                cd_loss.item(), occ_loss.item(), loss.item()))
    #         break   # just validate one batch

    torch.save(network.state_dict(), os.path.join(model_path, 'vipnet_finetune.pt'))
    print('save model succeeded!')

    # if epoch % 5 == 0:
    #     torch.save(network.state_dict(), os.path.join(model_path, 'vipnet_finetune_'+str(epoch)+'.pt'))
    #     print('save model succeeded!')

print('Training done!')
