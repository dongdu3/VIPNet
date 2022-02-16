from __future__ import print_function
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import time,datetime
from tensorboardX import SummaryWriter
from dataset_img2vox import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--log', type=str, default='log_img2vox', help='log path')
parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate.')
# parser.add_argument('--lrDecay', type=float, default=0.1)
# parser.add_argument('--lrStep', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

# Creat train/val dataloader
dataset = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                   vox_root=os.path.join(opt.dataRoot, 'vox64'),
                   filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                   cat=opt.cat, mode='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataset_val = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                       vox_root=os.path.join(opt.dataRoot, 'vox64'),
                       filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                       cat=opt.cat, mode='val')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
len_dataset_val = len(dataset_val)
print('training set num', len_dataset)
print('validation set num', len_dataset_val)

# Create network
network_img2vox = VoxelNetwork64(z_dim=128, gf_dim=128)
network_img2vox.cuda()

# Create Loss Module
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([5.]).cuda())

# Create optimizer
optimizer = optim.Adam(network_img2vox.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

# create path
model_path = os.path.join(opt.model, opt.cat)
log_path = os.path.join(opt.log, opt.cat)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = SummaryWriter(log_path)

it_step = 0
min_loss = 1e8
best_epoch = 0
for epoch in range(1, opt.nepoch+1):
    # TRAIN MODE
    network_img2vox.train()
    for i, data in enumerate(dataloader, 0):
        it_step += 1

        optimizer.zero_grad()
        img, vox, name, view_id = data
        img = img.cuda()
        vox = vox.cuda()

        vox_pre, vox_pre_sigmoid = network_img2vox(img)

        loss = criterion_bce(vox_pre, vox)

        if it_step % 10 == 0:
            logger.add_scalar('train/loss', loss, it_step)
            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / opt.batchSize, loss.item()))

        loss.backward()
        optimizer.step()

    #VALIDATION for every epoch
    network_img2vox.eval()

    loss_val = 0
    loss_n = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader_val, 0):
            img, vox, name, view_id = data
            img = img.cuda()
            vox = vox.cuda()

            vox_pre, vox_pre_sigmoid = network_img2vox(img)

            loss = criterion_bce(vox_pre, vox)

            loss_val += loss.item()
            loss_n += 1

            if i % 10 == 0:
                print('[%d: %d/%d] val loss: %f' % (epoch, i, len_dataset_val / opt.batchSize, loss.item()))

    loss_val /= loss_n
    logger.add_scalar('val/loss', loss_val, it_step)

    if loss_val < min_loss + 1e-4:
        min_loss = loss_val
        best_epoch = epoch
        print('Best epoch is:', best_epoch)

    torch.save(network_img2vox.state_dict(), os.path.join(model_path, 'img2vox64.pt'))
    print('save model')
    # if epoch % 10 == 0:
    #     torch.save(network_img2vox.state_dict(), '%s/img2vox64_%s.pt' % (model_path, str(epoch)))
    #     print('save model')

print('Training done!')
print('Best epoch is:', best_epoch)