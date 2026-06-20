import random
import time
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from config import *
from netIMG import *
from netIMG_large import *



# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cudnn.allow_spatial_tf32 = False



opt = get_opt()
logger = log(opt.log_path)
writer = SummaryWriter(opt.log_path)
logger.info(opt)
writer.add_text('settings', str(opt), 0)

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = opt.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if opt.architecture == "ModelImg1":
    model = ModelImg1()
elif opt.architecture == "ModelImg2":
    model = ModelImg2()
elif opt.architecture == "ModelImg3":
    model = ModelImg3(opt)
elif opt.architecture == "ModelImgLarge1":
    model = ModelImgLarge1()
elif opt.architecture == "ModelImgLarge2":
    model = ModelImgLarge2()
elif opt.architecture == "ModelImgHybrid":
    model = ModelImgHybrid()
else:
    print(f"error! Unknown architecture: {opt.architecture}")
    
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

# -------------------------------------- learning setttings --------------------------------------------------------------
if bool(opt.use_SGD):
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
else:
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.0005)
if opt.decay_mode == 1:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch, eta_min=opt.min_lr)
elif opt.decay_mode == 2:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_epoch, gamma=opt.decay_rate)
elif opt.decay_mode == 3:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01**(1 / opt.nepoch))
# ------------------------------------------------------------------------------------------------------------------------

data = torch.load(opt.data_path)
pairs = torch.load(opt.pairs_path)
cat_keys = ["chair", "table", "mug", "lamp"]
pairs_train = []
pairs_test = []
pairs_test_split = {"chair": [], "table": [], "mug": [], "lamp": []}
for key in cat_keys:
    num = len(pairs[key])
    idxs = np.arange(num)
    np.random.shuffle(idxs)
    for i in range(num):
        if i < num * 0.8:
            pairs_train.append(pairs[key][idxs[i]])
        else:
            pairs_test_split[key].append(pairs[key][idxs[i]])
            pairs_test.append(pairs[key][idxs[i]])

pairs_idx = torch.tensor(pairs_test).to(device)
pairs_split_idx = {}
for key in pairs_test_split:
    pairs_split_idx[key] = torch.tensor(pairs_test_split[key])[:, :2].long().to(device)

trainset = DatasetPair(pairs_train, data)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, drop_last=False)
testset = DatasetSingle(data)
testLoader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize * 2, shuffle=False, drop_last=False)
logger.info('train set length: %d' % len(trainset))


start = time.time()

for epoch in range(0, opt.nepoch):
    logger.info('Epoch %d:' % (epoch + 1))
    model.train()
    train_loss_bt = []
    train_acc_bt = []
    for batch_i, (shape1, shape2, margin_w) in enumerate(trainLoader):
        optimizer.zero_grad()
        shape1, shape2, margin_w = shape1.to(device), shape2.to(device), margin_w.to(device)
        current_bs = shape1.shape[0]
        s_global = model(torch.cat((shape1, shape2)))
        loss = weighted_loss(s_global[0:current_bs, :], s_global[current_bs:, :], margin_w, opt.margin)
        accuracy = torch.mean((s_global[0:current_bs, :] - s_global[current_bs:, :]).gt(0).float())
        train_loss_bt.append(loss.item())
        train_acc_bt.append(accuracy.item())
        loss.backward()
        optimizer.step()
    train_loss = np.mean(train_loss_bt)
    train_acc = np.mean(train_acc_bt)
    scheduler.step()

    model.eval()
    cache_scores = []
    val_acc_split = {}
    with torch.no_grad():
        for batch_i, shapes in enumerate(testLoader):
            shapes = shapes.to(device)
            cache_scores.append(model(shapes))
    cache_scores = torch.cat(cache_scores, dim=0)
    for key in pairs_split_idx:
        s1 = cache_scores[pairs_split_idx[key][:, 0], :]
        s2 = cache_scores[pairs_split_idx[key][:, 1], :]
        val_acc_split[key] = torch.mean((s1 - s2).gt(0).float()).item()

    s1_all = cache_scores[pairs_idx[:, 0].long(), :]
    s2_all = cache_scores[pairs_idx[:, 1].long(), :]
    val_loss = weighted_loss(s1_all, s2_all, pairs_idx[:, 2], opt.margin).item()
    val_acc = torch.mean((s1_all - s2_all).gt(0).float()).item()

    writer.add_scalar('Loss/train', train_loss, epoch + 1)
    writer.add_scalar('Loss/val', val_loss, epoch + 1)
    writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
    writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

    logger.info('Epoch: %d, Time: %.3f, Train_loss: %.6f, Train_acc: %.3f, Val_loss: %.6f, chair: %.3f, table: %.3f, mug: %.3f, lamp: %.3f' % (epoch + 1, time.time() - start, train_loss, train_acc, val_loss, val_acc_split["chair"], val_acc_split["table"], val_acc_split["mug"], val_acc_split["lamp"]))

    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'score_global': cache_scores,
        'config': opt
    }
    if (epoch + 1) % opt.save_rate == 0:
        torch.save(state, os.path.join(opt.log_path, 'paras_%d.pth' % (epoch + 1)))

    start = time.time()

writer.close()
