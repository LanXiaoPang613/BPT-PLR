from __future__ import print_function
import os
import matplotlib as mpl

# if os.environ.get('DISPLAY', '') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
import torchvision.models as models
from VGG_animal import vgg19_bn
from sklearn.mixture import GaussianMixture
import dataloader_animal10N as dataloader
import pdb
import io
import PIL
from torchvision import transforms
import seaborn as sns
import sklearn.metrics as metrics
import pickle
import json
import pandas as pd
import time
from pathlib import Path
from utils_plot import plot_guess_view, plot_histogram_loss_pred, \
    plot_model_view_histogram_loss, plot_model_view_histogram_pred, plot_tpr_fpr
import robust_loss, Contrastive_loss

sns.set()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='animal10N')
parser.add_argument('--data_path', default='C:/Users/Administrator/Desktop/DatasetAll/Animal-10N', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--vgg_en', default=True, type=bool)
parser.add_argument('--num_class', default=10, type=int)

parser.add_argument('--r', default=0., type=float, help='noise ratio')
parser.add_argument('--noise_mode', default='none')

parser.add_argument('--resume', default=False , type=bool, help='Resume from chekpoint')
parser.add_argument('--num_clean', default=5, type=int)
parser.add_argument('--run', default=0, type=int)
parser.add_argument('--yespenalty', default=1, type=int)
parser.add_argument('--dataset', default='animal10N', type=str)


args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

flat = False
info_nce_loss = Contrastive_loss.InfoNCELoss(temperature=0.1,
                                batch_size=args.batch_size * 2,
                                flat=flat,
                                n_views=2)
plr_loss = Contrastive_loss.PLRLoss(flat=flat)
device = torch.device("cuda")

# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, op, savelog=False):
    net.train()
    net2.eval()  # fix one network and train the other

    train_loss = train_loss_lx = train_loss_u = train_loss_penalty = 0

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    max_iters = ((len(labeled_trainloader.dataset) + len(unlabeled_trainloader.dataset)) // args.batch_size) + 1

    cont_iters = 0
    topK = 3
    if epoch >= 40:
        topK = 2
    elif epoch >= 70:
        topK = 1

    while (cont_iters < max_iters):  # longmix
        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, indices_x) in enumerate(
                labeled_trainloader):
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u, indices_u = unlabeled_train_iter.__next__()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u, indices_u = unlabeled_train_iter.__next__()
            batch_size = inputs_x.size(0)

            if inputs_u.size(0) <= 1 or batch_size <= 1:
                # Expected more than 1 value per channel when training, got input size torch.Size([1, 128])
                continue

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)

                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)
                      + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                ptu = pu ** (1 / args.T)  # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)

                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / args.T)  # temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            inputs = torch.cat([mixed_input, all_inputs], dim=0)
            logits, features = net(inputs, forward_pass='cls_proj')
            logits_x_u = logits[:mixed_input.shape[0]]
            logits_x = logits_x_u[:batch_size * 2]
            logits_u = logits_x_u[batch_size * 2:]
            # logits_x, logits_u = torch.chunk(logits[:mixed_input.shape[0]], 2, dim=0)

            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u,
                                     mixed_target[batch_size * 2:], epoch + batch_idx / num_iter, warm_up)
            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits[:mixed_input.shape[0]], dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))
            loss = Lx + lamb * Lu + penalty

            indices = torch.cat((indices_x, indices_u), dim=0)
            eval_outputs = op[indices, :]
            labels_u = labels_u.cuda()
            labels_x = torch.argmax(labels_x, dim=1)
            labels = torch.cat((labels_x, labels_u), dim=0)
            # device = torch.device("cuda")
            contrastive_mask = Contrastive_loss.build_mask_step(eval_outputs, topK, labels, device)

            fx3, fx4, fu3, fu4 = torch.chunk(features[mixed_input.shape[0]:], 4, dim=0)
            f1, f2 = torch.cat([fx3, fu3], dim=0), torch.cat([fx4, fu4], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_crl = plr_loss(features, mask=contrastive_mask)

            loss += loss_crl

            train_loss += loss
            train_loss_lx += Lx
            train_loss_u += Lu
            train_loss_penalty += penalty

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write(
                '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f, plr_cl loss: %.2f'
                % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                   Lx.item(), Lu.item(), loss_crl.item()))
            sys.stdout.flush()

            cont_iters = cont_iters + 1
            if cont_iters == max_iters:
                break

    if savelog:
        train_loss /= len(labeled_trainloader.dataset)
        train_loss_lx /= len(labeled_trainloader.dataset)
        train_loss_u /= len(labeled_trainloader.dataset)
        train_loss_penalty /= len(labeled_trainloader.dataset)
        # Record training loss from each epoch into the writer
        # writer_tensorboard.add_scalar('Train/Loss', train_loss.item(), epoch)
        # writer_tensorboard.add_scalar('Train/Lx', train_loss_lx.item(), epoch)
        # writer_tensorboard.add_scalar('Train/Lu', train_loss_u.item(), epoch)
        # writer_tensorboard.add_scalar('Train/penalty', train_loss_penalty.item(), epoch)
        # writer_tensorboard.close()


use_robust = False
def warmup(epoch, net, optimizer, dataloader, savelog=False):
    net.train()
    wm_loss = 0
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, inputs_aug1, inputs_aug2, labels, path) in enumerate(dataloader):
        inputs, inputs_aug1, inputs_aug2, labels = inputs.cuda(), inputs_aug1.cuda(), inputs_aug2.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        if use_robust:
            loss = warm_criterion(outputs, labels)
            L = loss
        else:
            loss = CEloss(outputs, labels)
            if args.noise_mode == 'asym':  # Penalize confident prediction for asymmetric noise
                penalty = conf_penalty(outputs)
                L = loss + penalty
            else:
                L = loss
            # penalty = conf_penalty(outputs)
            # L = loss + penalty

        L.backward()

        inputs_crl = torch.cat([inputs_aug1, inputs_aug2], dim=0)
        features = net(inputs_crl, forward_pass='proj')
        logits, labels = info_nce_loss(features)
        v = torch.logsumexp(logits, dim=1, keepdim=True)
        # dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).to(devide), logits], 1)
        # loss_crl = torch.exp(v - v.detach()).mean() -1 + CEloss(dummy_logits, labels)
        loss_crl = torch.exp(v - v.detach()).mean()
        # 下面代码证明了确实有梯度值
        # for name, parms in net.projector.named_parameters():
        #         #     print(f'name={name}\tparam_grad={parms.grad}\n')
        #         #     break
        #         # loss_crl.backward()
        #         # for name, parms in net.projector.named_parameters():
        #         #     print(f'after back, name={name}\tparam_grad={parms.grad}\n')
        #         #     break

        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


best_acc = 0.
def test(epoch, net1, net2):
    global best_acc
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            test_loss += CEloss(outputs1, targets)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    acc_hist.append(acc)
    if best_acc < acc:
        best_acc = acc
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%, Best_acc: %.2f%%\n" % (epoch, acc, best_acc))
    test_log.write('Epoch:%d   Accuracy:%.2f Best_acc: %.2f\n' % (epoch, acc, best_acc))
    test_log.flush()


def eval_train(model, all_loss, all_preds, all_hist, savelog=False):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    losses_proto = torch.zeros(len(eval_loader.dataset))
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), args.num_class)
    pl = torch.zeros(len(eval_loader.dataset), dtype=torch.long, device=device)
    op = torch.zeros(len(eval_loader.dataset), args.num_class, dtype=torch.float, device=device)
    pt = torch.zeros(len(eval_loader.dataset), args.num_class, dtype=torch.float, device=device)
    ft = torch.zeros(len(eval_loader.dataset), 128, dtype=torch.float, device=device)
    eval_loss = train_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, logits_proto, features = model(inputs, forward_pass='all')
            loss = CE(outputs, targets)
            eval_loss += CEloss(outputs, targets)
            loss_proto = CE(logits_proto, targets)

            _, pred = torch.max(outputs.data, -1)
            acc = float((pred == targets.data).sum())
            train_acc += acc
            eval_preds = F.softmax(outputs, -1).cpu().data

            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                losses_proto[index[b]] = loss_proto[b]
                preds[index[b]] = eval_preds[b][targets[b]]
                preds_classes[index[b]] = eval_preds[b]
                pl[index[b]] = pred[b]
                op[index[b]] = outputs[b]
                pt[index[b]] = logits_proto[b]
                ft[index[b]] = features[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses_proto = (losses_proto - losses_proto.min()) / (losses_proto.max() - losses_proto.min())

    # all_loss_proto.append(losses_proto)
    all_loss.append(losses)
    all_preds.append(preds)
    all_hist.append(preds_classes)

    if args.r == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
        input_loss_proto = losses_proto.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)
        input_loss_proto = losses_proto.reshape(-1, 1)

    input_loss = input_loss.cpu().numpy()
    input_loss_proto = input_loss_proto.cpu().numpy()
    gmm_input = np.column_stack((input_loss, input_loss_proto))

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(gmm_input)
    mean_square_dists = np.array([np.sum(np.square(gmm.means_[i])) for i in range(2)])
    argmin, argmax = mean_square_dists.argmin(), mean_square_dists.argmax()
    prob = gmm.predict_proba(gmm_input)
    prob = prob[:, argmin]

    return prob, all_loss, all_preds, all_hist, {'op': op, 'pl': pl, 'pt': pt, 'ft': ft}

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    if args.vgg_en == True:
        model = vgg19_bn(num_classes=args.num_class)
        model.classifier._modules['6'] = nn.Linear(4096, args.num_class)
    else:
        import CNN
        model = CNN.CNN(n_outputs=args.num_class)
        model = model.cuda()
    return model


def guess_unlabeled(net1, net2, unlabeled_trainloader):
    net1.eval()
    net2.eval()

    guessedPred_unlabeled = []
    for batch_idx, (inputs_u, inputs_u2, _, _) in enumerate(unlabeled_trainloader):
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net1(inputs_u)[1]
            outputs_u12 = net1(inputs_u2)[1]
            outputs_u21 = net2(inputs_u)[1]
            outputs_u22 = net2(inputs_u2)[1]

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                        dim=1) + torch.softmax(
                outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            _, guessed_u = torch.max(targets_u, dim=-1)
            guessedPred_unlabeled.append(guessed_u)

    return torch.cat(guessedPred_unlabeled)


def save_models(epoch, net1, optimizer1, net2, optimizer2, save_path):
    state = ({
        'epoch': epoch,
        'state_dict1': net1.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'state_dict2': net2.state_dict(),
        'optimizer2': optimizer2.state_dict()

    })
    state2 = ({'all_loss': all_loss,
               'all_preds': all_preds,
               'hist_preds': hist_preds,
               'all_idx_view_labeled': all_idx_view_labeled,
               'all_idx_view_unlabeled': all_idx_view_unlabeled,
               'all_superclean': all_superclean,
               'acc_hist': acc_hist
               })
    state3 = ({
        'all_superclean': all_superclean
    })

    if epoch % 1 == 0:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        fn3 = os.path.join(save_path, 'hcs_%s_cn%d_run%d.pth.tar' % (
        args.dataset, args.num_clean, args.run))
        torch.save(state2, fn3)

if __name__ == '__main__':

    name_exp = 'plr_single_cn%d' % args.num_clean

    exp_str = '%s_%s_lu_%d' % (args.dataset, name_exp, int(args.lambda_u))
    if args.run > 0:
        exp_str = exp_str + '_run%d' % args.run
    path_exp = './checkpoint/' + exp_str

    path_plot = os.path.join(path_exp, 'plots')

    Path(path_exp).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
    Path(path_plot).mkdir(parents=True, exist_ok=True)

    incomplete = os.path.exists("./checkpoint/%s/model_ckpt.pth.tar" % (exp_str))
    print('Incomplete...', incomplete)

    if incomplete == False:
        stats_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_stats.txt',
                         'w')
        test_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_acc.txt',
                        'w')
        time_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_time.txt',
                        'w')
    else:
        stats_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_stats.txt',
                         'a')
        test_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_acc.txt',
                        'a')
        time_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_time.txt',
                        'a')

    # writer_tensorboard = SummaryWriter('tensor_runs/'+exp_str)

    warm_up = 30 # 10

    loader = dataloader.animal_dataloader(root=args.data_path, batch_size=args.batch_size,
                                         num_workers=0)

    print('| Building net')
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    # scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 230, 1e-4)
    # scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 230, 1e-4)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()

    resume_epoch = 0

    if incomplete == True:
        print('loading Model...\n')
        load_path = 'checkpoint/%s/model_ckpt.pth.tar' % (exp_str)
        ckpt = torch.load(load_path)
        resume_epoch = ckpt['epoch']+1
        print('resume_epoch....', resume_epoch)
        net1.load_state_dict(ckpt['state_dict1'])
        net2.load_state_dict(ckpt['state_dict2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])

        all_superclean = [[], []]
        all_idx_view_labeled = [[], []]
        all_idx_view_unlabeled = [[], []]
        all_preds = [[], []]  # save the history of preds for two networks
        hist_preds = [[], []]
        acc_hist = []
        all_loss = [[], []]

        del ckpt
        # all_idx_view_labeled = ckpt['all_idx_view_labeled']
        # all_idx_view_unlabeled = ckpt['all_idx_view_unlabeled']
        # all_preds = ckpt['all_preds']
        # hist_preds = ckpt['hist_preds']
        # acc_hist = ckpt['acc_hist']
        # all_loss = ckpt['all_loss']
        # all_superclean = [[], []]
        # superclean_path = os.path.join('hcs/', 'hcs_%s_%.2f_%s_cn%d_run%d.pth.tar' % (
        # args.dataset, args.r, args.noise_mode, args.num_clean, args.run))
        # ckpt = torch.load(superclean_path)
        # all_superclean = ckpt['all_superclean']
    else:
        all_superclean = [[], []]
        all_idx_view_labeled = [[], []]
        all_idx_view_unlabeled = [[], []]
        all_preds = [[], []]  # save the history of preds for two networks
        hist_preds = [[], []]
        acc_hist = []
        all_loss = [[], []]  # save the history of losses from two networks

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    # name_exp111 = 'plr_stage1_cn%d' % args.num_clean
    # exp_str111 = '%s_%s_lu_%d' % (args.dataset, name_exp111, int(args.lambda_u))
    # ckpt_sc = torch.load(
    #     './checkpoint/%s/hcs_%s_cn%d_run%d.pth.tar' % (
    #         exp_str111, args.dataset, args.num_clean, args.run))
    # all_superclean = ckpt_sc['all_superclean']

    total_time = 0
    warmup_time = 0

    # maxsize = 0
    # max_i = 0
    # for i in range(1, int(args.num_epochs/2)+1):  # E/2,后半段
    #     size = len(all_superclean[0][-i])
    #     if size > maxsize:
    #         maxsize = size
    #         max_i = i
    #
    # print('max = %d, i=%d' % (maxsize, max_i))
    # idx_superclean = all_superclean[0][-max_i]

    warm_criterion = robust_loss.GCELoss(args.num_class, gpu='0')
    contrastive_criterion = Contrastive_loss.SupConLoss()
    second_ind = True

    for epoch in range(resume_epoch, args.num_epochs + 1):
        lr = args.lr
        if 140 > epoch >= 80:
            lr /= 10
        elif 200> epoch >= 140:
            lr /= 100
        elif epoch >= 200:
            lr /= 200
        print('learning rate: %.4f\n'%lr)
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')

            start_time = time.time()
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader, savelog=True)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader, savelog=False)

            Contrastive_loss.init_prototypes(net1, eval_loader, device)
            Contrastive_loss.init_prototypes(net2, eval_loader, device)

            end_time = round(time.time() - start_time)
            total_time += end_time
            warmup_time += end_time

            if epoch == (warm_up - 1):
                time_log.write('Warmup: %f \n' % (warmup_time))
                time_log.flush()
        else:
            start_time = time.time()

            prob1, all_loss[0], all_preds[0], hist_preds[0], op1 = eval_train(net1, all_loss[0], all_preds[0],
                                                                              hist_preds[0], savelog=True)

            prob2, all_loss[1], all_preds[1], hist_preds[1], op2 = eval_train(net2, all_loss[1], all_preds[1],
                                                                              hist_preds[1], savelog=False)

            # prob1[idx_superclean] = 1
            # prob2[idx_superclean] = 1

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            # 可以注释掉
            sample_rate1 = len((pred1).nonzero()[0]) / len(eval_loader.dataset)
            sample_rate2 = len((pred2).nonzero()[0]) / len(eval_loader.dataset)
            pred1_new = np.zeros(len(eval_loader.dataset)).astype(np.bool)
            pred2_new = np.zeros(len(eval_loader.dataset)).astype(np.bool)
            class_len1 = int(sample_rate1 * len(eval_loader.dataset) / args.num_class)
            class_len2 = int(sample_rate2 * len(eval_loader.dataset) / args.num_class)
            for i in range(args.num_class):
                class_indices = np.where(np.array(eval_loader.dataset.noise_label) == i)[0]
                size1 = len(class_indices)
                class_len_temp1 = min(size1, class_len1)
                class_len_temp2 = min(size1, class_len2)

                prob = np.argsort(-prob1[class_indices])
                select_idx = class_indices[prob[:class_len_temp1]]
                pred1_new[select_idx] = True

                prob = np.argsort(-prob2[class_indices])
                select_idx = class_indices[prob[:class_len_temp2]]
                pred2_new[select_idx] = True
            pred1 = pred1_new
            pred2 = pred2_new

            idx_view_labeled = (pred1).nonzero()[0]
            idx_view_unlabeled = (1 - pred1).nonzero()[0]
            all_idx_view_labeled[0].append(idx_view_labeled)
            all_idx_view_labeled[1].append((pred2).nonzero()[0])
            all_idx_view_unlabeled[0].append(idx_view_unlabeled)
            all_idx_view_unlabeled[1].append((1 - pred2).nonzero()[0])

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader, _ = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, op2['op'])
            # 这里还需要噪声修正
            # device = torch.device("cuda")
            all_indices_x = torch.tensor(pred2.nonzero()[0])
            clean_labels_x = Contrastive_loss.noise_correction(op2['pt'][all_indices_x, :],
                                                               op2['op'][all_indices_x, :],
                                                               op2['pl'][all_indices_x],
                                                               all_indices_x, device)
            all_indices_u = torch.tensor((1 - pred2).nonzero()[0])
            clean_labels_u = Contrastive_loss.noise_correction(op2['pt'][all_indices_u, :],
                                                               op2['op'][all_indices_u, :],
                                                               op2['pl'][all_indices_u],
                                                               all_indices_u, device)

            # update class prototypes
            features = op2['ft'].to(device)
            labels = op2['pl'].to(device)
            labels[all_indices_x] = clean_labels_x
            labels[all_indices_u] = clean_labels_u
            net1.update_prototypes(features, labels)

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader, u_map_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, op1['op'])  # train net2
            # 这里还需要噪声修正
            # device = torch.device("cuda")
            all_indices_x = torch.tensor(pred1.nonzero()[0])
            clean_labels_x = Contrastive_loss.noise_correction(op1['pt'][all_indices_x, :],
                                                               op1['op'][all_indices_x, :],
                                                               op1['pl'][all_indices_x],
                                                               all_indices_x, device)
            all_indices_u = torch.tensor((1 - pred1).nonzero()[0])
            clean_labels_u = Contrastive_loss.noise_correction(op1['pt'][all_indices_u, :],
                                                               op1['op'][all_indices_u, :],
                                                               op1['pl'][all_indices_u],
                                                               all_indices_u, device)

            # update class prototypes
            features = op1['ft'].to(device)
            labels = op1['pl'].to(device)
            labels[all_indices_x] = clean_labels_x
            labels[all_indices_u] = clean_labels_u
            net1.update_prototypes(features, labels)

            end_time = round(time.time() - start_time)
            total_time += end_time

        save_models(epoch, net1, optimizer1, net2, optimizer2, path_exp)

        test(epoch, net1, net2)
        # scheduler1.step()
        # scheduler2.step()

    test_log.write('\nBest:%.2f  avgLast10: %.2f\n' % (max(acc_hist), sum(acc_hist[-10:]) / 10.0))
    test_log.close()

    time_log.write('SSL Time: %f \n' % (total_time - warmup_time))
    time_log.write('Total Time: %f \n' % (total_time))
    time_log.close()

