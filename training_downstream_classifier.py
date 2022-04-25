import os
import argparse
import random

import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset_evaluation
from models import get_encoder_architecture_usage
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature


class FullNet(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, return_feature=False):
        feature = self.encoder(x)
        feature = F.normalize(feature, dim=1)
        output = self.classifier(feature)
        if return_feature:
            return output, feature
        else:
            return output


def net_test_clean(net, test_loader, epoch, criterion, keyword='Accuracy'):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0
    
    for data, target in test_loader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = net.forward(data, return_feature=False)
        test_loss += criterion(output, target.long()).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(
        keyword, test_acc, epoch))

    return test_acc


def net_test_pgd(net, test_loader, epoch, criterion, keyword='Accuracy', return_feature=False):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0
    correct_targeted = 0.0

    ### transform (1/2) ###
    if isinstance(test_loader_clean.dataset.transform, torchvision.transforms.Compose):
        mean = [0,0,0]
        std = [1,1,1]
        for T in test_loader_clean.dataset.transform.transforms:  # find normalization mean and std
            if isinstance(T, torchvision.transforms.Normalize):
                mean = T.mean
                std = T.std
                normalize = T
                break
            elif not isinstance(T, torchvision.transforms.ToTensor):
                print('[warning] this transform may cause unexpected result:', T)
    else:
        assert isinstance(test_loader_clean.dataset.transform, torchvision.transforms.ToTensor)
        mean = [0,0,0]
        std = [1,1,1]
        normalize = torchvision.transforms.Normalize(mean=mean, std=std)
    transform_backup = test_loader_clean.dataset.transform  # backup transform
    test_loader_clean.dataset.transform = torchvision.transforms.ToTensor()  # change transform
    #######################

    ### attack config ###
    import foolbox as fb
    preprocessing = dict(mean=mean, std=std, axis=-3)
    fmodel = fb.PyTorchModel(net, bounds=(0, 1), preprocessing=preprocessing)
    attack = fb.attacks.LinfPGD()
    epsilons = [8/255, 16/255]
    torch.manual_seed(7)
    #####################

    feature_clean = []
    feature_adv = [[] for _ in epsilons]
    feature_adv_targeted = [[] for _ in epsilons]

    for data, target in tqdm(test_loader, desc=keyword):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # untargeted
        _, advs, success = attack(fmodel, data, criterion=target, epsilons=epsilons)
        # targeted
        target_adv = torch.randint(0, num_of_classes, target.shape).type(target.dtype).to(target.device)
        target_adv[target_adv == target] = (target_adv[target_adv == target] + 1) % num_of_classes
        target_adv = fb.criteria.TargetedMisclassification(target_adv)
        _, advs_targeted, success_targeted = attack(fmodel, data, criterion=target_adv, epsilons=epsilons)

        # advs: clipped adv, list, len = len(epsilons), element's shape = data.shape
        # success: success adv, tensor of shape (len(epsilons), len(data)), dtype is boolean
        correct += success.sum().item()
        correct_targeted += success_targeted.sum().item()

        if return_feature:
            _, feature = net.forward(normalize(data), return_feature=True)
            feature_clean.append(feature.cpu().detach())

            for i, adv in enumerate(advs):
                _, feature = net.forward(normalize(adv), return_feature=True)
                feature_adv[i].append(feature.cpu().detach())

            for i, adv in enumerate(advs_targeted):
                _, feature = net.forward(normalize(adv), return_feature=True)
                feature_adv_targeted[i].append(feature.cpu().detach())
            
    test_acc = 100. * correct / (len(test_loader.dataset) * len(epsilons))
    test_acc_targeted = 100. * correct_targeted / (len(test_loader.dataset) * len(epsilons))
    print('{{"metric": "Untargeted - {}", "value": {}, "epoch": {}}}'.format(keyword, test_acc, epoch))
    print('{{"metric": "Targeted   - {}", "value": {}, "epoch": {}}}'.format(keyword, test_acc_targeted, epoch))

    ### transform (2/2) ###
    test_loader_clean.dataset.transform = transform_backup  # restore transform
    #######################

    if return_feature:
        feature_clean = np.concatenate(feature_clean)

        for i in range(len(feature_adv)):
            feature_adv[i] = np.concatenate(feature_adv[i])
        feature_adv = np.concatenate(feature_adv)

        for i in range(len(feature_adv_targeted)):
            feature_adv_targeted[i] = np.concatenate(feature_adv_targeted[i])
        feature_adv_targeted = np.concatenate(feature_adv_targeted)

        return (test_acc + test_acc_targeted)/2, feature_clean, np.concatenate([feature_adv, feature_adv_targeted])
    else:
        return (test_acc + test_acc_targeted)/2


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--reference_label', default=-1, type=int, help='target class in the target downstream task')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')

    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    ## note that the reference_file is not needed to train a downstream classifier
    parser.add_argument('--reference_file', default='', type=str, help='path to the reference file (default: none)')

    parser.add_argument('--backdoor_mode', default='badencoder', type=str, choices=['badencoder', 'dittoencoder', 'mewencoder'])
    parser.add_argument('--trigger_alpha', default=1, type=float, help='transparency of trigger, 0 means totally transparent, 1 means totally not transparent')

    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the downstream encoder')
    parser.add_argument('--eval', default='', type=str, metavar='PATH', help='eval trained downstream classifier, no training')
    parser.add_argument('--eval_attack', default='none', type=str, choices=['none', 'pgd', 'pgd_honeypot'])

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


    if args.backdoor_mode == 'badencoder':
        assert args.reference_label >= 0, 'Enter the correct target class'
    else:
        assert args.reference_label < 0, 'set reference_label < 0 to make test_data_backdoor return gt label'

    args.data_dir = f'./data/{args.dataset}/'
    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(args)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                   pin_memory=True)
    test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                      pin_memory=True)

    for normalization in test_loader_clean.dataset.transform.transforms:
        if isinstance(normalization, torchvision.transforms.Normalize):
            break

    # target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_of_classes = len(train_data.classes)

    model = get_encoder_architecture_usage(args).cuda()

    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        if args.encoder_usage_info in ['CLIP', 'imagenet'] and 'clean' in args.encoder:
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor)
        # feature_bank_target, label_bank_target = predict_feature(model.visual, target_loader)
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor)
        # feature_bank_target, label_bank_target = predict_feature(model.f, target_loader)

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)

    input_size = feature_bank_training.shape[1]

    criterion = nn.CrossEntropyLoss()

    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()


    if not args.eval:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
        for epoch in range(1, args.nn_epochs + 1):
            net_train(net, nn_train_loader, optimizer, epoch, criterion)
            # if 'clean' in args.encoder:
            #     net_test(net, nn_test_loader, epoch, criterion, 'Clean Accuracy (CA)')
            #     net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate-Baseline (ASR-B)')
            # else:
            #     net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
            #     net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
            net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR, untargeted)')

        if args.results_dir:
            os.makedirs(args.results_dir, exist_ok=True)
            torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_' + str(epoch) + '.pth')
    else:
        ckpt = torch.load(args.eval)
        epoch = ckpt['epoch']
        net.load_state_dict(ckpt['state_dict'])
        if args.encoder_usage_info in ['CLIP', 'imagenet']:
            raise NotImplementedError
        else:
            fullnet = FullNet(model.f, net)

        if args.eval_attack == 'none':
            net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR, untargeted)')
        elif args.eval_attack == 'pgd':
            net_test_pgd(fullnet, test_loader_clean, epoch, criterion, 'pgd')
        elif args.eval_attack == 'pgd_honeypot':
            _, feature_clean, feature_adv = net_test_pgd(fullnet, test_loader_clean, epoch, criterion, 'pgd', return_feature=True)

            root = f"output/pgd_honeypot/{next(x for x in str(args.eval).split('/') if 'encoder' in x)}_alpha{args.trigger_alpha}"
            os.makedirs(root, exist_ok=True)
            np.save(root + '/feature_clean.npy', feature_clean)
            np.save(root + '/feature_adv.npy', feature_adv)
            np.save(root + '/feature_bank_backdoor.npy', feature_bank_backdoor)
            print('results saved in', root)
        else:
            raise NotImplementedError