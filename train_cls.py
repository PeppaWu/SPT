from dataset import ModelNetDataLoader, ScanObjectNN
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf
from spikingjelly.clock_driven import functional
import torch.nn as nn
from torch.nn import functional as F

def test(model, loader, num_class=40):
    classifier = model.eval()
    functional.reset_net(classifier)
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0] if num_class != 15 else target
        points, target = points.cuda(), target.cuda()

        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        functional.reset_net(classifier)
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join([str(g) for g in args['gpu']]) 
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    root = hydra.utils.to_absolute_path(args.dataset.DATA_PATH)
    if args.dataset.name == 'ModelNet':        
        TRAIN_DATASET = ModelNetDataLoader(root=root, nclass=args.dataset.num_class, npoint=args.num_point, split='train', normal_channel=args.dataset.normal)
        TEST_DATASET = ModelNetDataLoader(root=root, nclass=args.dataset.num_class, npoint=args.num_point, split='test', normal_channel=args.dataset.normal)
    elif args.dataset.name == 'ScanObjectNN':
        TRAIN_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='training')
        TEST_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='test')
    else: raise NotImplementedError(f'{args.name} dataset is not found')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=6)

    '''MODEL LOADING'''
    args.input_dim = 6 if args.dataset.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs, which is {args['gpu']}")
        classifier = torch.nn.DataParallel(classifier)
        classifier.to(torch.device('cuda:0'))
    else:
        logger.info(f"Using {torch.cuda.device_count()} GPU, which is {args['gpu']}")
        classifier.to(torch.device('cuda:0'))
    # logger.info(f"Model Structure: {classifier}")

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'AdamW':        
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
        logger.info(f'Using AdamW as model optimizer, lr is {args.learning_rate}')
        logger.info(f'Using StepLR as model scheduler, learning rate decay {scheduler.gamma} for every {scheduler.step_size} epochs')
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), 
                                    lr=args.learning_rate, 
                                    weight_decay=args.weight_decay,
                                    momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)
        logger.info(f'Using SGD as model optimizer, lr is {args.learning_rate}')
        logger.info(f'Using MultiStepLR as model scheduler, learning rate is dropped by 10x at epochs 120 and 160')
    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        classifier.train()
        functional.reset_net(classifier)
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            target = target[:, 0] if args.dataset.name == 'ModelNet' else target
            points = provider.shuffle_points(points)
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            
            functional.reset_net(classifier)
            global_step += 1
            
        scheduler.step()
        
        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        logger.info(f"Train learning rate is {optimizer.param_groups[0]['lr']}")


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, args.dataset.num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Current Epoch: %d, Test Instance Accuracy: %f, Class Accuracy: %f'% ((epoch+1), instance_acc, class_acc))
            logger.info('Best Epoch: %d, Best Instance Accuracy: %f, Class Accuracy: %f'% (best_epoch, best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    main()
