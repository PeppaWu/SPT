from dataset import ModelNetDataLoader, ScanObjectNN
import numpy as np
import os
import sys
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import provider
import importlib
import shutil
import hydra
from hydra.core.hydra_config import HydraConfig
import omegaconf
from spikingjelly.clock_driven import functional
import time
import json
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, Sampler
from torch.utils.data._utils.collate import default_collate

def distributed():
    return dist.is_available() and dist.is_initialized()


def is_main():
    return not distributed() or dist.get_rank() == 0


class EvaluationSampler(Sampler):
    """Shard without padding: every validation sample is counted exactly once."""
    def __init__(self, dataset, rank, world_size):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def augment_batch(batch, dropout=0.2, scale_low=0.85, scale_high=1.15):
    # Run CPU augmentation in workers, before DataLoader pins the result.
    points, target = default_collate(batch)
    points = provider.shuffle_points(points.numpy())
    if dropout:
        points = provider.random_point_dropout(points, max_dropout_ratio=dropout)
    points[:, :, :3] = provider.random_scale_point_cloud(
        points[:, :, :3], scale_low=scale_low, scale_high=scale_high)
    points[:, :, :3] = provider.shift_point_cloud(points[:, :, :3])
    return torch.from_numpy(np.ascontiguousarray(points)), target.reshape(-1).long()


@torch.no_grad()
def test(model, loader, num_class=40, vote_num=1):
    if vote_num < 1:
        raise ValueError('vote_num must be at least 1')
    # Unequal validation shard sizes must not invoke DDP forward collectives.
    classifier = model.module if isinstance(model, DDP) else model
    classifier.eval()
    functional.reset_net(classifier)
    if distributed():
        for buffer in classifier.buffers():
            dist.broadcast(buffer, src=0)
    counts = torch.zeros(2, num_class, dtype=torch.long).cuda()
    for points, target in tqdm(loader, total=len(loader), disable=not is_main()):
        points = points.cuda(non_blocking=True)
        target = target.reshape(-1).long().cuda(non_blocking=True)
        pred = None
        for _ in range(vote_num):
            logits = classifier(points)
            functional.reset_net(classifier)
            if logits.shape != (len(target), num_class):
                raise ValueError(f'Expected logits [batch, {num_class}], got {tuple(logits.shape)}')
            pred = logits if pred is None else pred + logits
        correct = pred.argmax(1).eq(target)
        counts[0] += torch.bincount(target, minlength=num_class)
        counts[1] += torch.bincount(target[correct], minlength=num_class)
    if distributed():
        dist.all_reduce(counts)
    seen, correct = counts.cpu().numpy()
    if seen.sum() == 0:
        raise ValueError('Evaluation loader contains no samples')
    present = seen > 0
    return float(correct.sum() / seen.sum()), float(np.mean(correct[present] / seen[present]))


@hydra.main(config_path='config', config_name='cls', version_base='1.1')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    # torchrun's visible-device mapping takes precedence over Hydra's gpu list.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in args.gpu)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    visible_devices = torch.cuda.device_count()
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    if local_world_size > visible_devices or local_rank >= visible_devices:
        raise ValueError(
            f'{local_world_size} local processes requested but only {visible_devices} GPUs visible; '
            'set CUDA_VISIBLE_DEVICES or gpu to match --nproc_per_node')
    torch.cuda.set_device(local_rank)
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        run_dir = [HydraConfig.get().runtime.output_dir if is_main() else None]
        dist.broadcast_object_list(run_dir, src=0)
        os.chdir(run_dir[0])
    args.world_size = dist.get_world_size() if distributed() else 1
    args.global_batch_size = args.batch_size * args.world_size
    args.visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    try:
        return train(args)
    finally:
        if distributed():
            dist.destroy_process_group()


def train(args):
    logger = logging.getLogger(__name__)
    logger.disabled = not is_main()
    world_size = dist.get_world_size() if distributed() else 1
    rank = dist.get_rank() if distributed() else 0

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    root = hydra.utils.to_absolute_path(args.dataset.DATA_PATH)
    if args.dataset.name == 'ModelNet':        
        TRAIN_DATASET = ModelNetDataLoader(root=root, nclass=args.dataset.num_class, npoint=args.num_point, split='train', normal_channel=args.dataset.normal, random_points=args.dataset.get('random_points', False))
        TEST_DATASET = ModelNetDataLoader(root=root, nclass=args.dataset.num_class, npoint=args.num_point, split='test', normal_channel=args.dataset.normal)
    elif args.dataset.name == 'ScanObjectNN':
        TRAIN_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='training')
        TEST_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='test')
    else: raise NotImplementedError(f'{args.name} dataset is not found')
    logger.info('Dataset root=%s; train=%d; test=%d', root, len(TRAIN_DATASET), len(TEST_DATASET))
    # batch_size is per GPU. Keep it fixed for throughput scaling.
    train_sampler = DistributedSampler(TRAIN_DATASET, shuffle=True, drop_last=True) if distributed() else None
    eval_sampler = EvaluationSampler(TEST_DATASET, rank, world_size)
    workers = int(args.get('num_workers', 4))
    loader_options = dict(num_workers=workers, pin_memory=True,
                          persistent_workers=workers > 0)
    if workers:
        # CUDA/NCCL have already initialized; do not fork them into workers.
        loader_options['multiprocessing_context'] = 'spawn'
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, drop_last=True, collate_fn=partial(augment_batch, **dict(args.get('augmentation', {}))), **loader_options)
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, sampler=eval_sampler, **loader_options)
    if not len(trainDataLoader):
        raise ValueError('No training batches; reduce per-GPU batch_size or world size')
    logger.info('DDP world_size=%d; batch per GPU=%d; global batch=%d; workers per rank=%d',
                world_size, args.batch_size, args.batch_size * world_size, workers)

    '''MODEL LOADING'''
    args.input_dim = 6 if args.dataset.normal else 3
    if is_main():
        shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args)
    class_weight = None
    if args.class_weight_power > 0:
        if hasattr(TRAIN_DATASET, 'label'):
            train_labels = TRAIN_DATASET.label.reshape(-1)
        else:
            train_labels = [TRAIN_DATASET.classes[category] for category, _ in TRAIN_DATASET.datapath]
        class_counts = np.bincount(train_labels, minlength=args.dataset.num_class)
        assert np.all(class_counts > 0)
        class_weight = torch.as_tensor(class_counts.mean() / class_counts, dtype=torch.float32).cuda()
        class_weight = class_weight.pow(float(args.class_weight_power))
        class_weight /= class_weight.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=float(args.label_smoothing))

    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    classifier.to(device)
    resume = args.get('resume')
    checkpoint = None
    start_epoch = 0
    if resume:
        checkpoint_path = Path(hydra.utils.to_absolute_path(resume))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        weights = {k.removeprefix('module.'): v for k, v in checkpoint['model_state_dict'].items()}
        classifier.load_state_dict(weights)
        start_epoch = int(checkpoint.get('epoch', 0))
        logger.info('Loaded checkpoint %s; next epoch=%d', checkpoint_path, start_epoch + 1)
    elif is_main():
        Path('metrics.jsonl').write_text('')
        for name in ('best_model.pth', 'last_model.pth'):
            Path(name).unlink(missing_ok=True)
    if distributed():
        classifier = DDP(classifier, device_ids=[device.index], output_device=device.index,
                         gradient_as_bucket_view=True)

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
    
    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_instance_acc = checkpoint.get('best_instance_acc', 0.0) if checkpoint else 0.0
    best_class_acc = checkpoint.get('best_class_acc', 0.0) if checkpoint else 0.0
    best_epoch = checkpoint.get('best_epoch', 0) if checkpoint else 0
    cumulative_correct = checkpoint.get('cumulative_correct', 0) if checkpoint else 0
    cumulative_samples = checkpoint.get('cumulative_samples', 0) if checkpoint else 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d/%d:', epoch + 1, args.epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_lr = optimizer.param_groups[0]['lr']
        stats = torch.zeros(3, dtype=torch.float64).cuda()
        classifier.train()
        functional.reset_net(classifier)
        if distributed():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader),
                                   smoothing=0.9, disable=not is_main()):
            points = points.cuda(non_blocking=True)
            target = target.reshape(-1).long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = classifier(points)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            stats[0] += pred.detach().argmax(1).eq(target).sum()
            stats[1] += points.size(0)
            stats[2] += loss.detach().double() * points.size(0)
            functional.reset_net(classifier)
        if distributed():
            dist.all_reduce(stats)
        epoch_correct, epoch_samples, epoch_loss_sum = stats.cpu().tolist()
        train_seconds = time.perf_counter() - epoch_start
        logger.info('Train time %.3fs; throughput %.2f samples/s',
                    train_seconds, epoch_samples / train_seconds)
        scheduler.step()

        cumulative_correct += epoch_correct
        cumulative_samples += epoch_samples
        train_instance_acc = epoch_correct / epoch_samples
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        logger.info(f"Train learning rate is {optimizer.param_groups[0]['lr']}")


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, args.dataset.num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_class_acc = class_acc
                best_epoch = epoch + 1
            logger.info('Current Epoch: %d, Test Instance Accuracy: %f, Class Accuracy: %f'% ((epoch+1), instance_acc, class_acc))
            logger.info('Best Epoch: %d, Best Instance Accuracy: %f, Class Accuracy: %f'% (best_epoch, best_instance_acc, best_class_acc))

            if is_main():
                metrics = {
                    'epoch': epoch + 1,
                    'train_seconds': train_seconds,
                    'train_samples_per_second': epoch_samples / train_seconds,
                    'global_batch_size': args.batch_size * world_size,
                    'train_oa_cumulative': cumulative_correct / cumulative_samples,
                    'test_oa': float(instance_acc),
                    'test_macc': float(class_acc),
                    'learning_rate': epoch_lr,
                    'best_test_oa': float(best_instance_acc),
                    'best_test_macc': float(best_class_acc),
                    'best_oa_epoch': best_epoch,
                }
                if epoch_samples:
                    metrics['train_loss'] = epoch_loss_sum / epoch_samples
                    metrics['train_oa'] = epoch_correct / epoch_samples
                with open('metrics.jsonl', 'a') as stream:
                    stream.write(json.dumps(metrics) + '\n')

            save_best = instance_acc >= best_instance_acc
            save_every = int(args.get('save_last_every', 0))
            save_last = save_every > 0 and ((epoch + 1) % save_every == 0 or epoch + 1 == args.epoch)
            if is_main() and (save_best or save_last):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'model_state_dict': (classifier.module if isinstance(classifier, DDP) else classifier).state_dict(),
                    'config': omegaconf.OmegaConf.to_container(args, resolve=True),
                    'epoch': epoch + 1,
                    'instance_acc': float(instance_acc),
                    'class_acc': float(class_acc),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_instance_acc': best_instance_acc,
                    'best_class_acc': best_class_acc,
                    'best_epoch': best_epoch,
                    'cumulative_correct': cumulative_correct,
                    'cumulative_samples': cumulative_samples,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                for savepath in (['best_model.pth'] if save_best else []) + (['last_model.pth'] if save_last else []):
                    torch.save(state, savepath + '.tmp')
                    os.replace(savepath + '.tmp', savepath)


    logger.info('End of training...')

if __name__ == '__main__':
    if 'RANK' in os.environ:
        # Each process owns its Hydra metadata/log files, including during startup.
        rank = os.environ['RANK']
        sys.argv.extend([f'hydra.output_subdir=.hydra/rank{rank}',
                         f'hydra.job.name=train_cls_rank{rank}'])
    main()
