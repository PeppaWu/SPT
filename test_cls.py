"""Evaluate a saved SPT checkpoint, with optional test voting."""
from dataset import ModelNetDataLoader, ScanObjectNN
import importlib
import json
import logging
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import omegaconf
import torch
import torch.distributed as dist

from train_cls import EvaluationSampler, distributed, is_main, test


def test_novote(model, loader, num_class=40):
    return test(model, loader, num_class=num_class, vote_num=1)


@hydra.main(config_path='config', config_name='cls', version_base='1.1')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in args.gpu)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('nccl')
        run_dir = [HydraConfig.get().runtime.output_dir if is_main() else None]
        dist.broadcast_object_list(run_dir, src=0)
        os.chdir(run_dir[0])
    try:
        return evaluate(args)
    finally:
        if distributed():
            dist.destroy_process_group()


def evaluate(args):
    logger = logging.getLogger(__name__)
    logger.disabled = not is_main()
    checkpoint_arg = args.get('checkpoint')
    if not checkpoint_arg:
        raise ValueError('Set +checkpoint=/path/to/best_model.pth for evaluation')
    checkpoint_path = Path(hydra.utils.to_absolute_path(checkpoint_arg))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved = checkpoint.get('config', {})
    if saved:
        # The checkpoint determines the architecture, class count and point count.
        # DATA_PATH remains configurable so checkpoints work on another machine.
        args.model = omegaconf.OmegaConf.create(saved['model'])
        for key in ['name', 'num_class', 'normal']:
            args.dataset[key] = saved['dataset'][key]
        args.num_point = saved['num_point']
        args.batch_size = saved['batch_size']
    args.input_dim = 6 if args.dataset.normal else 3
    root = hydra.utils.to_absolute_path(args.dataset.DATA_PATH)
    if args.dataset.name == 'ModelNet':
        dataset = ModelNetDataLoader(root=root, nclass=args.dataset.num_class,
            npoint=args.num_point, split='test', normal_channel=args.dataset.normal)
    elif args.dataset.name == 'ScanObjectNN':
        dataset = ScanObjectNN(root=root, num_points=args.num_point, split='test')
    else:
        raise NotImplementedError(args.dataset.name)
    rank = dist.get_rank() if distributed() else 0
    world = dist.get_world_size() if distributed() else 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
        sampler=EvaluationSampler(dataset, rank, world), num_workers=0, pin_memory=True)
    cls = getattr(importlib.import_module(f'models.{args.model.name}.model'), 'PointTransformerCls')
    model = cls(args).cuda()
    weights = {k.removeprefix('module.'): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(weights, strict=True)
    votes = int(args.get('vote_num', 1))
    oa, macc = test(model, loader, args.dataset.num_class, vote_num=votes)
    result = dict(checkpoint=str(checkpoint_path), epoch=checkpoint.get('epoch'),
        dataset=args.dataset.name, num_class=args.dataset.num_class,
        timestep=args.model.timestep, num_samples=args.model.num_samples,
        num_points=args.num_point, input_dim=args.input_dim,
        world_size=world, per_rank_batch=args.batch_size, vote_num=votes,
        correct=round(oa * len(dataset)), total=len(dataset), oa=oa, macc=macc)
    if is_main():
        logger.info('Test Instance Accuracy: %f, Class Accuracy: %f', oa, macc)
        output_arg = args.get('output')
        output = (Path(hydra.utils.to_absolute_path(output_arg))
                  if output_arg else Path('evaluation.json'))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
