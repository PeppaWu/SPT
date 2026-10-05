# Spiking Point Transformer (AAAI 2025)

![Overview](./fig/Overview.png)

Official PyTorch implementation of [Spiking Point Transformer for Point Cloud Classification](https://arxiv.org/pdf/2502.15811).

## Installation

Use Linux, Python 3.9 and the CUDA 12.1 toolkit with `nvcc` and a GNU C++ compiler. Set `CUDA_HOME` and add `$CUDA_HOME/bin` to `PATH`. Adjust `TORCH_CUDA_ARCH_LIST` for your GPU (9.0 for H800).

```sh
git clone https://github.com/PeppaWu/SPT.git
cd SPT
export TORCH_CUDA_ARCH_LIST=9.0
python -m pip install setuptools==75.9.1 ninja==1.11.1.4
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
(cd ops/fps && python setup.py install)

git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git ../Pointnet2_PyTorch
git -C ../Pointnet2_PyTorch checkout b5ceb6d9ca0467ea34beb81023f96ee82228f626
sed -i 's/os.environ\["TORCH_CUDA_ARCH_LIST"\] = .*/os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")/' ../Pointnet2_PyTorch/pointnet2_ops_lib/setup.py ../Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py
python -m pip install --no-build-isolation ../Pointnet2_PyTorch/pointnet2_ops_lib
```

## Data

Download [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and ScanObjectNN **PB_T50_RS with background**, then arrange the files as follows:

```text
data/
├── modelnet40_normal_resampled/
│   ├── modelnet{10,40}_{shape_names,train,test}.txt
│   └── <class>/<shape_id>.txt
└── ScanObjectNN/main_split/
    ├── training_objectdataset_augmentedrot_scale75.h5
    └── test_objectdataset_augmentedrot_scale75.h5
```

Use `dataset.DATA_PATH=/path/to/data` to override the dataset root.

## Training

Use the configurations in `config/` with DDP. `batch_size` is per GPU.

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_cls.py --config-name modelnet10
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_cls.py --config-name modelnet40
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_cls.py --config-name scanobjectnn
```

Outputs are saved under `logs/`. Add `+resume=/path/to/last_model.pth` to resume training.

## Evaluation

Use the matching configuration and GPU count from training:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 test_cls.py --config-name scanobjectnn +checkpoint=/path/to/model.pth
```

Evaluation uses no voting by default. Add `+vote_num=100` for voting. OA and mAcc are saved in `evaluation.json`.

## Citation

```bibtex
@article{wu2025spiking,
  title={Spiking Point Transformer for Point Cloud Classification},
  author={Wu, Peixi and Chai, Bosong and Li, Hebei and Zheng, Menghua and Peng, Yansong and Wang, Zeyu and Nie, Xuan and Zhang, Yueyi and Sun, Xiaoyan},
  journal={arXiv preprint arXiv:2502.15811},
  year={2025}
}
```

## Acknowledgements

We thank [Point-Transformer](https://github.com/qq456cvb/Point-Transformers), [Spike-Driven-Transformer](https://github.com/BICLab/Spike-Driven-Transformer) and [SpikingJelly](https://github.com/fangwei123456/spikingjelly).
