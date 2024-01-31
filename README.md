## Requirements
* Pytorch>=1.6.0, <1.9.0 (>=1.1.0 should work but not tested)
* timm==0.3.2

## Experiments

### ISIC2017 Skin Lesion Segmentation Challenge
GPUs of memory>=4G shall be sufficient for this experiment. 

1. Preparing necessary data:
	+ downloading ISIC2017 training, validation and testing data from the [official site](https://challenge.isic-archive.com/data), put the unzipped data in `./data`.
	+ run `process.py` to preprocess all the data, which generates `data_{train, val, test}.npy` and `mask_{train, val, test}.npy`.
	+ alternatively, the processed data is provided in [Baidu Pan, pw:ymrh](https://pan.baidu.com/s/1EkMvfRj9pGCu1iqXjvg9ZA) and [Google Drive](https://drive.google.com/file/d/120hxkYc0vfzoSf4kYC6zpC7FH7XCVXqK/view?usp=sharing).

2. Testing:
	+ downloading our trained TransFuse-S from [Baidu Pan, pw:xd74](https://pan.baidu.com/s/1khwcCcTgwporZJcaTWedRg) or [Google Drive](https://drive.google.com/file/d/1hv1mfFkWEdYCR0FHPokovlf7OAFsnKgY/view?usp=sharing) to `./snapshots/`.
	+ run `test_isic.py --ckpt_path='snapshots/TransFuse-19_best.pth'`.

3. Training:
	+ downloading DeiT-small from [DeiT repo](https://github.com/facebookresearch/deit) to `./pretrained`.
	+ downloading resnet-34 from [timm Pytorch](https://download.pytorch.org/models/resnet34-333f7ec4.pth) to `./pretrained`.
	+ run `train_isic.py`; you may also want to change the default saving path or other hparams as well.

## Acknowledgement
The codes borrow heavily from TransFuse:https://github.com/Rayicer/TransFuse and we really appreciate it.
