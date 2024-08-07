# NPPNet
Neural Architecture Search for Joint Human Parsing and Pose Estimation (ICCV2021)

## Preparation
1. Download the LIP dataset from https://www.sysuhcp.com/lip. You can also download it from my baiduDisk: 
Link：https://pan.baidu.com/s/157uoJWO0H_nV4HbzNE62fg?pwd=h6em 
Password：h6em
2. Put the `prepare_file.zip` to your root path of LIP, such as `/home/data/LIP/prepare_file.zip`, and unzip it.
3. `pip install -r requirements.txt`

## Search
We only release the search of interaction (without encoder-decoder search) now.
Searching with 4 gpus:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 search_lip_sync.py --cfg ./experiments/lip/384_384.yaml`

## Train
Download the pretrained backbone from link: https://pan.baidu.com/s/1sIcNXZnM4_U6nthmJAo6NA?pwd=2050 
with the extracing code: 2050, and modify the model path in the line 205 of augment_lip_sync.py.
The newest model weight can be also downloaded from the link.

Training with 4 gpus:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 augment_lip_sync.py --cfg ./experiments/lip/384_384.yaml`

## Q&A
If you have any questions, please concat with `huangai@nudt.edu.cn`.

## Thanks
Thanks to the public repo: [mula(ECCV2018)](https://github.com/GuHuangAI/pytorch-mula) for providing the base code.
