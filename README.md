# NPP
Neural Architecture Search for Joint Human Parsing and Pose Estimation (ICCV2021)

## Preparation
1. Download the LIP dataset from https://www.sysuhcp.com/lip
2. Put the `prepare_file.zip` to your root path of LIP, such as `/home/data/LIP/prepare_file.zip`, and unzip it.
3. `pip install -r requirements.txt`

## Search
We only release the search of interaction (without encoder-decoder search) now.
Searching with 4 gpus:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 search_lip_sync.py --cfg ./experiments/lip/384_384.yaml`

## Train
Download the pretrained backbone from link: https://pan.baidu.com/s/1Cs_pf3WsZypwrexrEryS_g 
with the extracing code: 0000, and modify the model path in the line 205 of augment_lip_sybc.py.
The newest model weight can be also downloaded from the link.

Training with 4 gpus:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 augment_lip_sync.py --cfg ./experiments/lip/384_384.yaml`

## Q&A
If you have any questions, please concat with `huangyuhang@shu.edu.cn`.

## Thanks
Thanks to the public repo: [mula(ECCV2018)]() for providing the base code.
