# CLIFF [ECCV 2022 Oral]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cliff-carrying-location-information-in-full/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=cliff-carrying-location-information-in-full)

<p float="left">
   <img src="https://github.com/huawei-noah/noah-research/blob/master/CLIFF/assets/teaser.gif" width="100%">
</p>


## Introduction
This repo is highly built on the official [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) and contains an inference demo, and further adds accurate detector and multi-person tracker. For post-processing, motion interpolation and smooth are supported for better visualization results.


[**CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation**](
    https://arxiv.org/abs/2208.00571
).

## Major features
- [x] **[08/20/22]** Support YOLOX as single-person detector, better performance on single frame.
- [x] **[08/20/22]** Support ByteTrack as multi-person tracker, better performance on person Re-ID.
- [x] **[08/20/22]** Support linear interpolation as motion completion method, especially for occlusion.
- [x] **[08/20/22]** Support Smooth-Net as post-processing motion smooth for decresing motion jittering.
- [x] **[09/29/22]** Support SMPLify fitting given GT/Pred 2D keypoints for improving the quality of estimated SMPL params.
- [x] **[01/31/23]** Further support motion smooth for SMPL pose and translation besides of 3D joints.

## Preparation
```bash
conda create -n cliff python=3.10
pip install -r requirements.txt
```

1. Download [the SMPL models](https://smpl.is.tue.mpg.de) for rendering the reconstructed meshes
2. Download the pretrained checkpoints to run the demo [[Google Drive](
    https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP?usp=sharing)]
3. Install MMDetection and download [the pretrained checkpoints](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox)
4. Install MMTracking and download [the pretrained checkpoints](https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack)

Finally put these data following the directory structure as below:
```
${ROOT}
|-- data
    smpl_mean_params.npz
    |-- ckpt
        |-- hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
        |-- res50-PA45.7_MJE72.0_MVE85.3_3dpw.pt
        |-- hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
        |-- SMPL_NEUTRAL.pkl
|-- mmdetection
    |-- checkpoints
        |-- yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
|-- mmtracking
    |-- checkpoints
        |-- bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
```

## Demo

We provide demos for single-person and multi-person video.
### Single-person
Run the following command to test CLIFF on a single-person video:
```
python demo.py --ckpt data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt \
               --backbone hr48 \
               --input_path test_samples/01e222117f63f874010370037f551497ac_258.mp4 \
               --input_type video \
               --save_results \
               --make_video \
               --frame_rate 30
```
### Multi-person
Use the `--multi` flag to support multi-person tracking, `--infill` flag to support motion infill, `--smooth` flag to support motion smooth. Run the following command to test CLIFF on a multi-person video with post-processing:
```
python demo.py --ckpt data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt \
               --backbone hr48 \
               --input_path test_samples/62883594000000000102c16c.mp4 \
               --input_type video \
               --multi \
               --infill \
               --smooth \
               --save_results \
               --make_video \
               --frame_rate 30
```

## SMPLify Fitting

As the same as [SPIN](https://github.com/nkolot/SPIN), we apply SMPLify fitting after CLIFF, OpenPose format 2D Keypoints are required for convinence.

```
python3 demo_fit.py --img=examples/im1010.jpg \ 
                    --openpose=examples/im1010_openpose.json
```

## Citing
```
@Inproceedings{li2022cliff,
  Title     = {CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation},
  Author    = {Li, Zhihao and Liu, Jianzhuang and Zhang, Zhensong and Xu, Songcen and Yan, Youliang},
  Booktitle = {ECCV},
  Year      = {2022}
}
```

## Contact
If you have problems about usage, feel free to open an issue or directly contact me via: haofanwang.ai@gmail.com. But please note that I'm NOT the author of CLIFF, so for any question about the paper, contact the author.
