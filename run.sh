export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0

if true; then
CKPT_PATH=data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
BACKBONE=hr48
else
CKPT_PATH=data/ckpt/res50-PA45.7_MJE72.0_MVE85.3_3dpw.pt
BACKBONE=res50
fi

python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
               --input_path test_samples/01e222117f63f874010370037f551497ac_258.mp4 --input_type video \
               --save_results --make_video --frame_rate 30

python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
               --input_path test_samples/62883594000000000102c16c.mp4 --input_type video \
               --multi --infill --smooth --save_results --make_video --frame_rate 30