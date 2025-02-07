import os
import json

epoch1 = 260
epoch2 = 270
epoch3 = 280
epoch4 = 290
epoch5 = 300
epoch6 = 310
epoch7 = 320

# ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch1) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-1.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch2) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-2.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch3) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-3.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch4) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-4.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch5) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-5.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch6) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-6.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch7) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-7.txt 2>&1 &'
# os.system(ml)

# ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch1) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-11.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch2) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-22.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch3) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-33.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch4) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-44.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch5) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-55.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch6) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-66.txt 2>&1 &'
# os.system(ml)
# ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch7) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-77.txt 2>&1 &'
# os.system(ml)

ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch1) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES.txt 2>&1 &'
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name PCS --action-type all --score_type PCS --lr 5e-2 --epoch ' + str(epoch1) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > PCS.txt 2>&1 &'
os.system(ml)

ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch2) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-2.txt 2>&1 &'
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name PCS --action-type all --score_type PCS --lr 5e-2 --epoch ' + str(epoch2) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > PCS-2.txt 2>&1 &'
os.system(ml)

ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch3) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-3.txt 2>&1 &'
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name PCS --action-type all --score_type PCS --lr 5e-2 --epoch ' + str(epoch3) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > PCS-3.txt 2>&1 &'
os.system(ml)

ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch4) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-4.txt 2>&1 &'
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name PCS --action-type all --score_type PCS --lr 5e-2 --epoch ' + str(epoch4) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > PCS-4.txt 2>&1 &'
os.system(ml)

ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch5) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-5.txt 2>&1 &'
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name PCS --action-type all --score_type PCS --lr 5e-2 --epoch ' + str(epoch5) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > PCS-5.txt 2>&1 &'
os.system(ml)

ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch6) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-6.txt 2>&1 &'
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name PCS --action-type all --score_type PCS --lr 5e-2 --epoch ' + str(epoch6) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > PCS-6.txt 2>&1 &'
os.system(ml)

ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name TES --action-type all --score_type TES --lr 5e-2 --epoch ' + str(epoch7) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > TES-7.txt 2>&1 &'
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path /data/xhb/GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /data/xhb/GDLT/fis-v/train.txt --test-label-path /data/xhb/GDLT/fis-v/test.txt --model-name PCS --action-type all --score_type PCS --lr 5e-2 --epoch ' + str(epoch7) + ' --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.002 --dropout 0.7 > PCS-7.txt 2>&1 &'
os.system(ml)