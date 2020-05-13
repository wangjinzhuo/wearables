export CUDA_VISIBLE_DEVICES=7;\
python3 trainval.py \
	--network utime \
	--seq_len 35 \
	--batch_size 32 \
	--loss dice_loss \
	--opt 'sgd' \
	--lr 0.001
