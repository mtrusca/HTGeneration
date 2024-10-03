
CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision="fp16"  diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
    --dataset_name="/home/pricie/trusca/htr/data/data_iam_train.csv" \
    --enable_xformers_memory_efficient_attention \
    --resolution1=256 \
    --resolution2=256 \
    --random_flip \
    --train_batch_size=64 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=10000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=1e-04 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=469 \
    --cache_dir="/home/local/maria/cache"\
    --output_dir="/home/local/maria/htr/ckpt/model_pix_265_256"

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16"  diffusers/examples/instruct_pix2pix/train_instruct_pix2pix_scratch.py \
    --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
    --dataset_name="/home/pricie/trusca/htr/data/data_iam_train1.csv" \
    --enable_xformers_memory_efficient_attention \
    --resolution1=256 \
    --resolution2=16 \
    --random_flip \
    --train_batch_size=64 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=10000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=1e-04 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=469 \
    --cache_dir="/home/local/maria/cache"\
    --output_dir="/home/local/maria/htr/ckpt/model_pix_256_16_train"


CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16"  diffusers/examples/instruct_pix2pix/train_instruct_pix2pix_scratch_char.py \
    --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
    --dataset_name="/home/pricie/trusca/htr/data/data_iam_train1.csv" \
    --enable_xformers_memory_efficient_attention \
    --resolution1=256 \
    --resolution2=16 \
    --random_flip \
    --train_batch_size=64 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=10000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=1e-04 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=469 \
    --cache_dir="/home/local/maria/cache"\
    --output_dir="/home/local/maria/htr/ckpt/model_pix_256_16_train_char"

CUDA_VISIBLE_DEVICES=0 python /home/pricie/trusca/htr/diffusers/examples/instruct_pix2pix/test_instruct_pix2pix_multiple_images.py \
--path_data=/home/pricie/trusca/htr/data/data_iam_test1.csv \
--output_dir=/home/pricie/trusca/htr/data/results/model_pix_256_256

CUDA_VISIBLE_DEVICES=0 python /home/pricie/trusca/htr/diffusers/examples/instruct_pix2pix/test_instruct_pix2pix_one_image.py \
--prompt="A line with the text 'recitatione'" \
--img_path='/home/pricie/trusca/htr/data/results/input_image.jpg' \
--img_path_output='/home/pricie/trusca/htr/data/results/output_image.jpg' \
--font_path='/home/pricie/trusca/htr/data/utils/times.ttf'




