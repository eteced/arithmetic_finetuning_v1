import sys
import os
import string
import time

DATA_SPLIT_FLODER='./data_split/'
DATA_SPLIT_NUM = 2
TOTAL_EPOCH_ORIGIN = 12
CHECKPOINT_FOLDER='./checkpoint/'
CHECKPOINT_START_SPLIT_EPOCH=19
EPOCH_BASE = 12
EVAL_APPEND = ' --eval True'
# EVAL_APPEND = ''

os_cmd = "torchrun --nproc_per_node 1 finetuning_arthllama.py     --model Arth_Llama7B     --llama_model_path /home/eteced/dl_workspace/model_repo.folder/llama_ord/     --data_path {}     --arth_model_path ./checkpoint/     --max_seq_len 512     --batch_size 1     --epochs {}     --warmup_epochs 2     --blr 9e-3     --weight_decay 0.02     --output_dir {} --device cuda --resume {} "
os_cmd_nocheckpoint = "torchrun --nproc_per_node 1 finetuning_arthllama.py     --model Arth_Llama7B     --llama_model_path /home/eteced/dl_workspace/model_repo.folder/llama_ord/     --data_path {}     --arth_model_path ./checkpoint/     --max_seq_len 512     --batch_size 1     --epochs {}     --warmup_epochs 2     --blr 9e-3     --weight_decay 0.02     --output_dir {} --device cuda"

ckpt_start = CHECKPOINT_START_SPLIT_EPOCH
for i in range(DATA_SPLIT_NUM * TOTAL_EPOCH_ORIGIN):
    data_path_now = DATA_SPLIT_FLODER + 'data_{}.json'.format(i % DATA_SPLIT_NUM)
    check_point_resume = CHECKPOINT_FOLDER + 'checkpoint-{}.pth'.format(ckpt_start)
    if ckpt_start >= 0:
        if i == 0:
            cmd_now = os_cmd.format(data_path_now, TOTAL_EPOCH_ORIGIN * DATA_SPLIT_NUM + EPOCH_BASE * DATA_SPLIT_NUM, CHECKPOINT_FOLDER, check_point_resume) + EVAL_APPEND
        else:
            cmd_now = os_cmd.format(data_path_now, TOTAL_EPOCH_ORIGIN * DATA_SPLIT_NUM + EPOCH_BASE * DATA_SPLIT_NUM, CHECKPOINT_FOLDER, check_point_resume)
    else:
        cmd_now = os_cmd_nocheckpoint.format(data_path_now, TOTAL_EPOCH_ORIGIN * DATA_SPLIT_NUM + EPOCH_BASE * DATA_SPLIT_NUM, CHECKPOINT_FOLDER)
    ckpt_start = ckpt_start + 1
    print("[FA WARPPER] execute:", cmd_now)
    time.sleep(2)
    os.system(cmd_now)
    check_point_resume_new = CHECKPOINT_FOLDER + 'checkpoint-{}.pth'.format(ckpt_start)
    if not os.path.exists(check_point_resume_new):
        ckpt_start = ckpt_start - 1
        print("[FA WARPPER] execute error, last epoch training failed. retry now")