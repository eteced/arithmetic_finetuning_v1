import sys
import os
import string
import time

DATA_SPLIT_FLODER='./data_split/'
DATA_SPLIT_NUM = 1456
TOTAL_EPOCH_ORIGIN = 30
# CHECKPOINT_FOLDER='./checkpoint/'
FINAL_CHECKPOINT='./checkpoint/'
CHECKPOINT_FOLDER='/mnt/tmp/'
# DAO 3087
# -2DAO 3870
# mid->hard 4256
# hard->old 4511
# old 4751 -> BLR = 0.03, 5times
# old->hard 5009 BLR = 0.03, 5times
# hard->new_h 5109, BLR = 0.03, 5times
# new_h->from1_3, 5240, BLR = 0.03, 5times
# from1_3 -> from1_20, 5617, BLR = 0.03, 5times
CHECKPOINT_START_SPLIT_EPOCH=5693
DATA_SET_OFFSET = 0
EPOCH_BASE = 0
# EVAL_APPEND = ' --eval True'
REAL_SAVE_CHECKPOINT = 10
EVAL_APPEND = ''
EARLY_LOOP = 5
LOOP_DATA_NUM = 1

os_cmd = "torchrun --nproc_per_node 1 finetuning_arthllama.py     --model Arth_Llama7B     --llama_model_path /home/eteced/dl_workspace/model_repo.folder/llama_ord/     --data_path {}     --arth_model_path ./checkpoint/     --max_seq_len 512     --batch_size 1     --epochs {}  --start_epoch {}  --warmup_epochs 2     --blr 0.03     --weight_decay 0.02     --output_dir {} --device cuda --resume {} "
os_cmd_nocheckpoint = "torchrun --nproc_per_node 1 finetuning_arthllama.py     --model Arth_Llama7B     --llama_model_path /home/eteced/dl_workspace/model_repo.folder/llama_ord/     --data_path {}     --arth_model_path ./checkpoint/     --max_seq_len 512     --batch_size 1     --epochs {}  --start_epoch {}  --warmup_epochs 2     --blr 0.03    --weight_decay 0.02     --output_dir {} --device cuda"

ckpt_start = CHECKPOINT_START_SPLIT_EPOCH
for i in range(DATA_SPLIT_NUM * TOTAL_EPOCH_ORIGIN):
    index_ord = CHECKPOINT_START_SPLIT_EPOCH + DATA_SET_OFFSET + i
    data_path_now = DATA_SPLIT_FLODER + 'data_{}.json'.format(((index_ord) % LOOP_DATA_NUM + ((index_ord // LOOP_DATA_NUM) // EARLY_LOOP) * LOOP_DATA_NUM) % DATA_SPLIT_NUM)
    check_point_resume = CHECKPOINT_FOLDER + 'checkpoint-{}.pth'.format(ckpt_start)
    check_point_resume_new = CHECKPOINT_FOLDER + 'checkpoint-{}.pth'.format((ckpt_start + 1))
    if ckpt_start >= 0:
        if i == 0:
            cmd_now = os_cmd.format(data_path_now, TOTAL_EPOCH_ORIGIN * DATA_SPLIT_NUM + EPOCH_BASE * DATA_SPLIT_NUM, CHECKPOINT_START_SPLIT_EPOCH + 1 + i, CHECKPOINT_FOLDER, check_point_resume) + EVAL_APPEND
        else:
            cmd_now = os_cmd.format(data_path_now, TOTAL_EPOCH_ORIGIN * DATA_SPLIT_NUM + EPOCH_BASE * DATA_SPLIT_NUM, CHECKPOINT_START_SPLIT_EPOCH + 1 + i, CHECKPOINT_FOLDER, check_point_resume)
    else:
        cmd_now = os_cmd_nocheckpoint.format(data_path_now, TOTAL_EPOCH_ORIGIN * DATA_SPLIT_NUM + EPOCH_BASE * DATA_SPLIT_NUM, CHECKPOINT_START_SPLIT_EPOCH + 1 + i, CHECKPOINT_FOLDER)
    ckpt_start = ckpt_start + 1
    print("[FA WARPPER] execute:", cmd_now)
    time.sleep(2)
    os.system(cmd_now)
    if not os.path.exists(check_point_resume_new):
        ckpt_start = ckpt_start - 1
        print("[FA WARPPER] execute error, last epoch training failed. retry now")
    else:
        # remove last checkpoint
        os.system("rm " + check_point_resume)
        if i % REAL_SAVE_CHECKPOINT == 0:
            os.system("cp " + check_point_resume_new + " " + FINAL_CHECKPOINT)