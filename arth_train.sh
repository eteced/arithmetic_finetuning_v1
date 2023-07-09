torchrun --nproc_per_node 16 arthmodel_train.py \
    --llama_model_path /home/eteced/dl_workspace/model_repo.folder/llama_ord/ \
    --batch_size 32 \
    --epochs 5 \
    --warmup_epochs 2 \
    --blr 0.1 \
    --weight_decay 0.02 \
    --output_dir ./checkpoint/
    # --blr 9e-3 \
