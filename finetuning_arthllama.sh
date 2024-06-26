torchrun --nproc_per_node 1 finetuning_arthllama.py \
    --model Arth_Llama7B \
    --llama_model_path /home/eteced/dl_workspace/model_repo.folder/llama_ord/ \
    --data_path /home/eteced/dl_workspace/stanford_alpaca/alpaca_data_debug.json \
    --arth_model_path ./checkpoint/ \
    --max_seq_len 512 \
    --batch_size 1 \
    --epochs 5 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./checkpoint/
