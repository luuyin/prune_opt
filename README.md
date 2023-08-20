# Prune opt model to 0.1 Sparsity using global magnitude pruning


for model in 125m
do
    for sparsity in  0.1
    do
    python  run_clm_no_trainer_sparse.py \
        --freeze_weights \
        --dataset_name wikitext \
        --dataset_config_name wikitext-103-raw-v1 \
        --model_name_or_path facebook/opt-$model \
        --output_dir ./tmp/test \
        --sparse_init  one_shot_gm  \
        --sparsity $sparsity 
    done
done
