


### Prune opt model 

#### Prune the pre-trained opt-125m to 0.1 sparsity using global magnitude pruning, then fine-tune the pruned model


```
for model in 125m
do
    for sparsity in  0.1
    do
    python  run_clm_no_trainer_sparse.py \
        --freeze_weights \
        --noembed \
        --dataset_name wikitext \
        --dataset_config_name wikitext-103-raw-v1 \
        --model_name_or_path facebook/opt-$model \
        --output_dir ./tmp/test \
        --sparse_init  one_shot_gm  \
        --sparsity $sparsity 
    done
done

```

###  LMC on roberta-large model

#### indicate the checkpoint paths of two models using `sparse_path` and `dense_path`

```

sparse_path= sparse_path
dense_path=  dense_path


for seed in 41
do
  for TASK_NAME in qnli  
  do 
    for sparse_path in sparse_path
    do
      for validation_split_percentage in 100
      do
      python LMC.py \
        --noembed \
        --sparse_path $sparse_path \
        --dense_path $dense_path \
        --sparsity $sparsity \
        --model_name_or_path roberta-large \
        --task_name $TASK_NAME \
        --max_length 512 \
        --per_device_train_batch_size 16 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --seed $seed 
      done
    done
  done
done






```
