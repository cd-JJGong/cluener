CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export GLUE_DIR=/home/zy/jjgong/CBLUE/CMeEE-V2
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="medical"

# 禁用 matplotlib 和 PIL 的调试日志
export PYTHONWARNINGS="ignore"
export MPLBACKEND="Agg"

python run_ner_softmax.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=30.0 \
  --logging_steps=224 \
  --save_steps=224 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed=42 \
  --weight_decay=0.01 \
  --max_grad_norm=0.5 \
  --early_stopping_patience=5 \
  --early_stopping_delta=1e-4 \
  --visualize_training \
  --ignore_sep_token \
  --log_level=INFO

# 预测脚本
# python run_ner_softmax.py \
#   --model_type=bert \
#   --model_name_or_path=$BERT_BASE_DIR \
#   --task_name=$TASK_NAME \
#   --do_predict \
#   --do_lower_case \
#   --loss_type=ce \
#   --data_dir=$GLUE_DIR/${TASK_NAME}/ \
#   --train_max_seq_length=128 \
#   --eval_max_seq_length=512 \
#   --per_gpu_train_batch_size=24 \
#   --per_gpu_eval_batch_size=24 \
#   --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#   --overwrite_output_dir \
#   --seed=42 \
#   --weight_decay=0.01 \
#   --visualize_training \
#   --ignore_sep_token