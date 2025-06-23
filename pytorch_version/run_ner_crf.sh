CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export GLUE_DIR=/home/zy/jjgong/CBLUE/CMeEE-V2/
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="medical"

python run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=7e-5 \
  --num_train_epochs=20.0 \
  --logging_steps=448 \
  --save_steps=448 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output1/ \
  --overwrite_output_dir \
  --seed=42 \
  --early_stopping_patience=5 \
  --early_stopping_delta=1e-4 \
  --visualize_training \
  --weight_decay=0.01

# python run_ner_crf.py \
#   --model_type=bert \
#   --model_name_or_path=$BERT_BASE_DIR \
#   --task_name=$TASK_NAME \
#   --do_predict \
#   --do_lower_case \
#   --data_dir=$GLUE_DIR/${TASK_NAME}/ \
#   --train_max_seq_length=128 \
#   --eval_max_seq_length=512 \
#   --per_gpu_train_batch_size=24 \
#   --per_gpu_eval_batch_size=24 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=3.0 \
#   --logging_steps=448 \
#   --save_steps=448 \
#   --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#   --overwrite_output_dir \
#   --seed=42 \
#   --early_stopping_patience=5 \
#   --early_stopping_delta=1e-4 \
#   --weight_decay=0.01