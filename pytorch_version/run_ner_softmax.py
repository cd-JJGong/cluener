import argparse
import glob
import logging
import os
import json

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME,BertConfig,AlbertConfig
from models.bert_for_ner import BertSoftmaxForNer
from models.albert_for_ner import AlbertSoftmaxForNer
from processors.utils_ner import CNerTokenizer,get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from datetime import datetime
import pandas as pd
from pathlib import Path

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertSoftmaxForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertSoftmaxForNer, CNerTokenizer),
}

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # 创建可视化保存目录
    vis_dir = Path(args.output_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 创建TensorBoard写入器
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))
    
    # 初始化训练数据收集器
    training_stats = {
        'epoch': [],
        'train_loss': [],
        'eval_loss': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'learning_rate': [],
        'train_time_epoch': [],
        'eval_time': [],
        'grad_norm': []
    }
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                collate_fn=collate_fn)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 准备优化器和学习率调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                              num_training_steps=t_total)
    # 初始化Early Stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_delta)
    
    # 训练循环
    global_step = 0
    tr_loss = 0.0
    best_f1 = 0.0
    model.zero_grad()
    
    for epoch in range(int(args.num_train_epochs)):
        # epoch级别的初始化
        epoch_start_time = datetime.now()
        epoch_loss = 0.0
        num_steps = 0
        total_grad_norm = 0.0
        
        pbar = ProgressBar(n_total=len(train_dataloader), desc=f'Training Epoch {epoch}')
        
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            
            outputs = model(**inputs)
            loss = outputs[0]
            
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            # 记录梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            total_grad_norm += grad_norm.item()
            
            epoch_loss += loss.item()
            num_steps += 1
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                # 记录学习率
                current_lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar('learning_rate', current_lr, global_step)
            
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
        
        # 在epoch结束时进行评估
        eval_start_time = datetime.now()
        results = evaluate(args, model, tokenizer)
        eval_time = (datetime.now() - eval_start_time).total_seconds()
        
        # 记录评估指标
        for key, value in results.items():
            tb_writer.add_scalar(f'eval/{key}', value, epoch)
        
        # 更新Early Stopping
        eval_f1 = results.get('f1', 0.0)
        early_stopping(eval_f1)
        
        # 收集统计数据
        training_stats['epoch'].append(epoch)
        training_stats['train_loss'].append(epoch_loss / num_steps)
        training_stats['eval_loss'].append(results['loss'])
        training_stats['precision'].append(results.get('precision', 0))
        training_stats['recall'].append(results.get('recall', 0))
        training_stats['f1_score'].append(eval_f1)
        training_stats['learning_rate'].append(scheduler.get_last_lr()[0])
        training_stats['train_time_epoch'].append((datetime.now() - epoch_start_time).total_seconds())
        training_stats['eval_time'].append(eval_time)
        training_stats['grad_norm'].append(total_grad_norm / num_steps)
        
        # Early Stopping 检查
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    # 确保有数据要可视化
    if len(training_stats['epoch']) > 0:
        # 创建DataFrame
        df_stats = pd.DataFrame(training_stats)
        
        # 确保可视化目录存在
        vis_dir = Path(args.output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 损失和评估指标图
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.plot(df_stats['epoch'], df_stats['train_loss'], label='Training Loss')
            plt.plot(df_stats['epoch'], df_stats['eval_loss'], label='Evaluation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Evaluation Loss')
            plt.legend()
            
            # 2. 精确率、召回率和F1分数
            plt.subplot(2, 2, 2)
            plt.plot(df_stats['epoch'], df_stats['precision'], label='Precision')
            plt.plot(df_stats['epoch'], df_stats['recall'], label='Recall')
            plt.plot(df_stats['epoch'], df_stats['f1_score'], label='F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Model Performance Metrics')
            plt.legend()
            
            # 3. 学习率变化
            plt.subplot(2, 2, 3)
            plt.plot(df_stats['epoch'], df_stats['learning_rate'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            
            # 4. 梯度范数变化
            plt.subplot(2, 2, 4)
            plt.plot(df_stats['epoch'], df_stats['grad_norm'])
            plt.xlabel('Epoch')
            plt.ylabel('Gradient Norm')
            plt.title('Average Gradient Norm')
            
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'training_metrics.png'))
            plt.close()
            
            # 5. 训练时间分析
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(df_stats['epoch'], df_stats['train_time_epoch'])
            plt.xlabel('Epoch')
            plt.ylabel('Seconds')
            plt.title('Training Time per Epoch')
            
            plt.subplot(1, 2, 2)
            plt.plot(df_stats['epoch'], df_stats['eval_time'])
            plt.xlabel('Epoch')
            plt.ylabel('Seconds')
            plt.title('Evaluation Time')
            
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'time_analysis.png'))
            plt.close()
            
            # 6. 相关性热力图
            plt.figure(figsize=(10, 8))
            correlation_matrix = df_stats.drop(['epoch'], axis=1).corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Metrics Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'correlation_heatmap.png'))
            plt.close()
            
            # 保存训练统计数据
            df_stats.to_csv(str(vis_dir / 'training_stats.csv'), index=False)
            
            logger.info(f"Successfully saved visualizations to {vis_dir}")
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
    else:
        logger.warning("No training statistics collected, skipping visualization")
    
    # 关闭TensorBoard写入器
    tb_writer.close()
    
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss.item()
            
        nb_eval_steps += 1
        preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        attention_mask = inputs['attention_mask'].cpu().numpy().tolist()
        
        for i, (label, mask) in enumerate(zip(out_label_ids, attention_mask)):
            temp_1 = []  # 真实标签
            temp_2 = []  # 预测标签
            
            for j, (m, label_id) in enumerate(zip(mask, label)):
                # 跳过 [CLS] 标记
                if j == 0:
                    continue
                    
                # 如果遇到 padding 或 [SEP] 标记，就停止处理
                if m == 0 or label_id == tokenizer.sep_token_id:
                    break
                    
                # 对于其他标记，添加到临时列表中
                try:
                    temp_1.append(args.id2label[label_id])
                    temp_2.append(args.id2label[preds[i][j]])
                except KeyError:
                    # 如果遇到未知标签，跳过该标签
                    continue
                    
            # 只有当有效标签不为空时才更新评估指标
            if temp_1 and temp_2:
                metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                
        pbar(step)
        
    print(' ')
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
        
    return results

def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    test_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []

    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:-1] # [CLS]XXXX[SEP]
        tags = [args.id2label[x] for x in preds]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags)
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    print(" ")
    output_predic_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(os.path.join(args.data_dir,"test.json"), 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = x['id']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file,test_submit)

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type=='train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                               else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token = tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens,all_label_ids)
    return dataset

def init_logger(log_file=None):
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置为 INFO 级别
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir",default=None,type=str,required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",)
    parser.add_argument("--model_type",default=None,type=str,required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path",default=None,type=str,required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),)
    parser.add_argument("--output_dir",default=None,type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    # Other parameters
    parser.add_argument('--markup',default='bios',type=str,choices=['bios','bio'])
    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'])
    parser.add_argument( "--labels",default="",type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",)
    parser.add_argument( "--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",default="",type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir",default="",type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--eval_max_seq_length",default=512,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                      help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument( "--max_steps", default=-1,type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints",action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16",action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level",type=str,default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html  ",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument('--ignore_sep_token', action='store_true', help='Whether to ignore [SEP] token during evaluation')
    parser.add_argument("--early_stopping_patience", default=5, type=int,
                      help="Number of epochs to wait before early stopping")
    parser.add_argument("--early_stopping_delta", default=1e-4, type=float,
                      help="Minimum change in F1 score to qualify as an improvement")
    parser.add_argument("--visualize_training", action="store_true",
                      help="Whether to create visualizations of training metrics")
    parser.add_argument("--log_level", default="INFO", type=str,
                      help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.getLogger().setLevel(log_level)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # 初始化日志
    logger = init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
    logger.setLevel(log_level)  # 确保使用命令行参数设置的日志级别
    
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script  
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank,device,args.n_gpu, bool(args.local_rank != -1),args.fp16,)
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,loss_type = args.loss_type,
                                          cache_dir=args.cache_dir if args.cache_dir else None,)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,)
    model = model_class.from_pretrained(args.model_name_or_path,from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,cache_dir=args.cache_dir if args.cache_dir else None,)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer,prefix=prefix)

if __name__ == "__main__":
    main()