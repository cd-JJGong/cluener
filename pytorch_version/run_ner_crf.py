import glob
import logging
import os
import json
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from matplotlib import font_manager
import matplotlib
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bert_for_ner import BertCrfForNer
from models.albert_for_ner import AlbertCrfForNer
from processors.utils_ner import CNerTokenizer, get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.finetuning_argparse import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # 创建可视化保存目录
    vis_dir = Path(args.output_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 创建TensorBoard写入器
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))
    
    # 修改训练统计数据的初始化
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
        'grad_norm': []  # 添加梯度范数统计
    }
    
    # 初始化 Early Stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, 
                                 min_delta=args.early_stopping_delta)
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args.num_train_epochs)):
        epoch_start_time = datetime.now()
        epoch_loss = 0.0
        num_steps = 0
        total_grad_norm = 0.0
        
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            
            # 记录梯度范数
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            total_grad_norm += grad_norm.item()
            
            epoch_loss += loss.item()
            num_steps += 1
        
        # 在每个epoch结束时进行评估
        logger.info("\n开始第 %d 轮评估", _ + 1)
        eval_start_time = datetime.now()
        results = evaluate(args, model, tokenizer)
        eval_time = (datetime.now() - eval_start_time).total_seconds()
        logger.info("第 %d 轮评估完成，耗时: %.2f 秒", _ + 1, eval_time)
        
        # 更新训练统计数据
        training_stats['epoch'].append(_)
        training_stats['train_loss'].append(epoch_loss / num_steps)
        training_stats['eval_loss'].append(results['loss'])
        training_stats['precision'].append(results.get('precision', 0))
        training_stats['recall'].append(results.get('recall', 0))
        training_stats['f1_score'].append(results.get('f1', 0))
        training_stats['learning_rate'].append(scheduler.get_last_lr()[0])
        training_stats['train_time_epoch'].append((datetime.now() - epoch_start_time).total_seconds())
        training_stats['eval_time'].append(eval_time)
        training_stats['grad_norm'].append(total_grad_norm / num_steps)
        
        # 更新Early Stopping
        eval_f1 = results.get('f1', 0.0)
        early_stopping(eval_f1)
        
        # Early Stopping 检查
        if early_stopping.early_stop:
            logger.info("触发早停机制，停止训练")
            break
        
        logger.info("\n第 %d 轮训练完成，平均训练损失: %.4f，评估F1分数: %.4f", 
                   _ + 1, epoch_loss / num_steps, eval_f1)
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    
    # 在训练结束后添加可视化代码
    if len(training_stats['epoch']) > 0 and args.visualize_training:
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者其他支持中文的字体
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建 DataFrame 以方便绘图和保存
            df_stats = pd.DataFrame({
                'epoch': training_stats['epoch'],
                'train_loss': training_stats['train_loss'],
                'eval_loss': training_stats['eval_loss'],
                'precision': training_stats['precision'],
                'recall': training_stats['recall'],
                'f1_score': training_stats['f1_score'],
                'learning_rate': training_stats['learning_rate'],
                'grad_norm': training_stats['grad_norm']
            })
            
            # 保存详细的训练统计数据到CSV
            df_stats.to_csv(str(vis_dir / 'training_metrics.csv'), index=False)
            
            # 绘制训练指标图
            plt.figure(figsize=(15, 10))
            
            # 1. 损失图
            plt.subplot(2, 2, 1)
            plt.plot(df_stats['epoch'], df_stats['train_loss'], label='Training Loss')
            plt.plot(df_stats['epoch'], df_stats['eval_loss'], label='Evaluation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Evaluation Loss')
            plt.legend()
            plt.grid(True)
            
            # 2. 精确率、召回率和F1分数
            plt.subplot(2, 2, 2)
            plt.plot(df_stats['epoch'], df_stats['precision'], label='Precision')
            plt.plot(df_stats['epoch'], df_stats['recall'], label='Recall')
            plt.plot(df_stats['epoch'], df_stats['f1_score'], label='F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Model Performance Metrics')
            plt.legend()
            plt.grid(True)
            
            # 3. 学习率变化
            plt.subplot(2, 2, 3)
            plt.plot(df_stats['epoch'], df_stats['learning_rate'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            
            # 4. 梯度范数变化
            plt.subplot(2, 2, 4)
            plt.plot(df_stats['epoch'], df_stats['grad_norm'])
            plt.xlabel('Epoch')
            plt.ylabel('Gradient Norm')
            plt.title('Average Gradient Norm')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'training_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"成功保存训练指标可视化结果到 {vis_dir}")
            
        except Exception as e:
            logger.error(f"训练指标可视化过程发生错误: {str(e)}")
    
    # 关闭TensorBoard写入器
    tb_writer.close()
    
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    
    if isinstance(model, nn.DataParallel):
        model = model.module
        
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = inputs['input_lens'].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
        pbar(step)
        
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    
    # 只使用在 entity_info 中存在的实体
    entities = [entity for entity in args.id2label.values() 
               if entity != "O" and entity in entity_info]
    
    if not entities:  # 如果没有有效的实体
        results = {
            'loss': eval_loss,
            'precision': eval_info['acc'],
            'recall': eval_info['recall'],
            'f1': eval_info['f1'],
            'entity_info': entity_info,
            'micro_avg': {'precision': 0, 'recall': 0, 'f1': 0},
            'macro_avg': {'precision': 0, 'recall': 0, 'f1': 0},
            'weighted_avg': {'precision': 0, 'recall': 0, 'f1': 0}
        }
    else:
        # 计算样本总数
        n_samples = sum(entity_info[entity]['tp'] + entity_info[entity]['fn'] 
                       for entity in entities)
        
        # Micro average
        total_tp = sum(entity_info[entity]['tp'] for entity in entities)
        total_fp = sum(entity_info[entity]['fp'] for entity in entities)
        total_fn = sum(entity_info[entity]['fn'] for entity in entities)
        
        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
                  if micro_precision + micro_recall > 0 else 0
        
        # Macro average
        macro_precision = sum(entity_info[entity]['precision'] for entity in entities) / len(entities)
        macro_recall = sum(entity_info[entity]['recall'] for entity in entities) / len(entities)
        macro_f1 = sum(entity_info[entity]['f1'] for entity in entities) / len(entities)
        
        # Weighted average
        weighted_precision = sum(entity_info[entity]['precision'] * 
                               (entity_info[entity]['tp'] + entity_info[entity]['fn']) 
                               for entity in entities) / n_samples if n_samples > 0 else 0
        weighted_recall = sum(entity_info[entity]['recall'] * 
                            (entity_info[entity]['tp'] + entity_info[entity]['fn']) 
                            for entity in entities) / n_samples if n_samples > 0 else 0
        weighted_f1 = sum(entity_info[entity]['f1'] * 
                         (entity_info[entity]['tp'] + entity_info[entity]['fn']) 
                         for entity in entities) / n_samples if n_samples > 0 else 0
        
        results = {
            'loss': eval_loss,
            'precision': eval_info['acc'],
            'recall': eval_info['recall'],
            'f1': eval_info['f1'],
            'entity_info': entity_info,
            'micro_avg': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1': micro_f1
            },
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            }
        }
    
    # 输出评估结果
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items() 
                    if not isinstance(value, dict)])
    logger.info(info)
    
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    
    return results


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    if args.task_name == 'cluener':
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
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
    cached_features_file = os.path.join(args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
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
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                    else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
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
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def main():
    args = get_argparse().parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
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
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
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
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
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
            model = model_class.from_pretrained(checkpoint, config=config)
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
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)

if __name__ == "__main__":
    main()
