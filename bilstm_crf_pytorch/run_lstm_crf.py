import json
import torch
import argparse
import torch.nn as nn
from torch import optim
import config
from model import NERModel
from dataset_loader import DatasetLoader
from progressbar import ProgressBar
from ner_metrics import SeqEntityScore
from data_processor import CluenerProcessor, MedicalNerProcessor
from lr_scheduler import ReduceLROnPlateau
from utils_ner import get_entities
from common import (init_logger,
                    logger,
                    json_to_text,
                    load_model,
                    AverageMeter,
                    seed_everything)
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import font_manager
import matplotlib


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
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train(args, model, processor):
    # 创建可视化保存目录
    vis_dir = Path(args.output_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
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
        'eval_time': []
    }
    
    # 初始化 Early Stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, 
                                 min_delta=args.early_stopping_delta)
    
    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, sort=True,
                                vocab=processor.vocab, label2id=args.label2id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    best_f1 = 0
    
    for epoch in range(1, 1 + args.epochs):
        epoch_start_time = datetime.now()
        print(f"Epoch {epoch}/{args.epochs}")
        # pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        
        for step, batch in enumerate(train_loader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            # pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        
        print(" ")
        train_log = {'loss': train_loss.avg}
        
        # 评估开始时间
        eval_start_time = datetime.now()
        eval_log, class_info = evaluate(args, model, processor)
        eval_time = (datetime.now() - eval_start_time).total_seconds()
        
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        
        # 收集训练统计数据
        training_stats['epoch'].append(epoch)
        training_stats['train_loss'].append(train_loss.avg)
        training_stats['eval_loss'].append(eval_log['eval_loss'])
        training_stats['precision'].append(eval_log['eval_acc'])
        training_stats['recall'].append(eval_log['eval_recall'])
        training_stats['f1_score'].append(eval_log['eval_f1'])
        training_stats['learning_rate'].append(optimizer.param_groups[0]['lr'])
        training_stats['train_time_epoch'].append((datetime.now() - epoch_start_time).total_seconds())
        training_stats['eval_time'].append(eval_time)
        
        scheduler.epoch_step(logs['eval_f1'], epoch)
        
        # 在评估后添加 Early Stopping 检查
        eval_f1 = eval_log['eval_f1']
        early_stopping(eval_f1)
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered! 验证集性能已经连续5个epoch未提升。")
            break
        
        logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
        logger.info("save model to disk.")
        best_f1 = logs['eval_f1']
        # if isinstance(model, nn.DataParallel):
        #     model_stat_dict = model.module.state_dict()
        # else:
        #     model_stat_dict = model.state_dict()
        # state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
        # if logs['eval_f1'] > best_f1:
            
            # model_path = args.output_dir / 'best-model.bin'
            # torch.save(state, str(model_path))
        print("Eval Entity Score: ")
        for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)
    
    # 训练结束后生成可视化
    if len(training_stats['epoch']) > 0:
        # 设置中文字体
        try:
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False
            
            df_stats = pd.DataFrame(training_stats)
            
            # 1. 损失和评估指标图
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.plot(df_stats['epoch'], df_stats['train_loss'], label='Training Loss')
            plt.plot(df_stats['epoch'], df_stats['eval_loss'], label='Evaluation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Evaluation Loss')
            plt.legend()
            
            # 2. 准确率、召回率和F1分数
            plt.subplot(2, 2, 2)
            plt.plot(df_stats['epoch'], df_stats['precision'], label='precision')
            plt.plot(df_stats['epoch'], df_stats['recall'], label='recall')
            plt.plot(df_stats['epoch'], df_stats['f1_score'], label='f1_score')
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
            
            # 4. 训练和评估时间
            plt.subplot(2, 2, 4)
            plt.plot(df_stats['epoch'], df_stats['train_time_epoch'], label='train_time_epoch')
            plt.plot(df_stats['epoch'], df_stats['eval_time'], label='eval_time')
            plt.xlabel('Epoch')
            plt.ylabel('Seconds')
            plt.title('Training and Evaluation Time')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'training_metrics.png'))
            plt.close()
            
            # 5. 相关性热力图
            plt.figure(figsize=(10, 8))
            correlation_matrix = df_stats.drop(['epoch'], axis=1).corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap of Metrics')
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'correlation_heatmap.png'))
            plt.close()
            
            # 保存训练统计数据到CSV
            df_stats.to_csv(str(vis_dir / 'training_stats.csv'), index=False)
            
            logger.info(f"成功保存可视化结果到 {vis_dir}")
        except Exception as e:
            logger.error(f"可视化过程发生错误: {str(e)}")
    else:
        logger.warning("没有收集到训练统计数据，跳过可视化")

def evaluate(args,model,processor):
    eval_dataset = load_and_cache_examples(args,processor, data_type='dev')
    eval_dataloader = DatasetLoader(data=eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=False,
                                 vocab=processor.vocab, label2id=args.label2id)
    # pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tags, label_paths=target)
            # pbar(step=step)
    print(" ")
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info

def predict(args,model,processor):
    model_path = args.output_dir / 'best-model.bin'
    model = load_model(model, model_path=str(model_path))
    test_data = []
    with open(str(args.data_dir / "test.json"), 'r') as f:
        idx = 0
        for line in f:
            json_d = {}
            line = json.loads(line.strip())
            text = line['text']
            words = list(text)
            labels = ['O'] * len(words)
            json_d['id'] = idx
            json_d['context'] = " ".join(words)
            json_d['tag'] = " ".join(labels)
            json_d['raw_context'] = "".join(words)
            idx += 1
            test_data.append(json_d)
    # pbar = ProgressBar(n_total=len(test_data))
    results = []
    for step, line in enumerate(test_data):
        token_a = line['context'].split(" ")
        input_ids = [processor.vocab.to_index(w) for w in token_a]
        input_mask = [1] * len(token_a)
        input_lens = [len(token_a)]
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            input_lens = torch.tensor([input_lens], dtype=torch.long)
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            features = model.forward_loss(input_ids, input_mask, input_lens, input_tags=None)
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
        label_entities = get_entities(tags[0], args.id2label)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags[0])
        json_d['entities'] = label_entities
        results.append(json_d)
        # pbar(step=step)
    print(" ")
    output_predic_file = str(args.output_dir / "test_prediction.json")
    output_submit_file = str(args.output_dir / "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(str(args.data_dir / 'test.json'), 'r') as fr:
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
    json_to_text(output_submit_file, test_submit)

def load_and_cache_examples(args,processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_examples_file = args.data_dir / 'cached_crf-{}_{}_{}'.format(
        data_type,
        args.arch,
        str(args.task_name))
    if cached_examples_file.exists():
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        logger.info("Saving features into cached file %s", cached_examples_file)
        torch.save(examples, str(cached_examples_file))
    return examples

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')

    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'])
    parser.add_argument("--arch",default='bilstm_crf',type=str)
    parser.add_argument('--learning_rate',default=0.0001,type=float)
    parser.add_argument('--seed',default=1234,type=int)
    parser.add_argument('--gpu',default='0',type=str)
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--embedding_size',default=128,type=int)
    parser.add_argument('--hidden_size',default=384,type=int)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--task_name", type=str, default='ner')
    
    # 添加可视化相关参数
    parser.add_argument("--visualize_training", action="store_true",
                      help="是否创建训练过程的可视化图表")

    # 添加 Early Stopping 相关参数
    parser.add_argument('--early_stopping_patience', 
                       default=5, 
                       type=int,
                       help='Early Stopping 的耐心值，即在多少个epoch验证集性能未提升后停止训练')
    parser.add_argument('--early_stopping_delta', 
                       default=0.0, 
                       type=float,
                       help='判定性能提升的最小变化阈值')

    args = parser.parse_args()
    args.data_dir = config.data_dir
    if not config.output_dir.exists():
        args.output_dir.mkdir()
    args.output_dir = config.output_dir / '{}'.format(args.arch)
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    init_logger(log_file=str(args.output_dir / '{}-{}.log'.format(args.arch, args.task_name)))
    seed_everything(args.seed)
    if args.gpu!='':
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
    
    # 修改为医学NER的标签体系
    args.id2label = {0: 'O', 
                     1: 'B-dis', 2: 'I-dis', 3: 'S-dis',                  # 疾病相关标签
                     4: 'B-sym', 5: 'I-sym', 6: 'S-sym',                  # 症状相关标签
                     7: 'B-dru', 8: 'I-dru', 9: 'S-dru',                  # 药物相关标签
                     10: 'B-equ', 11: 'I-equ', 12: 'S-equ',              # 设备/仪器相关标签
                     13: 'B-pro', 14: 'I-pro', 15: 'S-pro',              # 检查方法相关标签
                     16: 'B-bod', 17: 'I-bod', 18: 'S-bod',              # 身体部位相关标签
                     19: 'B-ite', 20: 'I-ite', 21: 'S-ite',              # 其他生物实体相关标签
                     22: 'B-mic', 23: 'I-mic', 24: 'S-mic',              # 微生物相关标签
                     25: 'B-dep', 26: 'I-dep', 27: 'S-dep',              # 部门相关标签
                     28: 'B-procedure', 29: 'I-procedure', 30: 'S-procedure', # 手术相关标签
                     31: '<START>', 32: '<STOP>'}
    args.label2id = {v: k for k, v in args.id2label.items()}
    
    # 使用 MedicalNerProcessor 替代 CluenerProcessor
    processor = MedicalNerProcessor(data_dir=config.data_dir)
    processor.get_vocab()
    model = NERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
                     hidden_size=args.hidden_size,device=args.device,label2id=args.label2id)
    model.to(args.device)
    if args.do_train:
        train(args,model,processor)
    if args.do_eval:
        model_path = args.output_dir / 'best-model.bin'
        model = load_model(model, model_path=str(model_path))
        evaluate(args,model,processor)
    if args.do_predict:
        predict(args,model,processor)

if __name__ == "__main__":
    main()
