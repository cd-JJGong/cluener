from torch.nn import LayerNorm
import torch.nn as nn
from crf import CRF
import matplotlib.pyplot as plt
import pandas as pd

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NERModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,
                 label2id,device,drop_p = 0.1):
        super(NERModel, self).__init__()
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,
                              batch_first=True,num_layers=2,dropout=drop_p,
                              bidirectional=True)
        self.dropout = SpatialDropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2,len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def forward(self, inputs_ids, input_mask):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output= self.layer_norm(seqence_output)
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features

def train(args, train_dataset, model, tokenizer):
    # ... 现有的训练代码 ...
    
    # 确保有数据要可视化
    if len(training_stats['epoch']) > 0:
        # 创建DataFrame
        df_stats = pd.DataFrame(training_stats)
        
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