import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 新的标签数据分布（请根据实际分布数量填写，这里先用1占位）
label_distribution = {
    "bod": 1,
    "dep": 1,
    "dis": 1,
    "dru": 1,
    "equ": 1,
    "ite": 1,
    "mic": 1,
    "pro": 1,
    "sym": 1
}

subjects = list(label_distribution.keys())

# 读取日志文件
txt_file = '/home/zy/jjgong/CLUENER2020/bilstm_crf_pytorch/train.txt'

# 提取每个epoch的数据
epochs = []
acc_data = {subject: [] for subject in subjects}
recall_data = {subject: [] for subject in subjects}
f1_data = {subject: [] for subject in subjects}

with open(txt_file, 'r', encoding='utf-8') as f:
    txt_content = f.read()

print("开始解析日志文件...")
print(f"日志文件总长度: {len(txt_content)} 字符")

# 提取所有epoch的评估块
epoch_pattern = r'\*\*\*\*\* Eval results  \*\*\*\*\*\n.*?loss: ([\d.]+).*?\*\*\*\*\* Entity results  \*\*\*\*\*(.*?)(?=\*\*\*\*\* Eval results|\Z)'
epoch_matches = list(re.finditer(epoch_pattern, txt_content, re.DOTALL))
print(f"找到 {len(epoch_matches)} 个评估块")

epoch_count = 0
for match in epoch_matches:
    epoch_count += 1
    print(f"\n处理第 {epoch_count} 个评估块:")
    print(f"Loss: {match.group(1)}")
    
    epoch_content = match.group(2)
    print(f"实体评估内容长度: {len(epoch_content)} 字符")
    
    epochs.append(epoch_count)
    
    # 对每个实体类型提取指标
    for subject in subjects:
        # 适配日志中每个实体评估结果分为两行且前缀为任意内容的格式
        subject_pattern = (
            rf'{subject} results \*+\s*\n.*?acc: ([\d.]+) - recall: ([\d.]+) - f1: ([\d.]+)'
        )
        subject_match = re.search(subject_pattern, epoch_content, re.DOTALL)
        if subject_match:
            acc, recall, f1 = map(float, subject_match.groups())
            print(f"{subject}: acc={acc:.4f}, recall={recall:.4f}, f1={f1:.4f}")
            acc_data[subject].append(acc)
            recall_data[subject].append(recall)
            f1_data[subject].append(f1)
        else:
            print(f"{subject}: 未找到匹配")
            acc_data[subject].append(np.nan)
            recall_data[subject].append(np.nan)
            f1_data[subject].append(np.nan)

print(f"\n总共处理了 {len(epochs)} 个epoch")
print(f"每个实体的数据点数量:")
for subject in subjects:
    print(f"{subject}: {len(acc_data[subject])} 个数据点")

# 创建输出目录
output_dir = 'visualization_results'
os.makedirs(output_dir, exist_ok=True)

def plot_metrics(epochs, data, metric_name, output_path):
    plt.figure(figsize=(12, 6))
    for subject in subjects:
        plt.plot(epochs, data[subject], label=subject, marker='o', markersize=3)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} for Each Entity over Epochs')
    plt.legend(title='Entity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 绘制每个指标的图表
plot_metrics(epochs, acc_data, 'Accuracy', f'{output_dir}/accuracy.png')
plot_metrics(epochs, recall_data, 'Recall', f'{output_dir}/recall.png')
plot_metrics(epochs, f1_data, 'F1 Score', f'{output_dir}/f1_score.png')

# 计算平均值
def calculate_averages(data, label_distribution):
    total_samples = sum(label_distribution.values())
    n_classes = len(label_distribution)
    micro_avgs = []
    macro_avgs = []
    weighted_avgs = []
    
    for i in range(len(epochs)):
        current_metrics = [data[subject][i] for subject in subjects if not np.isnan(data[subject][i])]
        weights = [label_distribution[subject] / total_samples for subject in subjects if not np.isnan(data[subject][i])]
        
        macro_avg = np.mean(current_metrics)
        weighted_avg = np.sum(np.multiply(current_metrics, weights))
        micro_avg = np.sum(current_metrics) / len(current_metrics)
        
        macro_avgs.append(macro_avg)
        weighted_avgs.append(weighted_avg)
        micro_avgs.append(micro_avg)
    
    return micro_avgs, macro_avgs, weighted_avgs

# 计算每个指标的平均值
acc_micro, acc_macro, acc_weighted = calculate_averages(acc_data, label_distribution)
recall_micro, recall_macro, recall_weighted = calculate_averages(recall_data, label_distribution)
f1_micro, f1_macro, f1_weighted = calculate_averages(f1_data, label_distribution)

# 绘制平均值的变化情况
def plot_averages(epochs, micro, macro, weighted, metric_name, output_path):
    plt.figure(figsize=(12, 6))
    # plt.plot(epochs, micro, label='Micro Average', marker='o', markersize=3)
    plt.plot(epochs, macro, label='Macro Average', marker='s', markersize=3)
    # plt.plot(epochs, weighted, label='Weighted Average', marker='^', markersize=3)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'Average {metric_name} over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 绘制每个指标的平均值图表
plot_averages(epochs, acc_micro, acc_macro, acc_weighted, 'Accuracy', f'{output_dir}/average_accuracy.png')
plot_averages(epochs, recall_micro, recall_macro, recall_weighted, 'Recall', f'{output_dir}/average_recall.png')
plot_averages(epochs, f1_micro, f1_macro, f1_weighted, 'F1 Score', f'{output_dir}/average_f1_score.png')

print(f"指标可视化结果已保存到 {output_dir} 目录")