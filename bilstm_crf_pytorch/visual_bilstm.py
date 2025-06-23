import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 实体类型列表
subjects = ["bod", "dep", "dis", "dru", "equ", "ite", "mic", "pro", "sym"]

# 读取日志文件
txt_file = '/home/zy/jjgong/CLUENER2020/bilstm_crf_pytorch/train.txt'

# 提取每个epoch的数据
epochs = []
acc_data = {subject: [] for subject in subjects}
recall_data = {subject: [] for subject in subjects}
f1_data = {subject: [] for subject in subjects}

with open(txt_file, 'r', encoding='utf-8') as f:
    txt_content = f.read()

print("Parsing log file...")
print(f"Log file length: {len(txt_content)} characters")

# 提取所有epoch的评估块
epoch_pattern = r'Epoch (\d+)/\d+\n.*?Eval Entity Score: \n(.*?)(?=Epoch \d+/\d+|\Z)'
epoch_matches = list(re.finditer(epoch_pattern, txt_content, re.DOTALL))
print(f"Found {len(epoch_matches)} evaluation blocks")

for match in epoch_matches:
    epoch_num = int(match.group(1))
    epoch_content = match.group(2)
    print(f"\nProcessing epoch {epoch_num}")
    
    epochs.append(epoch_num)
    
    # 对每个实体类型提取指标
    for subject in subjects:
        # 匹配新的日志格式
        subject_pattern = rf'Subject: {subject} - Acc: ([\d.]+) - Recall: ([\d.]+) - F1: ([\d.]+)'
        subject_match = re.search(subject_pattern, epoch_content)
        if subject_match:
            acc, recall, f1 = map(float, subject_match.groups())
            print(f"{subject}: acc={acc:.4f}, recall={recall:.4f}, f1={f1:.4f}")
            acc_data[subject].append(acc)
            recall_data[subject].append(recall)
            f1_data[subject].append(f1)
        else:
            print(f"{subject}: No match found")
            acc_data[subject].append(np.nan)
            recall_data[subject].append(np.nan)
            f1_data[subject].append(np.nan)

print(f"\nProcessed {len(epochs)} epochs")
print(f"Data points per entity:")
for subject in subjects:
    print(f"{subject}: {len(acc_data[subject])} points")

# Create output directory
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

# Plot metrics
plot_metrics(epochs, acc_data, 'Accuracy', f'{output_dir}/accuracy.png')
plot_metrics(epochs, recall_data, 'Recall', f'{output_dir}/recall.png')
plot_metrics(epochs, f1_data, 'F1 Score', f'{output_dir}/f1_score.png')

# Calculate averages
def calculate_averages(data):
    macro_avgs = []
    micro_avgs = []
    
    for i in range(len(epochs)):
        current_metrics = [data[subject][i] for subject in subjects if not np.isnan(data[subject][i])]
        macro_avg = np.mean(current_metrics)
        micro_avg = np.sum(current_metrics) / len(current_metrics)
        
        macro_avgs.append(macro_avg)
        micro_avgs.append(micro_avg)
    
    return micro_avgs, macro_avgs

# Calculate averages for each metric
acc_micro, acc_macro = calculate_averages(acc_data)
recall_micro, recall_macro = calculate_averages(recall_data)
f1_micro, f1_macro = calculate_averages(f1_data)

# Plot averages
def plot_averages(epochs, micro, macro, metric_name, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, macro, label='Macro Average', marker='s', markersize=3)
    # plt.plot(epochs, micro, label='Micro Average', marker='o', markersize=3)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'Average {metric_name} over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Plot average metrics
plot_averages(epochs, acc_micro, acc_macro, 'Accuracy', f'{output_dir}/average_accuracy.png')
plot_averages(epochs, recall_micro, recall_macro, 'Recall', f'{output_dir}/average_recall.png')
plot_averages(epochs, f1_micro, f1_macro, 'F1 Score', f'{output_dir}/average_f1_score.png')

print(f"Visualization results saved to {output_dir} directory")