import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = '/home/zy/jjgong/CLUENER2020/bilstm_crf_pytorch/outputs/bilstm_crf/visualizations/training_stats.csv'
data = pd.read_csv(file_path)

# 选取 16 行，均匀分布
total_epochs = len(data)
step = total_epochs // 16
selected_indices = list(range(0, total_epochs, step))[:16]
selected_data = data.iloc[selected_indices]

# 重新编号
selected_data['epoch'] = range(1, len(selected_data) + 1)

# 绘制 train_loss 和 eval_loss 的曲线
plt.figure(figsize=(12, 6))
plt.plot(selected_data['epoch'], selected_data['train_loss'], label='Training Loss')
plt.plot(selected_data['epoch'], selected_data['eval_loss'], label='Evaluation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
plt.show()
