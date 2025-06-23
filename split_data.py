import json
import random
import os
from typing import List, Dict, Any

def split_json_file(input_file: str, train_ratio: float = 0.8) -> None:
    """
    将JSON文件按指定比例分割成训练集和验证集
    
    Args:
        input_file: 输入JSON文件的路径
        train_ratio: 训练集比例，默认为0.8
    """
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 将数据转换为列表形式
    if isinstance(data, dict):
        data = [data]
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算分割点
    split_point = int(len(data) * train_ratio)
    
    # 分割数据
    train_data = data[:split_point]
    val_data = data[split_point:]
    
    # 生成输出文件名
    base_name = os.path.splitext(input_file)[0]
    train_file = f"{base_name}_train.json"
    val_file = f"{base_name}_val.json"
    
    # 保存训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据集分割完成：")
    print(f"训练集大小：{len(train_data)}，保存至：{train_file}")
    print(f"验证集大小：{len(val_data)}，保存至：{val_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将JSON文件按比例分割成训练集和验证集')
    parser.add_argument('input_file', help='输入JSON文件的路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例，默认为0.8')
    
    args = parser.parse_args()
    split_json_file(args.input_file, args.train_ratio) 