import json

# 读取第一个 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 转换格式
def convert_format(data):
    result = []
    for item in data:
        text = item["text"]
        entities = item["entities"]
        
        # 初始化 label 字典
        label = {}
        
        # 遍历 entities，按 type 分类并转换格式
        for entity in entities:
            entity_type = entity["type"]
            entity_name = entity["entity"]
            start_idx = entity["start_idx"]
            end_idx = entity["end_idx"]
            
            # 将 type 转换为第二个文件的格式
            if entity_type == "pro":
                label_type = "procedure"
            elif entity_type == "dis":
                label_type = "disease"
            else:
                label_type = entity_type  # 如果有其他类型，直接使用
            
            # 如果 label_type 不存在于 label 中，初始化为空字典
            if label_type not in label:
                label[label_type] = {}
            
            # 如果 entity_name 不存在于 label[label_type] 中，初始化为空列表
            if entity_name not in label[label_type]:
                label[label_type][entity_name] = []
            
            # 添加位置信息
            label[label_type][entity_name].append([start_idx, end_idx])
        
        # 将转换后的结果加入到 result 中
        result.append({
            "text": text,
            "label": label
        })
    
    return result

# 保存转换后的结果到新的 JSON 文件
def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')  # 每个 JSON 对象占一行

# 主程序
if __name__ == "__main__":
    # 输入文件路径
    input_file_path = "/home/zy/jjgong/CBLUE/CMeEE-V2/train.json"  # 替换为你的第一个 JSON 文件路径
    # 输出文件路径
    output_file_path = "/home/zy/jjgong/CBLUE/CMeEE-V2/cluener/train.json"  # 替换为你希望保存的输出文件路径
    
    # 读取第一个 JSON 文件
    data1 = read_json_file(input_file_path)
    
    # 转换格式
    converted_data = convert_format(data1)
    
    # 保存转换后的结果到新的 JSON 文件
    save_json_file(converted_data, output_file_path)
    
    print(f"转换完成，结果已保存到 {output_file_path}")