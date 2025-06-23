from pathlib import Path

data_dir = Path("/home/zy/jjgong/CBLUE/CMeEE-V2/medical")
train_path = data_dir / 'train.json'
dev_path = data_dir / 'dev.json'
test_path = data_dir / 'test.json'
output_dir = Path("./outputs")

label2id = {
    "O": 0,
    "B-dis": 1, "I-dis": 2, "S-dis": 3,                  # 疾病相关标签
    "B-sym": 4, "I-sym": 5, "S-sym": 6,                  # 症状相关标签
    "B-dru": 7, "I-dru": 8, "S-dru": 9,                  # 药物相关标签
    "B-equ": 10, "I-equ": 11, "S-equ": 12,              # 设备/仪器相关标签
    "B-pro": 13, "I-pro": 14, "S-pro": 15,              # 检查方法相关标签
    "B-bod": 16, "I-bod": 17, "S-bod": 18,              # 身体部位相关标签
    "B-ite": 19, "I-ite": 20, "S-ite": 21,              # 其他生物实体相关标签
    "B-mic": 22, "I-mic": 23, "S-mic": 24,              # 微生物相关标签
    "B-dep": 25, "I-dep": 26, "S-dep": 27,              # 部门相关标签
    "B-procedure": 28, "I-procedure": 29, "S-procedure": 30, # 手术相关标签
    "<START>": 31,
    "<STOP>": 32
}