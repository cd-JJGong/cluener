""" Named entity recognition fine-tuning: utilities to work with medical NER task. """
import os
import json
import chardet
from .utils_ner import DataProcessor, InputExample

class MedicalNerProcessor(DataProcessor):
    """Processor for the medical NER data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "test.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "test-%s" % i
            text = line['text']
            words = list(text)
            labels = ['O'] * len(words)
            examples.append(InputExample(guid=guid, text_a=words, labels=labels))
        return examples

    def get_labels(self):
        """See base class."""
        return ["O",
            "B-dis", "I-dis", "S-dis",                  # 疾病相关标签
            "B-sym", "I-sym", "S-sym",                  # 症状相关标签
            "B-dru", "I-dru", "S-dru",                  # 药物相关标签
            "B-equ", "I-equ", "S-equ",                  # 设备/仪器相关标签
            "B-pro", "I-pro", "S-pro",                  # 检查方法相关标签
            "B-bod", "I-bod", "S-bod",                  # 身体部位相关标签
            "B-ite", "I-ite", "S-ite",                  # 其他生物实体相关标签
            "B-mic", "I-mic", "S-mic",                  # 微生物相关标签
            "B-dep", "I-dep", "S-dep",                  # 部门相关标签
            "B-procedure", "I-procedure", "S-procedure", # 手术相关标签
            "[START]", "[END]"]

    def _read_json(self, input_file):
        """Reads a json file with automatic encoding detection."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data
                except json.JSONDecodeError:
                    pass
                
                lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            lines.append(item)
                        except json.JSONDecodeError:
                            continue
                return lines
                
        except UnicodeDecodeError:
            with open(input_file, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding']
                
                with open(input_file, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            return data
                    except json.JSONDecodeError:
                        pass
                    
                    lines = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:
                            try:
                                lines.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line['text']
            words = list(text)
            labels = ['O'] * len(words)
            
            # 处理 entities 字段（训练集和开发集都使用这种格式）
            if 'entities' in line:
                for entity in line['entities']:
                    if not isinstance(entity, dict):
                        continue
                        
                    entity_type = entity.get('type')
                    start = entity.get('start_idx')
                    end = entity.get('end_idx')
                    
                    if entity_type is None or start is None or end is None:
                        continue
                    
                    if start < 0 or end >= len(words) or start > end:
                        continue
                    
                    if start == end:
                        labels[start] = f"S-{entity_type}"
                    else:
                        labels[start] = f"B-{entity_type}"
                        for idx in range(start + 1, end + 1):
                            labels[idx] = f"I-{entity_type}"
            
            # 保留对 label 字段的处理（向后兼容）
            elif 'label' in line:
                label_dict = line['label']
                for entity_type, entities in label_dict.items():
                    for entity_name, positions in entities.items():
                        for start, end in positions:
                            if start < 0 or end >= len(words) or start > end:
                                continue
                            
                            if start == end:
                                labels[start] = f"S-{entity_type}"
                            else:
                                labels[start] = f"B-{entity_type}"
                                for idx in range(start + 1, end + 1):
                                    labels[idx] = f"I-{entity_type}"
            
            examples.append(InputExample(guid=guid, text_a=words, labels=labels))
        
        return examples