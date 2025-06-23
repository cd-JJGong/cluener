import json
from vocabulary import Vocabulary

class CluenerProcessor:
    """Processor for the chinese ner data set."""
    def __init__(self,data_dir):
        self.vocab = Vocabulary()
        self.data_dir = data_dir

    def get_vocab(self):
        vocab_path = self.data_dir / 'vocab.pkl'
        if vocab_path.exists():
            self.vocab.load_from_file(str(vocab_path))
        else:
            files = ["train.json", "dev.json", "test.json"]
            for file in files:
                with open(str(self.data_dir / file), 'r') as fr:
                    for line in fr:
                        line = json.loads(line.strip())
                        text = line['text']
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "train.json"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "dev.json"), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "test.json"), "test")

    def _create_examples(self,input_path,mode):
        examples = []
        with open(input_path, 'r') as f:
            idx = 0
            for line in f:
                json_d = {}
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = " ".join(words)
                json_d['tag'] = " ".join(labels)
                json_d['raw_context'] = "".join(words)
                idx += 1
                examples.append(json_d)
        return examples

class MedicalNerProcessor:
    """Processor for the medical NER data set."""
    def __init__(self,data_dir):
        self.vocab = Vocabulary()
        self.data_dir = data_dir

    def get_vocab(self):
        vocab_path = self.data_dir / 'vocab.pkl'
        if vocab_path.exists():
            self.vocab.load_from_file(str(vocab_path))
        else:
            files = ["train.json", "dev.json", "test.json"]
            for file in files:
                with open(str(self.data_dir / file), 'r', encoding='utf-8') as fr:
                    data = json.load(fr)  # 直接读取整个JSON数组
                    for line in data:
                        text = line['text']
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "train.json"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "dev.json"), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "test.json"), "test")

    def _create_examples(self,input_path,mode):
        examples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接读取整个JSON数组
            for idx, line in enumerate(data):
                json_d = {}
                text = line['text']
                words = list(text)
                labels = ['O'] * len(words)
                
                # 处理entities字段
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
                
                # 保留对label字段的处理（向后兼容）
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
                
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = " ".join(words)
                json_d['tag'] = " ".join(labels)
                json_d['raw_context'] = "".join(words)
                examples.append(json_d)
        return examples


