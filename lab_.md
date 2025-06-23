**2. 实验设计与数据处理**

### 2.1 数据集选择

本次实验仅使用CoNLL-2003数据集进行测试。

### 2.2 数据处理

1.  **分词与标签转换**：
    -   采用BIO格式进行NER标注，以适应不同的模型架构。
    -   使用BERT自带的Tokenizer对文本进行分词，并确保每个subword正确映射到对应的NER标签。
2.  **数据集划分**：
    -   按照标准划分，将CoNLL-2003数据集分为训练集（train）、验证集（validation）和测试集（test），保持原始数据集的划分比例。
    -   采用固定随机种子确保实验的可复现性。

### 2.3 实验设计

1.  **实验设计**：
    -   本实验包含如下三个子问题：
        -   BERT做NER要不要加CRF层
        -   BERT-CRF加上BiLSTM有没有用
        -   BERT-BiLSTM-Softmax与BERT+BiLSTM+CRF相比，谁效果好
    -   实验预计测试以下四种NER模型架构：
        -   **BERT+Softmax**：直接使用BERT获取token embedding，并通过Softmax进行分类。
        -   **BERT+CRF**：在BERT输出后添加CRF层，以建模前后标签的依赖关系。
        -   **BERT+BiLSTM+CRF**：在BERT之后添加BiLSTM层，再接CRF层，以增强序列建模能力。
        -   **BERT+BiLSTM+Softmax**：在BERT后加入BiLSTM层，并使用Softmax进行分类，以验证BiLSTM对Softmax模型的影响。
2.  **超参数选择**：
    -   预训练模型：BERT-base。
    -   学习率设定：
        -   BERT层：5e-5，或通过参数搜索方法寻得。
        -   CRF层：100倍于BERT层的学习率。
        -   BiLSTM层：1e-3，或通过参数搜索方法寻得。
3.  **训练策略**：
    -   采用AdamW优化器，并使用适当的weight decay。
    -   设定Early Stopping策略，在验证集性能连续5个epoch未提升时停止训练。
    -   采用gradient clipping以避免梯度爆炸。
4.  **实验变量**：
    -   以BERT+Softmax作为基准模型，对比不同架构的表现。
    -   主要评价指标为F1-score，以衡量不同模型在CoNLL-2003数据集上的性能。