import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


# 加载训练和验证数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    texts = [item['text'] for item in data]  # 假设 'text' 是文本的键
    labels = [item['label'] for item in data]  # 假设 'label' 是标签的键
    return texts, labels


# 加载训练集和验证集
train_texts, train_labels = load_data(r'data/train.json')
val_texts, val_labels = load_data(r'data/valid.json')

# 使用 MultiLabelBinarizer 进行多标签编码
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)
y_val = mlb.transform(val_labels)

# 定义参数组合
vectorizer_params = [
    {'max_df': 0.9, 'min_df': 2, 'ngram_range': (1, 1)},
    {'max_df': 0.8, 'min_df': 3, 'ngram_range': (1, 2)},
]
nb_params = [0.1, 0.5, 1.0]  # alpha 参数

# 用于记录实验结果
results = []

for vec_param in vectorizer_params:
    for alpha in nb_params:
        # 文本特征向量化
        vectorizer = CountVectorizer(max_df=vec_param['max_df'], min_df=vec_param['min_df'],
                                     ngram_range=vec_param['ngram_range'])
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)

        # 初始化朴素贝叶斯模型
        nb_model = OneVsRestClassifier(MultinomialNB(alpha=alpha))
        nb_model.fit(X_train, y_train)

        # 模型验证
        y_pred = nb_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # 记录参数和结果
        results.append({
            'max_df': vec_param['max_df'],
            'min_df': vec_param['min_df'],
            'ngram_range': vec_param['ngram_range'],
            'alpha': alpha,
            'accuracy': accuracy
        })
        print(
            f"参数组合：max_df={vec_param['max_df']}, min_df={vec_param['min_df']}, ngram_range={vec_param['ngram_range']}, alpha={alpha}")
        print(f"验证集的整体准确率: {accuracy:.4f}")
        print(classification_report(y_val, y_pred, target_names=mlb.classes_, zero_division=1))

# 结果存储为DataFrame并输出
results_df = pd.DataFrame(results)
print("\n各参数组合的验证集准确率：")
print(results_df)

# 选择最佳参数组合的模型应用于测试集
best_params = results_df.loc[results_df['accuracy'].idxmax()]
best_vectorizer = CountVectorizer(max_df=best_params['max_df'], min_df=best_params['min_df'],
                                  ngram_range=best_params['ngram_range'])
X_train = best_vectorizer.fit_transform(train_texts)
X_val = best_vectorizer.transform(val_texts)

# 使用最佳参数训练模型
best_nb_model = OneVsRestClassifier(MultinomialNB(alpha=best_params['alpha']))
best_nb_model.fit(X_train, y_train)

# 加载测试集
test_file_path = r'data/test.txt'
with open(test_file_path, 'r', encoding='utf-8') as f:
    test_texts = [line.strip() for line in f]

# 对测试集进行向量化并预测
X_test = best_vectorizer.transform(test_texts)
test_predictions = best_nb_model.predict(X_test)

# 将预测结果和原始文本一起保存到txt文件中
output_file = "1.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    for text, pred in zip(test_texts, test_predictions):
        # 将预测的多标签编码转换为原始标签
        pred_labels = mlb.inverse_transform(np.array([pred]))[0]  # 确保pred是二维数组
        # 将原始文本和预测标签按指定格式写入文件
        f.write(f"{text} - {';'.join(pred_labels)}\n")

print(f"预测结果已保存到 {output_file}")
