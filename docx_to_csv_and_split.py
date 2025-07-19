import os
import random
import pandas as pd
from docx import Document


def extract_texts_from_docx(docx_path):
    """
    从 Word 文档中提取所有非空段落文本。
    """
    doc = Document(docx_path)
    texts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            texts.append(text)
    return texts


def main():
    # 输入文件路径
    real_docx = 'E:\竞赛\健康知识库.docx'    # 真是知识文档
    rumor_docx = 'E:\竞赛\谣言检测库.docx'    # 谣言文档
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    # 提取文本
    real_texts = extract_texts_from_docx(real_docx)
    rumor_texts = extract_texts_from_docx(rumor_docx)

    # 构造带标签的列表：真实0，谣言1
    rows = [(text, 0) for text in real_texts] + [(text, 1) for text in rumor_texts]

    # 打乱顺序
    random.seed(42)
    random.shuffle(rows)

    # 构造成 DataFrame
    df = pd.DataFrame(rows, columns=['text', 'label'])

    # 划分比例：80%/10%/10%
    n = len(df)
    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)

    df.iloc[:train_end].to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8-sig')
    df.iloc[train_end:valid_end].to_csv(os.path.join(output_dir, 'valid.csv'), index=False, encoding='utf-8-sig')
    df.iloc[valid_end:].to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8-sig')

    print(f"已生成数据集：train({train_end}), valid({valid_end - train_end}), test({n - valid_end}) 条")


if __name__ == '__main__':
    main()
