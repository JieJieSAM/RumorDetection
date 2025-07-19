import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import create_data_loader
from model.BertRumorDetector import BertRumorDetector
import config


def load_model(device):
    """Load the trained BertRumorDetector model from checkpoint."""
    model = BertRumorDetector(
        pretrained_model_name=config.PRE_TRAINED_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        dropout_prob=config.DROPOUT
    )
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    df_test = pd.read_csv(config.TEST_CSV)
    tokenizer = config.TOKENIZER if hasattr(config, 'TOKENIZER') else None
    if tokenizer is None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

    test_loader = create_data_loader(
        df_test,
        tokenizer,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # 加载模型
    model = load_model(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=config.LABEL_NAMES,
        digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # 可选：保存混淆矩阵到文件
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.LABEL_NAMES, yticklabels=config.LABEL_NAMES)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig(config.CONFUSION_MATRIX_PATH)
        print(f"Saved confusion matrix to {config.CONFUSION_MATRIX_PATH}")
    except ImportError:
        pass


if __name__ == '__main__':
    evaluate()
