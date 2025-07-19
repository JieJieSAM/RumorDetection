# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch import nn
# from torch.optim import AdamW
# from transformers import get_linear_schedule_with_warmup, BertTokenizer
# from data_loader import create_data_loader
# from model.BertRumorDetector import BertRumorDetector
# import config
# from sklearn.metrics import classification_report
#
#
# def set_seed(seed: int):
#     """Set random seed for reproducibility."""
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#
# def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
#     model.train()
#     losses = []
#     correct_predictions = 0
#
#     for batch in data_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         loss = loss_fn(outputs, labels)
#         _, preds = torch.max(outputs, dim=1)
#         correct_predictions += torch.sum(preds == labels)
#         losses.append(loss.item())
#
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
#
#     return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)
#
#
# def eval_model(model, data_loader, loss_fn, device):
#     model.eval()
#     losses = []
#     correct_predictions = 0
#     all_labels = []
#     all_preds = []
#
#     with torch.no_grad():
#         for batch in data_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             loss = loss_fn(outputs, labels)
#             _, preds = torch.max(outputs, dim=1)
#
#             correct_predictions += torch.sum(preds == labels)
#             losses.append(loss.item())
#
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#
#     report = classification_report(
#         all_labels,
#         all_preds,
#         target_names=config.LABEL_NAMES,
#         digits=4
#     )
#     return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), report
#
#
# def main():
#     # 初始化环境
#     set_seed(config.SEED)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 加载数据集
#     train_df = pd.read_csv(config.TRAIN_CSV)
#     val_df = pd.read_csv(config.VALID_CSV)
#
#     # Tokenizer & DataLoader
#     tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
#     train_loader = create_data_loader(
#         train_df,
#         tokenizer,
#         max_len=config.MAX_LEN,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=config.NUM_WORKERS
#     )
#     val_loader = create_data_loader(
#         val_df,
#         tokenizer,
#         max_len=config.MAX_LEN,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS
#     )
#
#     # 初始化模型
#     model = BertRumorDetector(
#         pretrained_model_name=config.PRE_TRAINED_MODEL_NAME,
#         num_labels=config.NUM_LABELS,
#         dropout_prob=config.DROPOUT
#     ).to(device)
#
#     # 优化器和学习率调度
#     optimizer = AdamW(
#         model.parameters(),
#         lr=config.LEARNING_RATE,
#         weight_decay=config.WEIGHT_DECAY
#     )
#     total_steps = len(train_loader) * config.EPOCHS
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=int(0.1 * total_steps),
#         num_training_steps=total_steps
#     )
#
#     loss_fn = nn.CrossEntropyLoss().to(device)
#
#     # 训练和验证
#     best_loss = float('inf')
#     for epoch in range(config.EPOCHS):
#         print(f"Epoch {epoch+1}/{config.EPOCHS}")
#         train_acc, train_loss = train_epoch(
#             model, train_loader, loss_fn, optimizer, device, scheduler
#         )
#         print(f"Train loss: {train_loss:.4f} | accuracy: {train_acc:.4f}")
#
#         val_acc, val_loss, val_report = eval_model(
#             model, val_loader, loss_fn, device
#         )
#         print(f"Val   loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")
#         print(val_report)
#
#         # 保存最优模型
#         if val_loss < best_loss:
#             best_loss = val_loss
#             os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
#             torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
#             print(f"Saved best model to {config.MODEL_SAVE_PATH}")
#
#     print("Training complete.")
#
#
# if __name__ == '__main__':
#     main()


import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from data_loader import create_data_loader
from model.BertRumorDetector import BertRumorDetector
import config
from sklearn.metrics import classification_report


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=config.LABEL_NAMES,
        digits=4
    )
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), report


def main():
    # 初始化环境
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VALID_CSV)

    # Tokenizer & DataLoader
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    train_loader = create_data_loader(
        train_df,
        tokenizer,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_loader = create_data_loader(
        val_df,
        tokenizer,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # 初始化模型
    model = BertRumorDetector(
        pretrained_model_name=config.PRE_TRAINED_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        dropout_prob=config.DROPOUT
    ).to(device)

    # 优化器和学习率调度
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    # 训练和验证
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler
        )
        print(f"Train loss: {train_loss:.4f} | accuracy: {train_acc:.4f}")

        val_acc, val_loss, val_report = eval_model(
            model, val_loader, loss_fn, device
        )
        print(f"Val   loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")
        print(val_report)

        # 保存最优模型（基于最高验证准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best acc {best_acc:.4f}, model saved to {config.MODEL_SAVE_PATH}")

    print("Training complete.")


if __name__ == '__main__':
    main()
