# import torch
# from transformers import BertTokenizer
# import config
# from model.BertRumorDetector import BertRumorDetector
# from utils import load_checkpoint, get_device
#
# def predict(text: str):
#     """
#     Predict whether the input text is a rumor or not.
#
#     Args:
#         text (str): Input text to classify.
#     Returns:
#         tuple: (label, confidence)
#     """
#     device = get_device()
#
#     # Load tokenizer and model (cache loaded)
#     tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
#     model = BertRumorDetector(
#         pretrained_model_name=config.PRE_TRAINED_MODEL_NAME,
#         num_labels=config.NUM_LABELS,
#         dropout_prob=config.DROPOUT
#     )
#     load_checkpoint(model, config.MODEL_SAVE_PATH, device)
#
#     # Encode input text
#     encoding = tokenizer(
#         text,
#         add_special_tokens=True,
#         max_length=config.MAX_LEN,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
#
#     # Inference
#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         probs = torch.softmax(outputs, dim=1).cpu().squeeze().numpy()
#         pred_idx = int(probs.argmax())
#         label = config.LABEL_NAMES[pred_idx]
#         confidence = float(probs[pred_idx])
#     return label, confidence
#
# if __name__ == '__main__':
#     text = input("请输入要检测的内容：\n")
#     label, conf = predict(text)
#     print(f"预测结果: {label} (置信度: {conf:.4f})")



import os
import json
import torch
from openai import OpenAI
from transformers import BertTokenizer
import config
from model.BertRumorDetector import BertRumorDetector
from utils import load_checkpoint, get_device

os.environ["OPENAI_API_KEY"] = "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"
#写密钥

# ----------------- GPT 配置 -----------------
OPENAI_MODEL = "gpt-4o-mini"          # 可换成 gpt-4o
LOWER, UPPER = 0.001, 0.999               # 灰区阈值
client = OpenAI()                     # 读取 OPENAI_API_KEY

# ---------- Prompt & Schema ----------
SYSTEM_MSG = "你是 RumourGPT，一名循证的中文谣言检测专家。你必须遵循提供的 JSON Schema，仅返回有效 JSON。"
SCHEMA = {
    "name": "rumor_report",
    "description": "谣言检测结果与解释报告",
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["rumor", "not_rumor", "uncertain"]
            },
            "credibility": {"type": "integer", "minimum": 0, "maximum": 100},
            "rationale": {"type": "array", "items": {"type": "string"}},
            "suggested_checks": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["label", "credibility", "rationale"]
    }
}

USER_TEMPLATE = (
    "【待检测文本】\n{claim}\n\n"
    "【现有模型输出】\n初步判定: {label_cn}（置信度: {score:.2f}）\n\n"
    "请按要求生成检测报告：\n"
    "1. 重新判断（rumor / not_rumor / uncertain）；\n"
    "2. 给出 0-100 可信度；\n"
    "3. 用简短要点列出关键依据；\n"
    "4. 提供可验证来源或查证建议；\n"
    "只返回符合 JSON Schema 的结果。"
)

# ----------------- 核心函数 -----------------
def load_local_model():
    device = get_device()
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    model = BertRumorDetector(
        pretrained_model_name=config.PRE_TRAINED_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        dropout_prob=config.DROPOUT
    )
    load_checkpoint(model, config.MODEL_SAVE_PATH, device)
    return tokenizer, model, device

tokenizer, base_model, DEVICE = load_local_model()

def base_predict(text: str):
    """现有 BERT 模型推断，返回 (label_cn, prob_float)."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    base_model.eval()
    with torch.no_grad():
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1).cpu().squeeze().numpy()
    idx = int(probs.argmax())
    return config.LABEL_NAMES[idx], float(probs[idx])       # 中文标签, 置信度

def call_gpt_explainer(text: str, label_cn: str, score: float) -> dict:
    """调用 GPT，返回结构化 JSON 报告（可能覆盖标签）"""
    user_msg = USER_TEMPLATE.format(
        claim=text.strip(),
        label_cn=label_cn,
        score=score
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        tools=[{"type": "function", "function": SCHEMA}],
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg}
        ],
        response_format={"type": "json_object"}
    )
    # 提取函数调用结果
    fn_args = resp.choices[0].message.tool_calls[0].function.arguments
    return json.loads(fn_args)

def predict_with_explainer(text: str):
    """完整流程：本地 BERT + GPT 灰区解释，返回 (final_label_cn, base_prob, gpt_report_dict)"""
    base_label, base_prob = base_predict(text)
    if LOWER <= base_prob <= UPPER:
        gpt_report = call_gpt_explainer(text, base_label, base_prob)
        # 将 GPT label 英文 → 中文映射，方便统一
        gpt2cn = {"rumor": "谣言", "not_rumor": "非谣言", "uncertain": "不确定"}
        final_label = gpt2cn.get(gpt_report["label"], base_label)
    else:
        gpt_report = None
        final_label = base_label
    return final_label, base_prob, gpt_report

# ----------------- CLI 演示 -----------------
if __name__ == "__main__":
    while True:
        text = input("\n请输入要检测的文本（空行退出）：\n")
        if not text.strip():
            break
        label, prob, report = predict_with_explainer(text)
        print(f"\n→ 最终判定：{label}   基础置信度：{prob:.2f}")
        if report:
            print("→ GPT 检测报告(JSON)：")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print("→ 置信度高/低，未调用 GPT。")
