import torch.nn as nn
from transformers import BertModel

class BertRumorDetector(nn.Module):
    """
    BERT-based model for health rumor detection.
    """
    def __init__(self,
                 pretrained_model_name: str = 'bert-base-chinese',
                 num_labels: int = 2,
                 dropout_prob: float = 0.3):
        super(BertRumorDetector, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        # Classification head: dropout + linear layer
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        Args:
            input_ids (torch.LongTensor): Token IDs tensor of shape (batch_size, seq_len)
            attention_mask (torch.LongTensor): Attention mask tensor of shape (batch_size, seq_len)
        Returns:
            torch.FloatTensor: Logits tensor of shape (batch_size, num_labels)
        """
        # BERT outputs: (last_hidden_state, pooler_output, ...)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        # pooler_output is the [CLS] token embedding
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits
