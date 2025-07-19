import os

# Paths to dataset CSV files (must exist, or adjust accordingly)
TRAIN_CSV = os.path.join('data', 'train.csv')
VALID_CSV = os.path.join('data', 'valid.csv')
TEST_CSV  = os.path.join('data', 'test.csv')

# Model checkpoint paths
MODEL_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')

# Pre-trained BERT model
PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'

# Tokenizer cache (optional)
# from transformers import BertTokenizer
# TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# Random seed for reproducibility
SEED = 42

# DataLoader settings
BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_LEN = 128

# Training hyperparameters
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.2

# Classification labels
NUM_LABELS = 2
LABEL_NAMES = ['非谣言', '谣言']
