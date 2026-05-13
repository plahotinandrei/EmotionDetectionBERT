import tempfile
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

model_path = "../models/emotion_model/checkpoint-1484"

login(token="token")

print("Загрузка модели и токенизатора...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

with tempfile.TemporaryDirectory() as tmp_dir:
    print(f"Сохраняем чистую модель во временную папку {tmp_dir}")
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    repo_id = "plakhotin94/emotion-bert"
    print(f"Загружаем на Hub в репозиторий {repo_id}...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

print("Модель загружена на hf!")