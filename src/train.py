from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Используется устройство: {device}")

dataset = load_from_disk("../data/tokenized_cedr")
dataset = dataset.train_test_split(test_size=0.2, seed=42)

model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
model.to(device)

label_list = [sample["labels"] for sample in dataset["train"]]
class_weights = torch.tensor(
    compute_class_weight("balanced", classes=np.unique(label_list), y=label_list),
    dtype=torch.float
)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        lbl = inputs.pop("labels")          # колонка теперь labels
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))(
            outputs.logits, lbl
        )
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="../models/emotion_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    report_to="none",
    dataloader_pin_memory=False   # для MPS
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("../models/emotion_model")
tokenizer.save_pretrained("../models/emotion_model")
print("Обучение завершено.")