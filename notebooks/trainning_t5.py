import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from ray import tune
import json

model_name = "t5-small"
print("✂️ Tokenizando dados para entrada no modelo RankT5...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

df_train = pd.read_csv("../data/rankt5_train.csv")
df_val = pd.read_csv("../data/rankt5_val.csv")
df_test = pd.read_csv("../data/rankt5_test.csv")
best_run = json.load(open("../data/best_hyperparameters-first.json"))

def preprocess(line):
    input_text = str(line["input_text"])
    label_text = str(line["label"])

    input_enc = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=64
    )
    label_enc = tokenizer(
        label_text,
        padding="max_length",
        truncation=True,
        max_length=4
    )

    input_enc["labels"] = label_enc["input_ids"]
    return input_enc

def main():
    train_dataset = Dataset.from_pandas(df_train).map(preprocess, remove_columns=df_train.columns.tolist())
    val_dataset = Dataset.from_pandas(df_val).map(preprocess, remove_columns=df_val.columns.tolist())
    test_dataset = Dataset.from_pandas(df_test).map(preprocess, remove_columns=df_test.columns.tolist())

    # 5. Atualizar argumentos de treino com melhores hiperparâmetros
    training_args_final = TrainingArguments(
        output_dir="../data/rankt5_output_final",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_run["learning_rate"],
        weight_decay=best_run["weight_decay"],
        per_device_train_batch_size=best_run["per_device_train_batch_size"],
        per_device_eval_batch_size=best_run["per_device_train_batch_size"],
        num_train_epochs=best_run["num_train_epochs"],
        warmup_steps=best_run["warmup_steps"],
        logging_dir="./logs_final",
        logging_steps=10,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    trainer_final = Trainer(
        model=model,
        args=training_args_final,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 7. Treinar modelo final
    print("🚀 Treinando modelo final com melhores hiperparâmetros...")
    trainer_final.train()

    # 8. Avaliar no conjunto de teste
    print("🧪 Avaliando desempenho no conjunto de teste...")
    metrics = trainer_final.evaluate(test_dataset)
    print("📈 Métricas no conjunto de teste:", metrics)

if __name__ == "__main__":
    main()
