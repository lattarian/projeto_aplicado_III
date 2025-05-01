import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from ray import tune
import json

def main():
    # 1. Carregar e preparar os dados
    df_train = pd.read_csv("../data/rankt5_train.csv")
    df_val = pd.read_csv("../data/rankt5_val.csv")
    df_test = pd.read_csv("../data/rankt5_test.csv")

    # Dataset Hugging Face
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

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    def model_init():
        return T5ForConditionalGeneration.from_pretrained("t5-small")

    train_dataset = Dataset.from_pandas(df_train).map(preprocess, remove_columns=df_train.columns.tolist())
    val_dataset = Dataset.from_pandas(df_val).map(preprocess, remove_columns=df_val.columns.tolist())
    test_dataset = Dataset.from_pandas(df_test).map(preprocess, remove_columns=df_test.columns.tolist())

    # 2. Definir argumentos de treino iniciais
    training_args = TrainingArguments(
        output_dir="../data/rankt5_output_search",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=10,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"  # Desativa WandB ou Huggingface Hub
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 3. Definir espaço de busca para hiperparâmetros
    def model_hp_space(trial):
        return {
            "learning_rate": tune.loguniform(1e-5, 5e-4),
            "weight_decay": tune.uniform(0.0, 0.3),
            "num_train_epochs": tune.choice([2, 3, 4, 5]),
            "per_device_train_batch_size": tune.choice([8, 16]),
            "warmup_steps": tune.choice([0, 100, 300, 500])
        }

    # 4. Executar busca de hiperparâmetros
    print("🔍 Iniciando busca de hiperparâmetros...")
    best_run = trainer.hyperparameter_search(
        direction="minimize",
        backend="ray",
        n_trials=5,
        hp_space=model_hp_space,
        resources_per_trial={"cpu": 5, "gpu": 0}
    )

    print("🏆 Melhor configuração encontrada:")
    print(best_run)

    with open("../data/best_hyperparameters.json", "w") as f:
        json.dump(best_run.hyperparameters, f, indent=4)

    print("✅ Hiperparâmetros salvos em best_hyperparameters.json")

if __name__ == "__main__":
    main()
