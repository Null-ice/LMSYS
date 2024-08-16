import argparse
import logging
from dataclasses import dataclass, asdict
import os
import copy
import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
from huggingface_hub import HfApi
from datasets import concatenate_datasets
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import EvalPrediction
from dataclasses import field
from typing import List
from datasets import concatenate_datasets
import pandas as pd
import torch.nn as nn
import warnings

warnings.simplefilter('ignore')

@dataclass
class Config:
    VER: int = 1
    max_length: int = 1024
    n_splits: int = 5
    fold_idx: int = 0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 8
    n_epochs: int = 1
    freeze_layers: int = 0
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    eval_strategy: str = "epoch"
    save_strategy: str = "steps"
    save_steps: int = 200
    logging_steps: int = 10
    output_dir: str = None
    checkpoint: str = "/root/autodl-tmp/lmsys/gemma/gemma-2-9b-it-bnb-4bit_mirror"
    mirror_url: str = "https://hf-mirror.com"
    optim_type: str = "adamw_8bit"
    train_csv: str = "/root/autodl-tmp/lmsys/data/official_data/train.csv"
    extra_train: str = "none"
    # load_best_model_at_end = True  # 训练结束时加载最佳模型
    # greater_is_better = False
    # metric_for_best_model = "eval_log_loss"  # 用于选择最佳模型的度量标准
    # evaluation_strategy = 'steps'  # 更改为 steps 评估策略
    # eval_steps = 200
    # save_strategy = 'steps'  # 更改为 steps 保存策略
    # save_steps = 200  # 每多少步保存一次模型
    # save_total_limit = 1  # 保存检查点总数限制

class CustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None: # ->None 不返回任何值
        self.tokenizer = tokenizer
        self.max_length = max_length

    # __call__ 方法：允许类的实例像函数一样被调用。在这里用于对批量数据进行处理。
    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        # \n: 1 个字符 \n: 1 个字符 <response_a>: : 13 个字符
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        # 对文本进行分词
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        # 生成标签
        labels=[]
        for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
            if a_win:
                label = 0
            elif b_win:
                label = 1
            else:
                label = 2
            labels.append(label)
        return {**tokenized, "labels": labels}

    # 作用：@staticmethod 装饰器用于定义一个静态方法。静态方法属于类而不是类的实例，可以直接通过类本身调用，而不需要实例化对象。
    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))

class CustomClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(config.hidden_size * 2, config.hidden_size // 2, bias=False)
        self.fc3 = nn.Linear(config.hidden_size // 2, config.num_labels, bias=False)
    
    def forward(self, features):
        x= self.fc1(features)
        x= self.fc2(x)
        x= self.fc3(x)
        return x

class EvaluationCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, logger):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 400 == 0:
            self.logger.info(f"Evaluating at step {state.global_step}...")
            metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)
            self.logger.info(f"Step {state.global_step} evaluation results:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value}")
                
def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

def setup_logging(config):
    log_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_v{config.VER}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class ParallelClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Original FC layers
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(config.hidden_size * 2, config.hidden_size // 2, bias=False)
        
        # Attention layers
        self.attention = nn.MultiheadAttention(config.hidden_size, num_heads=4, batch_first=True)
        self.attn_fc1 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.attn_fc2 = nn.Linear(config.hidden_size * 2, config.hidden_size // 2)
        
        # Combine outputs
        self.combine = nn.Linear(config.hidden_size, config.num_labels)
        
        self.activation = nn.GELU()

    def forward(self, features):
        # Original FC path
        x_fc = self.fc1(features)
        x_fc = self.fc2(x_fc)
        
        # Attention path
        attn_output, _ = self.attention(features, features, features)
        x_attn = features + attn_output  # Residual connection
        x_attn = self.attn_fc1(x_attn)
        x_attn = self.activation(x_attn)
        x_attn = self.attn_fc2(x_attn)
        
        # Combine outputs
        x_combined = torch.cat([x_fc, x_attn], dim=-1)
        x_combined = self.activation(x_combined)

        output = self.combine(x_combined)
        
        return output

def log_parameters(logger, config):
    logger.info("=== Parameter Settings ===")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    logger.info("="*100)

def main(args):
    config = Config(
        VER=args.ver,
        max_length=args.max_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        warmup_steps = args.warmup_steps,
        save_steps = args.save_steps,
        per_device_eval_batch_size = args.eval_batch_size,
        extra_train = args.extra_data,
        freeze_layers = args.freeze_layers,

        lora_r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,

        lr=args.lr,
        n_splits=args.n_splits,
        fold_idx=args.fold_idx,
        n_epochs=args.epochs,
    )
    config.output_dir = f"Gemma2_QLoRA_ft_v{config.VER}"
    
    logger = setup_logging(config)
    log_parameters(logger, config)

     # Load and process data
    train = pd.read_csv(config.train_csv)
    train['label'] = train[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    label_encoder = LabelEncoder()
    train['label'] = label_encoder.fit_transform(train['label'])
    train = train[["prompt", "response_a", "response_b", 'winner_model_a', 'winner_model_b', 'winner_tie', 'label']]
    
    # Load and process extra data if provided
    if config.extra_train != "none":
        extra_data = pd.read_csv(config.extra_train)
        extra_data['label'] = extra_data[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
        extra_data['label'] = label_encoder.transform(extra_data['label'])
        extra_data = extra_data[["prompt", "response_a", "response_b", 'winner_model_a', 'winner_model_b', 'winner_tie', 'label']]
        logger.info(f"Loaded extra data with {len(extra_data)} samples")

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        report_to="none",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=config.logging_steps,
        evaluation_strategy="no",  # 我们将手动控制评估
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        optim=config.optim_type,
        bf16=True,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
    )

    # LoRA配置
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
    )

    # 加载分词器和模型
    tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"

    model = Gemma2ForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=3,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        attn_implementation="eager",
    )       
    model.score = CustomClassificationHead(model.config)
    model.score = model.score.to('cuda')
    # model.score = ParallelClassificationHead(model.config)
    # model.score = model.score.to('cuda')
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    logger.info("Model loaded and prepared for training.")
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")

    # Create datasets
    custom_tokenizer = CustomTokenizer(tokenizer, config.max_length)
    original_ds = Dataset.from_pandas(train).map(
        custom_tokenizer, 
        batched=True, 
        remove_columns=train.columns.tolist()
    )
    if config.extra_train != "none":
        extra_ds = Dataset.from_pandas(extra_data).map(
            custom_tokenizer, 
            batched=True, 
            remove_columns=extra_data.columns.tolist()
        )
    # Create train/eval split
    folds = [
        (
            [i for i in range(len(original_ds)) if i % config.n_splits != config.fold_idx],
            [i for i in range(len(original_ds)) if i % config.n_splits == config.fold_idx]
        ) 
    ]
    train_idx, eval_idx = folds[config.fold_idx]

    # Combine original training data with extra data
    if config.extra_train != "none":
        train_ds = concatenate_datasets([original_ds.select(train_idx), extra_ds])
    else:
        train_ds = original_ds.select(train_idx)
    
    eval_ds = original_ds.select(eval_idx)

    logger.info(f"Training dataset size: {len(train_ds)}")
    logger.info(f"Evaluation dataset size: {len(eval_ds)}")

    # 创建Trainer
    trainer = Trainer(
        args=training_args, 
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # Add custom callback
    evaluation_callback = EvaluationCallback(trainer, eval_ds, logger)
    trainer.add_callback(evaluation_callback)

    # 开始训练
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # 记录训练结果
    logger.info("Training completed. Results:")
    for key, value in train_result.metrics.items():
        logger.info(f"  {key}: {value}")

    # 进行最终评估
    logger.info("Performing final evaluation...")
    eval_result = trainer.evaluate()
    logger.info("Final evaluation results:")
    for key, value in eval_result.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gemma2 model with QLoRA")
    parser.add_argument("--ver", type=int, default=0, help="Version number")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=2, help="Gradient_accumulation_steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup_steps")
    # parser.add_argument("--warm_up_ratios", type=int, default=20, help="Warmup_ratios")
    parser.add_argument("--save_steps", type=int, default=200, help="Save_steps")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="eval_batch_size")
    parser.add_argument("--extra_data", type=str, default="none", help="External Train Data Path")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Freeze_layers")
    
    parser.add_argument("--lora_r", type=int, default=16, help="lora_r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora_dropout")

    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of data splits for cross-validation")
    parser.add_argument("--fold_idx", type=int, default=0, help="Index of the current fold")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
