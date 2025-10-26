# %%
import os
import re
import traceback

import pandas as pd
import torch
from accelerate import PartialState
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    GPT2TokenizerFast,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer
from trl.models.utils import unwrap_model_for_generation

import datasets
import wandb
import Levenshtein
import numpy as np

load_dotenv()
from accelerate import Accelerator
import tqdm
from accelerate.utils import gather_object

accelerator = Accelerator()
device_string = PartialState().process_index

# %%


def get_role_by_idx(convo: list[dict[str, str]], role: str, idx: int) -> str:
    found = 0
    for message in convo:
        if message["role"] == role:
            if found == idx:
                return message["content"]
            found += 1
    raise ValueError(f"Role {role} not found {idx} times")


def calculate_typo_metrics(predicted: str, expected: str) -> dict:
    """Calculate metrics for typo correction task."""
    try:
        exact_match = predicted.strip() == expected.strip()
        
        # Character-level metrics
        char_distance = Levenshtein.distance(predicted, expected)
        total_chars = len(expected)
        char_accuracy = (total_chars - char_distance) / total_chars if total_chars > 0 else 0
        cer = char_distance / total_chars if total_chars > 0 else 0
        
        # Word-level metrics  
        predicted_words = predicted.split()
        expected_words = expected.split()
        word_distance = Levenshtein.distance(' '.join(predicted_words), ' '.join(expected_words))
        total_words = len(expected_words)
        word_accuracy = 1 - (word_distance / max(len(' '.join(predicted_words)), len(' '.join(expected_words)))) if total_words > 0 else 0
        
        return {
            'exact_match': int(exact_match),
            'char_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'cer': cer,
            'char_distance': char_distance,
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'exact_match': 0,
            'char_accuracy': 0.0,
            'word_accuracy': 0.0,
            'cer': 1.0,
            'char_distance': 999,
        }


class LLMSampleCB(TrainerCallback):
    def __init__(
        self,
        trainer,
        test_dataset,
        num_samples=50,
        max_new_tokens=450,
        log_model="checkpoint",
    ):
        "A CallBack to log typo correction samples to wandb.Table during training"
        super().__init__()
        self._log_model = log_model
        self.trainer = trainer

        # Sample random examples for evaluation
        self.sample_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

        self.model, self.tokenizer = trainer.model_wrapped, trainer.tokenizer
        self.tokenizer.padding_side = "left"

        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path, temperature=0.001, max_new_tokens=max_new_tokens
        )
        self.idx = 0
        self.baseline_metrics = None  # Store initial metrics before training

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        accelerator = self.trainer.accelerator

        # Create original prompts before distribution to use as keys
        original_prompts = self.tokenizer.apply_chat_template(conversations, tokenize=False)
        original_prompt_to_idx = {self._normalize_string(prompt): idx for idx, prompt in enumerate(original_prompts)}

        completions = [None] * len(conversations)  # Pre-allocate result array

        with accelerator.split_between_processes(conversations) as conversation_subset:
            model = self.trainer.model_wrapped
            with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                prompts = self.tokenizer.apply_chat_template(conversation_subset, tokenize=False)

                tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                with torch.inference_mode():
                    print("Generating...")
                    generations = unwrapped_model.generate(**tokenized_prompts, generation_config=self.gen_config).cpu()
                    print("Generated!")

                results = []
                for prompt_str, prompt_tokens, generation in zip(prompts, tokenized_prompts.input_ids, generations):
                    # Remove prompt from generation
                    generation = generation[len(prompt_tokens) :]
                    completion = self.tokenizer.decode(generation, skip_special_tokens=True)
                    results.append((prompt_str, completion))

        # Gather results from all processes
        all_results = gather_object(results)

        # Place completions in their original positions
        for prompt_str, completion in all_results:
            norm_prompt = self._normalize_string(prompt_str)
            if norm_prompt in original_prompt_to_idx:
                idx = original_prompt_to_idx[norm_prompt]
                completions[idx] = completion

        return completions

    def evaluate_typos(self):
        """Evaluate typo correction on sample dataset and create wandb.Table"""
        records_table = wandb.Table(columns=["typo_text", "correct_text", "prediction", "exact_match", "char_acc", "word_acc", "cer"])
        all_metrics = []
        
        batch_size = 16
        
        for i in tqdm.trange(0, len(self.sample_dataset), batch_size, desc="Evaluating typos"):
            batch = self.sample_dataset[i : i + batch_size]
            batch_data = []
            batch_inputs = []
            
            for example in batch:
                try:
                    messages = example["messages"]
                    user_msg = get_role_by_idx(messages, "user", 0)
                    correct_text = get_role_by_idx(messages, "assistant", 0)
                    
                    # Extract typo text from prompt
                    if "Napraw literówki" in user_msg and "`" in user_msg:
                        typo_text = user_msg.split("`")[1] if user_msg.count("`") >= 2 else ""
                    else:
                        typo_text = user_msg
                    
                    batch_inputs.append(messages[:-1])  # Remove assistant message
                    batch_data.append((typo_text, correct_text))
                except Exception as e:
                    print(f"Error processing example: {e}")
                    continue
            
            if not batch_inputs:
                continue
                
            # Generate predictions
            try:
                predictions = self.generate(batch_inputs)
            except Exception as e:
                print(f"Generation error: {e}")
                continue
            
            # Calculate metrics
            if self.trainer.accelerator.is_main_process:
                for idx, (typo_text, correct_text) in enumerate(batch_data):
                    if idx >= len(predictions):
                        continue
                        
                    prediction = predictions[idx]
                    metrics = calculate_typo_metrics(prediction, correct_text)
                    all_metrics.append(metrics)
                    
                    records_table.add_data(
                        typo_text[:200],
                        correct_text[:200], 
                        prediction[:200],
                        metrics['exact_match'],
                        f"{metrics['char_accuracy']*100:.2f}%",
                        f"{metrics['word_accuracy']*100:.2f}%",
                        f"{metrics['cer']*100:.2f}%"
                    )
        
        # Calculate aggregate metrics
        if all_metrics:
            avg_metrics = {
                'exact_match_rate': np.mean([m['exact_match'] for m in all_metrics]),
                'avg_char_accuracy': np.mean([m['char_accuracy'] for m in all_metrics]),
                'avg_word_accuracy': np.mean([m['word_accuracy'] for m in all_metrics]),
                'avg_cer': np.mean([m['cer'] for m in all_metrics]),
            }
        else:
            avg_metrics = {'exact_match_rate': 0, 'avg_char_accuracy': 0, 'avg_word_accuracy': 0, 'avg_cer': 1}
        
        return records_table, avg_metrics

    def on_evaluate(self, *args, **kwargs):
        "Log typo correction metrics to wandb after evaluation"
        if self.trainer.accelerator.is_main_process:
            print(f"\n{'='*80}\nRunning typo evaluation {self.idx}\n{'='*80}")
        
        try:
            # Evaluate typo correction
            typo_table, typo_metrics = self.evaluate_typos()
            
            # Store baseline metrics from first evaluation
            if self.baseline_metrics is None:
                self.baseline_metrics = typo_metrics.copy()
                if self.trainer.accelerator.is_main_process:
                    print(f"📊 Baseline metrics saved (before training)")
            
            if self.trainer.accelerator.is_main_process:
                # Calculate improvement vs baseline
                improvement = {
                    'exact_match_improvement': typo_metrics['exact_match_rate'] - self.baseline_metrics['exact_match_rate'],
                    'char_accuracy_improvement': typo_metrics['avg_char_accuracy'] - self.baseline_metrics['avg_char_accuracy'],
                    'word_accuracy_improvement': typo_metrics['avg_word_accuracy'] - self.baseline_metrics['avg_word_accuracy'],
                    'cer_improvement': self.baseline_metrics['avg_cer'] - typo_metrics['avg_cer'],  # Lower is better
                }
                
                # Create comparison table
                comparison_table = wandb.Table(
                    columns=["Metric", "Baseline (Before)", "Current", "Improvement", "% Change"],
                    data=[
                        ["Exact Match", 
                         f"{self.baseline_metrics['exact_match_rate']*100:.2f}%",
                         f"{typo_metrics['exact_match_rate']*100:.2f}%",
                         f"{improvement['exact_match_improvement']*100:+.2f}pp",
                         f"{(improvement['exact_match_improvement']/max(self.baseline_metrics['exact_match_rate'], 0.001))*100:+.1f}%"],
                        ["Char Accuracy",
                         f"{self.baseline_metrics['avg_char_accuracy']*100:.2f}%",
                         f"{typo_metrics['avg_char_accuracy']*100:.2f}%",
                         f"{improvement['char_accuracy_improvement']*100:+.2f}pp",
                         f"{(improvement['char_accuracy_improvement']/max(self.baseline_metrics['avg_char_accuracy'], 0.001))*100:+.1f}%"],
                        ["Word Accuracy",
                         f"{self.baseline_metrics['avg_word_accuracy']*100:.2f}%",
                         f"{typo_metrics['avg_word_accuracy']*100:.2f}%",
                         f"{improvement['word_accuracy_improvement']*100:+.2f}pp",
                         f"{(improvement['word_accuracy_improvement']/max(self.baseline_metrics['avg_word_accuracy'], 0.001))*100:+.1f}%"],
                        ["CER (lower=better)",
                         f"{self.baseline_metrics['avg_cer']*100:.2f}%",
                         f"{typo_metrics['avg_cer']*100:.2f}%",
                         f"{improvement['cer_improvement']*100:+.2f}pp",
                         f"{(improvement['cer_improvement']/max(self.baseline_metrics['avg_cer'], 0.001))*100:+.1f}%"],
                    ]
                )
                
                wandb.log({
                    f"typo_predictions_{self.idx}": typo_table,
                    f"comparison_table_{self.idx}": comparison_table,
                    "typo_exact_match": typo_metrics['exact_match_rate'],
                    "typo_char_accuracy": typo_metrics['avg_char_accuracy'],
                    "typo_word_accuracy": typo_metrics['avg_word_accuracy'],
                    "typo_cer": typo_metrics['avg_cer'],
                    # Improvement metrics
                    "improvement_exact_match": improvement['exact_match_improvement'],
                    "improvement_char_accuracy": improvement['char_accuracy_improvement'],
                    "improvement_word_accuracy": improvement['word_accuracy_improvement'],
                    "improvement_cer": improvement['cer_improvement'],
                })
                
                print(f"✓ Current metrics:")
                print(f"  - Exact match: {typo_metrics['exact_match_rate']*100:.2f}%")
                print(f"  - Char accuracy: {typo_metrics['avg_char_accuracy']*100:.2f}%")
                print(f"  - Word accuracy: {typo_metrics['avg_word_accuracy']*100:.2f}%")
                print(f"  - CER: {typo_metrics['avg_cer']*100:.2f}%")
                
                print(f"\n✨ Improvement vs baseline:")
                print(f"  - Exact match: {improvement['exact_match_improvement']*100:+.2f}pp")
                print(f"  - Char accuracy: {improvement['char_accuracy_improvement']*100:+.2f}pp")
                print(f"  - Word accuracy: {improvement['word_accuracy_improvement']*100:+.2f}pp")
                print(f"  - CER reduction: {improvement['cer_improvement']*100:+.2f}pp")
                        
        except Exception as e:
            print(f"⚠️  Error during evaluation: {e}")
            traceback.print_exc()

        self.idx += 1

    def _normalize_string(self, s):
        """Normalize string to avoid whitespace/newline comparison issues"""
        if s is None:
            return ""
        # Remove all whitespace and convert to lowercase for more robust matching
        return re.sub(r'\s+', '', s).lower()


class ErrorMonitorCallback(TrainerCallback):
    """Monitor and log training errors to wandb"""
    
    def __init__(self):
        super().__init__()
        self.error_count = 0
        self.last_loss = None
        self.loss_spike_threshold = 2.0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Monitor training metrics for anomalies"""
        if logs is None:
            return
            
        try:
            # Track loss spikes
            if 'loss' in logs:
                current_loss = logs['loss']
                
                if self.last_loss is not None:
                    if current_loss > self.last_loss * self.loss_spike_threshold:
                        print(f"⚠️  Loss spike detected: {self.last_loss:.4f} -> {current_loss:.4f}")
                        wandb.log({
                            "loss_spike": 1,
                            "loss_spike_from": self.last_loss,
                            "loss_spike_to": current_loss,
                        })
                    
                self.last_loss = current_loss
                
            # Check for NaN/Inf
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        print(f"⚠️  Invalid value detected: {key} = {value}")
                        wandb.log({f"invalid_{key}": 1})
                        self.error_count += 1
                        
            # Log error count
            if self.error_count > 0:
                wandb.log({"total_errors": self.error_count})
                
        except Exception as e:
            print(f"Error in ErrorMonitorCallback: {e}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Reset counters at training start"""
        self.error_count = 0
        self.last_loss = None
        print("✓ Error monitoring initialized")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Log final error summary"""
        print(f"\n{'='*80}")
        print(f"Training completed. Total errors detected: {self.error_count}")
        print(f"{'='*80}\n")
        wandb.log({"final_error_count": self.error_count})


def load_typo_dataset(csv_path: str, task_name: str = "typo") -> datasets.Dataset:
    """Load typo correction dataset from CSV and convert to chat format"""
    df = pd.read_csv(csv_path, header=None, names=['correct', 'typos'], dtype=str)
    
    print(f"Loading {len(df)} samples from {csv_path}")
    
    # Convert to messages format
    data = []
    for _, row in df.iterrows():
        correct_text = str(row['correct']).strip()
        typo_text = str(row['typos']).strip()
        
        messages = [
            {"role": "user", "content": f"Napraw literówki w poniższym tekście. Niczego nie dodawaj, niczego nie pomijaj. Zwróć tylko poprawiony tekst: `{typo_text}`"},
            {"role": "assistant", "content": correct_text}
        ]
        
        data.append({
            "messages": messages,
            "task": task_name
        })
    
    # Convert to HuggingFace dataset with explicit schema
    # Use from_pandas to avoid PyArrow type inference issues
    temp_df = pd.DataFrame(data)
    return datasets.Dataset.from_pandas(temp_df, preserve_index=False)


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "bielik-fixer"  # name your W&B project

    # %%
    # model_name = "speakleash/Bielik-11B-v2.2-Instruct"
    # tokenizer_name_or_path = "speakleash/Bielik-11B-v2.2-Instruct"
    model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"
    tokenizer_name_or_path = "speakleash/Bielik-4.5B-v3.0-Instruct"

    new_model_name = "./models/Bielik-4.5B-typo-fixer"

    # %%
    print("\n" + "="*80)
    print("Loading typo correction datasets")
    print("="*80)
    
    typo_train = load_typo_dataset("broken_text/dataset_train.csv", task_name="typo")
    typo_test = load_typo_dataset("broken_text/dataset_test.csv", task_name="typo")
    
    print(f"✓ Training samples: {len(typo_train)}")
    print(f"✓ Test samples: {len(typo_test)}")

    # If you want to mix with capability data, uncomment:
    # capability_train = datasets.load_dataset(
    #     "json",
    #     data_files="datasets/capability/dataset_json_capability_instructions.jsonl",
    #     split="train",
    # ).shuffle().select(range(200000))
    # capability_test = datasets.load_dataset(
    #     "json",
    #     data_files="datasets/capability/dataset_json_capability_instructions_tests.jsonl",
    #     split="train",
    # ).shuffle()
    # capability_test = capability_test.add_column("task", ["capability"] * len(capability_test))
    # dset_train = datasets.concatenate_datasets([typo_train, capability_train]).shuffle()
    # dset_test = datasets.concatenate_datasets([typo_test, capability_test]).shuffle()

    # Use only typo data
    dset_train = typo_train.shuffle()
    dset_test = typo_test.shuffle()

    # %%
    # Get example from typo test data for before/after comparison
    example = dset_test[0]
    example_typo = get_role_by_idx(example["messages"], "user", 0)
    example_correct = get_role_by_idx(example["messages"], "assistant", 0)
    
    print("\n" + "="*80)
    print("Example training sample:")
    print("="*80)
    print(f"Input (with typos): {example_typo[:200]}...")
    print(f"\nExpected output: {example_correct[:200]}...")
    print("="*80)

    # %%
    bnb_config = BitsAndBytesConfig(
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = AutoConfig.from_pretrained(model_name)
    model_config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        device_map={"": device_string},
        attn_implementation="flash_attention_2",
        config=model_config,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # %% Generate a sample answer before fine-tuning
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Use the example messages without the assistant response
    chat = example["messages"][:-1]  # Remove assistant message
    chat_row = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # %%
    tokenized = tokenizer(chat_row, return_tensors="pt")
    tokenized["input_ids"] = tokenized["input_ids"].cuda()
    tokenized["attention_mask"] = tokenized["attention_mask"].cuda()

    print("\n" + "="*80)
    print("Testing model BEFORE fine-tuning")
    print("="*80)
    
    out = model.generate(**tokenized, max_new_tokens=450, eos_token_id=terminators, temperature=0.001)

    decoded = tokenizer.decode(out[0])
    print(f"\nPrompt:\n{example_typo[:200]}...")
    print(f"\nModel output:\n{decoded[len(chat_row):].strip()[:200]}...")
    print(f"\nExpected:\n{example_correct[:200]}...")
    print("="*80)

    # %%
    peft_config = LoraConfig(
        r=8,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # use_dora=True
    )

    trainer = accelerator.prepare(
        SFTTrainer(
            model,
            train_dataset=dset_train,
            eval_dataset=dset_test,
            peft_config=peft_config,
            args=SFTConfig(
                output_dir="./models/checkpoints",
                logging_steps=1,
                report_to="wandb",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                optim="paged_adamw_8bit",
                lr_scheduler_type="cosine_with_restarts",
                warmup_steps=10,
                learning_rate=4e-5,
                weight_decay=0.01,
                gradient_checkpointing=True,
                eval_strategy="steps",
                eval_steps=250,
                packing=True,
                save_strategy="steps",
                save_steps=250,  # Must match eval_steps for load_best_model_at_end
                fp16=True,
                # Improved stability settings
                max_grad_norm=1.0,  # Gradient clipping to prevent exploding gradients
                logging_first_step=True,  # Log first step for debugging
                logging_nan_inf_filter=False,  # Don't filter NaN/Inf to catch them
                save_total_limit=3,  # Keep only last 3 checkpoints
                load_best_model_at_end=True,  # Load best model after training
                metric_for_best_model="eval_loss",  # Use eval loss as best metric
                greater_is_better=False,  # Lower eval loss is better
            ),
        )
    )

    trainer.tokenizer.padding_side = "left"

    # %%
    # Add monitoring callbacks
    print("Setting up monitoring callbacks...")
    wandb_callback = LLMSampleCB(trainer, dset_test, num_samples=50, max_new_tokens=450)
    error_callback = ErrorMonitorCallback()
    
    trainer.add_callback(wandb_callback)
    trainer.add_callback(error_callback)
    
    print(f"✓ Callbacks configured")
    print(f"✓ Training dataset size: {len(dset_train)}")
    print(f"✓ Test dataset size: {len(dset_test)}")
    print(f"✓ Model: {model_name}")
    print(f"✓ Output: {new_model_name}")

    # %%
    print(f"\n{'='*80}")
    print("Starting the training")
    print(f"{'='*80}\n")
    
    try:
        trainer.train()
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n{'='*80}")
        print("⚠️  TRAINING ERROR")
        print(f"{'='*80}")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        
        # Log error to wandb
        try:
            wandb.log({
                "training_failed": 1,
                "error_message": str(e),
                "error_traceback": traceback.format_exc()
            })
        except:
            pass
        
        raise  # Re-raise to stop execution

    # %%
    print("\n" + "="*80)
    print("Saving LoRA adapter...")
    print("="*80)
    
    try:
        trainer.model.save_pretrained("./models/lora")
        print("✓ LoRA adapter saved to ./models/lora")
    except Exception as e:
        print(f"⚠️  Error saving LoRA adapter: {e}")
        traceback.print_exc()

    if accelerator.is_local_main_process:
        print("\n" + "="*80)
        print("Testing model AFTER fine-tuning...")
        print("="*80)
        
        try:
            out = trainer.model.generate(**tokenized, max_new_tokens=450, eos_token_id=terminators, temperature=0.001)
            decoded = trainer.tokenizer.decode(out[0])
            model_output = decoded[len(chat_row):].strip()
            
            print(f"\nPrompt:\n{example_typo[:200]}...")
            print(f"\nModel output:\n{model_output[:200]}...")
            print(f"\nExpected:\n{example_correct[:200]}...")
            
            # Calculate metrics
            metrics = calculate_typo_metrics(model_output, example_correct)
            print(f"\nMetrics:")
            print(f"  - Exact match: {metrics['exact_match']}")
            print(f"  - Char accuracy: {metrics['char_accuracy']*100:.2f}%")
            print(f"  - Word accuracy: {metrics['word_accuracy']*100:.2f}%")
            print(f"  - CER: {metrics['cer']*100:.2f}%")
            print("-" * 80)
        except Exception as e:
            print(f"⚠️  Error during test generation: {e}")
            traceback.print_exc()

        print("\n" + "="*80)
        print("Merging and saving final model...")
        print("="*80)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            peft_model = PeftModel.from_pretrained(model, "./models/lora", torch_dtype=torch.bfloat16)
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(new_model_name)
            tokenizer.save_pretrained(new_model_name)
                
            print(f"✓ Final model saved to {new_model_name}")
            print(f"✓ Tokenizer saved to {new_model_name}")
            
            wandb.log({
                "training_completed": 1,
                "final_model_path": new_model_name
            })
            
            print("\n" + "="*80)
            print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"⚠️  Error during model merge/save: {e}")
            print(traceback.format_exc())
            wandb.log({
                "model_save_failed": 1,
                "save_error": str(e)
            })
            raise
