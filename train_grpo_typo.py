#!/usr/bin/env python3
"""
GRPO Training for Typo Correction

Simple character-by-character matching reward function.
Train model to directly output corrected text without thinking/reasoning.

Uses Group Relative Policy Optimization (GRPO) from TRL with LoRA.
"""

import os
import re
import pandas as pd
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
import wandb

# ============================================================================
# Configuration
# ============================================================================

# MODEL_NAME = "speakleash/Bielik-4.5B-v3.0-Instruct"
MODEL_NAME = "./models/Bielik-4.5B-typo-fixer"
OUTPUT_NAME = "./models/Bielik-4.5B-grpo-typo-fixer-v0.4"
MAX_SEQ_LENGTH = 2048
DEVICE = "cuda:0"  # Specific GPU to use

os.environ["WANDB_PROJECT"] = "bielik-grpo-typo-fixer"

# Initialize wandb early to log configuration
wandb.init(
    project=os.environ["WANDB_PROJECT"],
    config={
        "model": MODEL_NAME,
        "device": DEVICE,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
)

# ============================================================================
# Best Model Tracking
# ============================================================================

class BestModelTracker(TrainerCallback):
    """Track the best model checkpoint based on mean reward"""
    
    def __init__(self):
        self.best_reward = float('-inf')
        self.best_checkpoint = None
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics is None:
            return
        
        # Look for mean reward metric
        reward_metric = metrics.get('eval_reward/mean', None)
        if reward_metric is None:
            # Try alternative metric names
            reward_metric = metrics.get('eval_rewards/mean', None)
        
        if reward_metric is not None and reward_metric > self.best_reward:
            self.best_reward = reward_metric
            # Checkpoint name format: checkpoint-{step}
            self.best_checkpoint = f"{args.output_dir}/checkpoint-{state.global_step}"
            print(f"\n🏆 New best checkpoint! Reward: {reward_metric:.4f} at step {state.global_step}")
            print(f"   Checkpoint: {self.best_checkpoint}")
            
            # Log to wandb
            wandb.log({
                'best_model/reward': self.best_reward,
                'best_model/step': state.global_step,
            })

# ============================================================================
# Load Dataset
# ============================================================================

def load_typo_dataset(csv_path: str, tokenizer) -> Dataset:
    """Load typo correction dataset with pre-formatted prompts for GRPO"""
    df = pd.read_csv(csv_path, header=None, names=['typos', 'correct'], dtype=str)
    
    data = []
    for _, row in df.iterrows():
        correct_text = str(row['correct']).strip()
        typo_text = str(row['typos']).strip()
        
        # Create simple prompt for typo correction
        user_content = f"""Napraw literówki w poniższym tekście:

{typo_text}"""
        
        # Format with chat template and convert to string
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        # Apply chat template to get formatted string prompt
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        data.append({
            "prompt": formatted_prompt,  # GRPOTrainer expects "prompt" column with formatted STRING
            "correct_text": correct_text,
            "typo_text": typo_text,
        })
    
    temp_df = pd.DataFrame(data)
    return Dataset.from_pandas(temp_df, preserve_index=False)

# Datasets will be loaded after tokenizer is initialized

# ============================================================================
# Reward Function
# ============================================================================

# def reward_function(prompts, completions, correct_text, typo_text, **kwargs):
#     """
#     Multi-component reward function for typo correction with reasoning.
    
#     Rewards:
#     1. Format adherence (has <think> and <answer> tags)
#     2. Answer quality (low CER = high reward)
#     3. Thinking quality (reasonable length, mentions errors)
    
#     Args:
#         prompts: List of prompts
#         completions: List of completions
#         correct_text: List of correct texts (from dataset column)
#         typo_text: List of typo texts (from dataset column)
#         **kwargs: Additional columns from dataset
#     """
#     # Debug: print what we receive on first call
#     if not hasattr(reward_function, '_debug_printed'):
#         print(f"\n🔍 DEBUG reward_function called:")
#         print(f"  - prompts: {len(prompts)} (type: {type(prompts[0]) if prompts else 'N/A'})")
#         print(f"  - completions: {len(completions)} (type: {type(completions[0]) if completions else 'N/A'})")
#         print(f"  - correct_text: {len(correct_text)}")
#         print(f"  - typo_text: {len(typo_text)}")
#         print(f"  - kwargs keys: {list(kwargs.keys())}")
        
#         # Show first completion sample
#         if completions:
#             print(f"\n  First completion type: {type(completions[0])}")
#             print(f"  First completion length: {len(str(completions[0]))} chars")
#             print(f"  First completion (first 300 chars): {str(completions[0])[:300]}...")
        
#         reward_function._debug_printed = True
    
#     # Convert completions to strings if they're not already
#     # (GRPOTrainer may pass them as various formats)
#     processed_completions = []
#     for comp in completions:
#         if isinstance(comp, str):
#             processed_completions.append(comp)
#         elif isinstance(comp, list):
#             # If it's a list, join it or take first element
#             if comp and isinstance(comp[0], dict) and 'content' in comp[0]:
#                 # Messages format: [{"role": "assistant", "content": "..."}]
#                 processed_completions.append(comp[0]['content'])
#             else:
#                 processed_completions.append(str(comp))
#         else:
#             processed_completions.append(str(comp))
    
#     completions = processed_completions
    
#     rewards = []
#     reward_components = []  # Track components for debugging
    
#     for idx, (completion, correct_txt) in enumerate(zip(completions, correct_text)):
#         total_reward = 0.0
#         components = {}  # Debug info
        
#         # Component 1: Format reward (0-1 points)
#         has_think, has_answer, proper_format = check_format(completion)
        
#         format_reward = 0.0
#         if proper_format:
#             format_reward = 1.0  # Full format reward
#         elif has_think and has_answer:
#             format_reward = 0.5  # Partial reward for tags but wrong order
#         elif has_answer:
#             format_reward = 0.3  # Small reward for at least having answer
        
#         total_reward += format_reward
#         components['format'] = format_reward
        
#         # Component 2: Answer quality reward (0-3 points)
#         answer_reward = 0.0
#         if has_answer:
#             predicted_text = extract_answer(completion)
#             cer = calculate_cer(predicted_text, correct_txt)
            
#             # Convert CER to reward: CER=0 -> reward=3, CER=1 -> reward=0
#             answer_reward = max(0, 3 * (1 - cer))
#             total_reward += answer_reward
#             components['answer'] = answer_reward
#             components['cer'] = cer
#         else:
#             # Fallback: use whole completion as answer
#             cer = calculate_cer(completion, correct_txt)
#             answer_reward = max(0, 2 * (1 - cer))  # Lower max reward without tags
#             total_reward += answer_reward
#             components['answer'] = answer_reward
#             components['cer'] = cer
        
#         # Component 3: Thinking quality reward (0-1 points)
#         think_reward = 0.0
#         if has_think:
#             think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
#             if think_match:
#                 think_content = think_match.group(1).strip()
#                 think_len = len(think_content)
                
#                 # Reward thinking that's not too short or too long
#                 if 20 < think_len < 500:
#                     think_reward += 0.5
                    
#                     # Bonus if thinking mentions errors/corrections
#                     if any(word in think_content.lower() for word in 
#                            ['błąd', 'literówk', 'poprawi', 'zmieni', 'powinno', 'zamiast']):
#                         think_reward += 0.5
                
#                 total_reward += think_reward
#                 components['think'] = think_reward
        
#         rewards.append(total_reward)
#         reward_components.append(components)
        
#         # Debug first few rewards
#         if not hasattr(reward_function, '_rewards_printed') and idx < 2:
#             print(f"\n  Sample {idx} reward breakdown:")
#             print(f"    - Format: {'✓' if proper_format else '✗'} (think={has_think}, answer={has_answer})")
#             print(f"    - Components: {components}")
#             print(f"    - Total reward: {total_reward:.2f} / 5.0")
#             print(f"    - Completion (first 200 chars): {completion[:200]}...")
    
#     # Mark as printed
#     if not hasattr(reward_function, '_rewards_printed'):
#         reward_function._rewards_printed = True
        
#         # Calculate completion statistics
#         completion_lengths = [len(c) for c in completions]
        
#         print(f"\n  Completion statistics:")
#         print(f"    - Mean length: {sum(completion_lengths)/len(completion_lengths):.0f} chars")
#         print(f"    - Min length: {min(completion_lengths)}")
#         print(f"    - Max length: {max(completion_lengths)}")
        
#         print(f"\n  Reward statistics:")
#         print(f"    - Mean: {sum(rewards)/len(rewards):.2f}")
#         print(f"    - Min: {min(rewards):.2f}")
#         print(f"    - Max: {max(rewards):.2f}")
#         print(f"    - Non-zero count: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")
    
#     return rewards

# ============================================================================
# Main Reward Function
# ============================================================================

def reward_function(prompts, completions, correct_text, typo_text, **kwargs):
    """
    Ultra-simple character-by-character reward function.
    
    Scoring:
    - +1 point for each correct character at correct position
    - Normalized by length of correct text
    - +2 bonus points for perfect match
    - Max score: 3.0 (1.0 for perfect char match + 2.0 bonus)
    
    Args:
        prompts: List of prompts
        completions: List of completions
        correct_text: List of correct texts (from dataset column)
        typo_text: List of typo texts (from dataset column)
        **kwargs: Additional columns from dataset
    """    
    # Convert completions to strings if they're not already
    processed_completions = []
    for idx, comp in enumerate(completions):
        try:
            if isinstance(comp, str):
                processed_completions.append(comp.strip())
            elif isinstance(comp, list):
                if comp and isinstance(comp[0], dict) and 'content' in comp[0]:
                    processed_completions.append(comp[0]['content'].strip())
                else:
                    processed_completions.append(str(comp).strip())
            else:
                processed_completions.append(str(comp).strip())
        except Exception as e:
            print(f"⚠️  Error processing completion {idx}: {e}")
            processed_completions.append("")
    
    completions = processed_completions
    
    rewards = []
    reward_components = []
    
    for idx, (completion, correct_txt) in enumerate(zip(completions, correct_text)):
        try:
            # Count matching characters at same positions
            matches = 0
            max_len = max(len(completion), len(correct_txt))
            
            for i, (c1, c2) in enumerate(zip(completion, correct_txt)):
                if c1 == c2:
                    matches += 1
            
            # Normalize: 1.0 for perfect character match
            if len(correct_txt) > 0:
                char_accuracy = matches / len(correct_txt)
            else:
                char_accuracy = 0.0
            
            reward = char_accuracy
            
            # Perfect match bonus: +2 points
            is_perfect = (completion == correct_txt)
            if is_perfect:
                reward += 2.0
            
            # Length penalty: if completion is way too long or short
            len_ratio = len(completion) / max(len(correct_txt), 1)
            if len_ratio < 0.5 or len_ratio > 2.0:
                reward *= 0.5
            
            rewards.append(reward)
            components = {
                'char_accuracy': char_accuracy,
                'matches': matches,
                'expected_len': len(correct_txt),
                'actual_len': len(completion),
                'perfect_match': is_perfect,
            }
            reward_components.append(components)
            
            # Debug first few rewards
            if not hasattr(reward_function, '_rewards_printed') and idx < 2:
                print(f"\n  Sample {idx} reward breakdown:")
                print(f"    - Char matches: {matches}/{len(correct_txt)} ({char_accuracy*100:.1f}%)")
                print(f"    - Length: {len(completion)} vs {len(correct_txt)}")
                print(f"    - Perfect match: {'✓ +2.0' if is_perfect else '✗'}")
                print(f"    - TOTAL: {reward:.3f}/3.0")
                print(f"    - Predicted: {completion[:150]}...")
                print(f"    - Expected: {correct_txt[:150]}...")
        
        except Exception as e:
            print(f"\n⚠️  ERROR calculating reward for completion {idx}: {e}")
            import traceback
            traceback.print_exc()
            rewards.append(0.0)
            reward_components.append({'error': str(e)})
    
    # Mark as printed and log metrics to wandb
    if not hasattr(reward_function, '_rewards_printed'):
        reward_function._rewards_printed = True
        
        # Calculate completion statistics
        completion_lengths = [len(c) for c in completions]
        
        print(f"\n  Completion statistics:")
        print(f"    - Mean length: {sum(completion_lengths)/len(completion_lengths):.0f} chars")
        print(f"    - Min length: {min(completion_lengths)}")
        print(f"    - Max length: {max(completion_lengths)}")
        
        print(f"\n  Reward statistics:")
        print(f"    - Mean: {sum(rewards)/len(rewards):.2f}")
        print(f"    - Min: {min(rewards):.2f}")
        print(f"    - Max: {max(rewards):.2f}")
        print(f"    - Non-zero count: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")
    
    # Log detailed metrics to wandb
    if hasattr(reward_function, '_call_count'):
        reward_function._call_count += 1
    else:
        reward_function._call_count = 1
    
    # Aggregate metrics
    batch_char_accs = [comp.get('char_accuracy', 0) for comp in reward_components]
    batch_perfect = [comp.get('perfect_match', False) for comp in reward_components]
    
    # Log to wandb
    try:
        log_dict = {
            'reward/char_accuracy': sum(batch_char_accs) / len(batch_char_accs),
            'reward/perfect_match_ratio': sum(1 for p in batch_perfect if p) / len(batch_perfect),
            'reward/mean': sum(rewards) / len(rewards),
            'reward/min': min(rewards),
            'reward/max': max(rewards),
            'reward/std': (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5,
            'reward/call_count': reward_function._call_count,
        }
        
        # Log example outputs every 10 calls
        if reward_function._call_count % 10 == 0:
            examples_table = wandb.Table(columns=[
                "Typo Text", 
                "Generated", 
                "Correct Text", 
                "Reward",
                "Char Match %",
                "Perfect"
            ])
            
            num_examples = min(3, len(completions))
            for i in range(num_examples):
                comp = reward_components[i] if i < len(reward_components) else {}
                examples_table.add_data(
                    typo_text[i][:200],
                    completions[i][:300],
                    correct_text[i][:200],
                    f"{rewards[i]:.3f}",
                    f"{comp.get('char_accuracy', 0)*100:.1f}%",
                    "✓" if comp.get('perfect_match', False) else "✗"
                )
            
            log_dict['examples/completions'] = examples_table
        
        wandb.log(log_dict)
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")
    
    return rewards

# Log reward function code to wandb for reproducibility
import inspect
reward_function_code = inspect.getsource(reward_function)
wandb.config.update({
    "reward_function_code": reward_function_code,
    "reward_max_score": 3.0,
    "reward_components": {
        "char_match": "0-1 (normalized by correct text length)",
        "perfect_match_bonus": "+2.0 (if exact match)",
        "length_penalty": "0.5x if len ratio < 0.5 or > 2.0"
    }
})

# Also log as artifact for better visibility
artifact = wandb.Artifact('reward_function', type='code')
artifact.add_file(__file__, name='train_grpo_typo.py')
wandb.log_artifact(artifact)

print("✓ Reward function logged to wandb")

# ============================================================================
# Load Model
# ============================================================================

print("\nLoading model and tokenizer...")
device = torch.device(DEVICE)
device_id = int(DEVICE.split(":")[-1])
print(f"Using device: {device} (GPU {device_id})")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": device_id},  # Force to specific GPU
)

# NOTE: torch.compile() nie działa z PEFT/LoRA w tym setupie
# Dostajemy KeyError: '_orig_mod' w extract_model_from_parallel
# Zostawiamy model bez compile - vLLM i tak bardzo przyśpiesza

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Diagnostyka tokenizera
print(f"✓ Model and tokenizer loaded on {device}")
print(f"\nTokenizer diagnostics:")
print(f"  - EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
print(f"  - PAD token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
print(f"  - BOS token: '{tokenizer.bos_token}' (id: {tokenizer.bos_token_id})")
print(f"  - Has chat template: {tokenizer.chat_template is not None}")

if tokenizer.chat_template:
    print(f"\nChat template (first 300 chars):\n{tokenizer.chat_template[:300]}...")
else:
    print(f"  ⚠️  No chat template found!")

# ============================================================================
# Load Datasets (now that we have tokenizer)
# ============================================================================

print("\nLoading datasets...")
train_dataset = load_typo_dataset("broken_text/dataset_train_grpo_no_first_last.csv", tokenizer)
eval_dataset = load_typo_dataset("broken_text/dataset_test_no_first_last.csv", tokenizer).select(range(50))  # Smaller eval set

# OPCJA: Dla szybszego testowania użyj mniejszego datasetu
# Odkomentuj poniższą linię żeby użyć tylko 500 samples do treningu:
# train_dataset = train_dataset.select(range(200))
# train_dataset = train_dataset.select(range(3000))

print(f"✓ Train: {len(train_dataset)} samples")
print(f"✓ Eval: {len(eval_dataset)} samples")

# Show example
print("\n" + "="*80)
print("Example from dataset (formatted prompt):")
print("="*80)
example = train_dataset[0]
print(f"Prompt (first 500 chars):\n{example['prompt'][:500]}...")
print(f"\nCorrect text:\n{example['correct_text'][:200]}...")
print(f"\nTypo text:\n{example['typo_text'][:200]}...")
print("="*80 + "\n")

# ============================================================================
# Test Generation BEFORE Training
# ============================================================================

print("\n" + "="*80)
print("Testing model BEFORE training...")
print("="*80)

test_example = train_dataset[0]
test_prompt = test_example['prompt']  # Already formatted with chat template
test_correct = test_example['correct_text']
test_typo = test_example['typo_text']

# ============================================================================
# LoRA Configuration
# ============================================================================

print("Configuring LoRA...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
print(f"✓ LoRA config created (r={peft_config.r}, alpha={peft_config.lora_alpha})")

# ============================================================================
# Training Configuration
# ============================================================================

training_args = GRPOConfig(
    output_dir="./models/grpo_checkpoints",
    num_train_epochs=1,  # Więcej epok
    
    # GRPO specific - ZOPTYMALIZOWANE NA STABILNOŚĆ
    num_generations=4,  # Zmniejszone z 4 do 2 dla stabilności (i 2x szybciej!)
    max_completion_length=1024,  # Zmniejszone z 1024 (wystarczy dla <think>+<answer>)
    temperature=0.7,  # Trochę więcej diversity
    
    # Generation parameters dla stabilności
    remove_unused_columns=False,  # CRITICAL: Keep all dataset columns for reward function!
    
    # Batch sizes - ZOPTYMALIZOWANE NA STABILNOŚĆ
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Zwiększone żeby effective batch = 8
    per_device_eval_batch_size=4, 
    
    # Optimization
    learning_rate=1e-4,  # Niższy stabilny LR
    lr_scheduler_type="cosine",  # Dodaj scheduler
    warmup_ratio=0.01,  # Używaj ratio zamiast steps (1% treningu)
    optim="paged_adamw_8bit",  # Standardowy optimizer (paged_adamw_8bit może powodować problemy)
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Logging & Saving - RZADZIEJ dla szybkości
    logging_steps=5,  # Co 5 kroków zamiast 1 (mniej I/O)
    eval_strategy="steps",
    eval_steps=50,  # Co 100 kroków (eval jest wolny)
    save_strategy="steps",
    save_steps=20,  # Rzadziej zapisuj (I/O jest wolne)
    save_total_limit=2,  # Mniej checkpointów
    
    # Compute
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Better for LoRA
    
    # W&B
    report_to="wandb",
    
    # GRPO optimization - PRÓBUJEMY PONOWNIE Z KONSERWATYWNYMI USTAWIENIAMI
    # Z bardzo małym batch_size i num_generations=2, vLLM powinno działać
    # Jeśli crashuje z shape error, zmień na False
    use_vllm=True,  # 3-5x szybciej! Ale może być buggy...
)

print("\n" + "="*80)
print("GRPO Configuration (STABILITY OPTIMIZED):")
print(f"  - Generations per prompt: {training_args.num_generations}")
print(f"  - Max completion length: {training_args.max_completion_length}")
print(f"  - Temperature: {training_args.temperature}")
print(f"  - Batch size per device: {training_args.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Completions per forward: {training_args.per_device_train_batch_size * training_args.num_generations}")
print(f"  - Use vLLM: {training_args.use_vllm} ({'FAST but buggy' if training_args.use_vllm else 'STABLE but slower'})")
print(f"  - Remove unused columns: {training_args.remove_unused_columns}")
print("="*80 + "\n")

# ============================================================================
# Initialize Trainer
# ============================================================================

print("\n🔧 Initializing GRPOTrainer...")
if training_args.use_vllm:
    print("  ⏳ Using vLLM - initialization may take 2-5 minutes...")
else:
    print("  ✅ Using standard generation (no vLLM) - more stable!")
print("     Please wait...\n")

# Initialize best model tracker
best_model_tracker = BestModelTracker()
print("✓ Best model tracker initialized")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[best_model_tracker],
)

print("✓ GRPOTrainer initialized successfully!")

# Print trainable parameters
trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in trainer.model.parameters())
print(f"\n✓ Model prepared with LoRA:")
print(f"  - Trainable params: {trainable_params:,} ({100*trainable_params/all_params:.2f}%)")
print(f"  - All params: {all_params:,}")

# ============================================================================
# Training
# ============================================================================

print("\n" + "="*80)
print("Starting GRPO training...")
print("="*80 + "\n")

# Test reward function before training
print("Testing reward function with example...")
example = train_dataset[0]
test_completions = [
    example['correct_text'],  # Perfect match
    example['typo_text'],     # No correction
    example['correct_text'][:len(example['correct_text'])//2],  # Partial
]
test_rewards = reward_function(
    prompts=[example['prompt']] * len(test_completions),
    completions=test_completions,
    correct_text=[example['correct_text']] * len(test_completions),
    typo_text=[example['typo_text']] * len(test_completions)
)
print("Example rewards:")
for comp, rew in zip(test_completions, test_rewards):
    print(f"  Reward {rew:.2f}: {comp[:80]}...")
print()

try:
    trainer.train()
    print("\n✓ Training complete!")
except Exception as e:
    print(f"\n⚠️  Training error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Save Model (Best Checkpoint)
# ============================================================================

print("\n" + "="*80)
print("Selecting best model for final save...")
print("="*80)

# Check if we have a best checkpoint
if best_model_tracker.best_checkpoint and os.path.exists(best_model_tracker.best_checkpoint):
    print(f"\n🏆 Using BEST checkpoint with reward: {best_model_tracker.best_reward:.4f}")
    print(f"   Checkpoint: {best_model_tracker.best_checkpoint}")
    
    # Load the best checkpoint
    print("\nLoading best checkpoint...")
    best_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device_id},
    )
    best_model = PeftModel.from_pretrained(best_model, best_model_tracker.best_checkpoint)
    print("✓ Best checkpoint loaded")
    
    model_to_save = best_model
    save_source = f"best checkpoint (step {best_model_tracker.best_checkpoint.split('-')[-1]})"
else:
    print("\n⚠️  No best checkpoint found, using final model")
    print(f"   This might happen if evaluation didn't run or no checkpoints were saved")
    model_to_save = trainer.model
    save_source = "final training state"

print(f"\nSaving model from: {save_source}")

# Save LoRA adapters (lightweight)
lora_output = f"{OUTPUT_NAME}-lora"
if hasattr(model_to_save, 'save_pretrained'):
    model_to_save.save_pretrained(lora_output)
else:
    trainer.save_model(lora_output)
print(f"✓ LoRA adapters saved to {lora_output}")

# Merge LoRA with base model and save full model
print("\nMerging LoRA adapters with base model...")
merged_model = model_to_save.merge_and_unload()
merged_model.save_pretrained(OUTPUT_NAME)
tokenizer.save_pretrained(OUTPUT_NAME)
print(f"✓ Full merged model saved to {OUTPUT_NAME}")
print(f"  Model size: ~9GB (full weights)")
if best_model_tracker.best_checkpoint:
    print(f"  Best reward: {best_model_tracker.best_reward:.4f}")
print(f"  Source: {save_source}")

# Log to wandb
wandb.log({
    'final_model/best_reward': best_model_tracker.best_reward if best_model_tracker.best_checkpoint else 0,
    'final_model/from_best_checkpoint': bool(best_model_tracker.best_checkpoint),
})

# ============================================================================
# Test Generation
# ============================================================================

print("\n" + "="*80)
print("Testing trained model (from best checkpoint)...")
print("="*80)

# Use the merged model for testing
test_model = merged_model
test_model.eval()

test_example = eval_dataset[0]
test_prompt = test_example['prompt']
test_correct = test_example['correct_text']

inputs = tokenizer(test_prompt, return_tensors="pt").to(test_model.device)
with torch.no_grad():
    outputs = test_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
    )
    
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_response = generated_text[len(test_prompt):].strip()

print(f"\nPrompt:\n{test_prompt[:200]}...\n")
print(f"Generated:\n{generated_response[:400]}...\n")
print(f"Expected answer:\n{test_correct[:200]}...\n")

# Calculate final reward
final_reward = reward_function(
    prompts=[test_prompt],
    completions=[generated_response],
    correct_text=[test_correct],
    typo_text=[test_example['typo_text']]
)[0]
print(f"Final reward: {final_reward:.3f} / 3.0")

# Calculate character matches
matches = sum(1 for c1, c2 in zip(generated_response, test_correct) if c1 == c2)
char_acc = matches / len(test_correct) if len(test_correct) > 0 else 0

print(f"\nQuality metrics:")
print(f"  Char matches: {matches}/{len(test_correct)} ({char_acc*100:.1f}%)")
print(f"  Length: {len(generated_response)} vs {len(test_correct)}")
print(f"  Perfect match: {'✓' if generated_response == test_correct else '✗'}")

print("\n" + "="*80)
print("🎉 GRPO TRAINING COMPLETED!")
print("="*80)

wandb.finish()

