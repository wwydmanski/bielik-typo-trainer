#!/usr/bin/env python3
"""
GRPO Training for Typo Correction with Reasoning

Based on DeepSeek R1 approach - train model to:
1. Think through the typos (<think>[thinking]</think>)
2. Provide corrected text (<answer>...</answer>)

Uses Group Relative Policy Optimization (GRPO) from TRL.
"""

import os
import re
import pandas as pd
import torch
import Levenshtein
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
import wandb

# ============================================================================
# Configuration
# ============================================================================

# MODEL_NAME = "speakleash/Bielik-4.5B-v3.0-Instruct"
MODEL_NAME = "./models/Bielik-4.5B-pre-thinking-typo-fixer"
OUTPUT_NAME = "./models/Bielik-4.5B-grpo-thinking-typo-fixer-v0.2"
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
# Helper Functions
# ============================================================================

def calculate_cer(predicted: str, expected: str) -> float:
    """Calculate Character Error Rate"""
    distance = Levenshtein.distance(predicted, expected)
    return distance / max(len(expected), 1)

def find_typo_words(typo_text: str, correct_text: str) -> list[tuple[str, str]]:
    """
    Find words that differ between typo and correct text.
    
    Returns:
        List of (typo_word, correct_word) tuples
    """
    import difflib
    
    typo_words = typo_text.lower().split()
    correct_words = correct_text.lower().split()
    
    # Use difflib to align sequences
    matcher = difflib.SequenceMatcher(None, typo_words, correct_words)
    
    typo_pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Word was changed
            for typo_w, correct_w in zip(typo_words[i1:i2], correct_words[j1:j2]):
                if typo_w != correct_w:
                    typo_pairs.append((typo_w, correct_w))
        elif tag == 'delete':
            # Word was removed (extra in typo)
            for typo_w in typo_words[i1:i2]:
                typo_pairs.append((typo_w, ''))
        elif tag == 'insert':
            # Word was added (missing in typo)
            for correct_w in correct_words[j1:j2]:
                typo_pairs.append(('', correct_w))
    
    return typo_pairs

def extract_answer(completion: str) -> str:
    """Extract answer from <answer>...</answer> tags"""
    match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try to find content after "Odpowiedź:" or similar
    if "Odpowiedź:" in completion:
        return completion.split("Odpowiedź:")[-1].strip()
    return completion.strip()

def check_format(completion: str) -> tuple[bool, bool, bool, float]:
    """
    Check if completion follows the desired format.
    
    Returns:
        has_think: bool - whether <think> tags exist
        has_answer: bool - whether <answer> tags exist  
        proper_format: bool - whether tags are in correct order
        outside_ratio: float - ratio of text outside tags (0.0 = all inside, 1.0 = all outside)
    """
    has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
    
    # Check if format is proper (think before answer)
    if has_think and has_answer:
        think_pos = completion.find('<think>')
        answer_pos = completion.find('<answer>')
        proper_format = think_pos < answer_pos
    else:
        proper_format = False
    
    # Calculate how much text is OUTSIDE tags
    outside_ratio = 0.0
    total_len = len(completion.strip())
    
    if total_len > 0:
        # Remove everything inside <think>...</think> and <answer>...</answer>
        text_copy = completion
        
        if has_think:
            text_copy = re.sub(r'<think>.*?</think>', '', text_copy, flags=re.DOTALL)
        if has_answer:
            text_copy = re.sub(r'<answer>.*?</answer>', '', text_copy, flags=re.DOTALL)
        
        # Remove the tag markers themselves (shouldn't be there if properly closed)
        text_copy = text_copy.replace('<think>', '').replace('</think>', '')
        text_copy = text_copy.replace('<answer>', '').replace('</answer>', '')
        
        # Count remaining non-whitespace characters
        outside_text = text_copy.strip()
        outside_len = len(outside_text)
        
        outside_ratio = outside_len / total_len if total_len > 0 else 0.0
    
    return has_think, has_answer, proper_format, outside_ratio

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
    df = pd.read_csv(csv_path, header=None, names=['correct', 'typos'], dtype=str)
    
    data = []
    for _, row in df.iterrows():
        correct_text = str(row['correct']).strip()
        typo_text = str(row['typos']).strip()
        
        # Create user prompt that encourages reasoning
        user_content = f"""Napraw literówki w poniższym tekście. 

Najpierw w znacznikach <think></think> opisz jakie błędy zauważyłeś, a następnie podaj poprawiony tekst w znacznikach <answer></answer>.

Tekst z błędami:
`{typo_text}`
"""
        
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
# Reward Components (Helper Functions)
# ============================================================================

def calculate_format_reward(completion: str) -> tuple[float, dict]:
    """Calculate format reward (0-1 points)"""
    has_think, has_answer, proper_format, outside_ratio = check_format(completion)
    components = {}
    
    format_reward = 0.0
    if proper_format:
        format_reward = 1.0
    elif has_think and has_answer:
        format_reward = 0.5
    elif has_answer:
        format_reward = 0.3
    
    # Penalty for text outside tags
    if outside_ratio > 0.1:
        outside_penalty = min(0.5, outside_ratio)
        format_reward = max(0, format_reward - outside_penalty)
        components['outside_penalty'] = outside_penalty
        components['outside_ratio'] = outside_ratio
    
    components['has_think'] = has_think
    components['has_answer'] = has_answer
    components['proper_format'] = proper_format
    
    return format_reward, components

def calculate_answer_reward(completion: str, correct_text: str) -> tuple[float, dict]:
    """Calculate answer quality reward (0-4 points)"""
    components = {}
    
    # Extract predicted text
    has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
    if has_answer:
        predicted_text = extract_answer(completion)
    else:
        predicted_text = completion.strip()
    
    # 2a. Character-level accuracy (0-2.5 points)
    cer = calculate_cer(predicted_text, correct_text)
    char_reward = max(0, 2.5 * (1 - cer))
    
    # 2b. Word-level accuracy (0-1.5 points)
    pred_words = predicted_text.split()
    correct_words = correct_text.split()
    word_distance = Levenshtein.distance(' '.join(pred_words), ' '.join(correct_words))
    max_word_len = max(len(' '.join(pred_words)), len(' '.join(correct_words)))
    word_accuracy = 1 - (word_distance / max_word_len) if max_word_len > 0 else 0
    word_reward = max(0, 1.5 * word_accuracy)
    
    answer_reward = char_reward + word_reward
    
    # Bonus: exact match
    if predicted_text == correct_text:
        answer_reward += 0.5
        components['exact_match'] = True
    
    # Penalty: if answer is way too long
    len_ratio = len(predicted_text) / max(len(correct_text), 1)
    if len_ratio > 1.2:
        answer_reward *= 0.8
        components['length_penalty'] = True
    
    components['cer'] = cer
    components['word_acc'] = word_accuracy
    components['predicted_text'] = predicted_text
    
    return answer_reward, components

def calculate_thinking_reward(completion: str, typo_text: str, correct_text: str) -> tuple[float, dict]:
    """Calculate thinking quality reward (0-2.5 points) - IMPROVED"""
    components = {}
    think_reward = 0.0
    
    # Check if thinking exists
    has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
    if not has_think:
        return 0.0, components
    
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    if not think_match:
        return 0.0, components
    
    think_content = think_match.group(1).strip()
    think_content_lower = think_content.lower()
    think_len = len(think_content)
    
    # 3a. Length reward (0-0.5)
    if think_len < 10:
        length_reward = 0
    elif think_len < 50:
        length_reward = 0.25 * (think_len / 50)
    elif think_len <= 300:
        length_reward = 0.5
    elif think_len <= 500:
        length_reward = 0.5 * (1 - (think_len - 300) / 200)
    else:
        length_reward = 0.1
    
    think_reward += length_reward
    
    # 3b. Content quality (0-0.5): mentions error-related words
    error_words = ['błąd', 'literówk', 'poprawi', 'zmieni', 'powinno', 
                   'zamiast', 'nieprawidłow', 'pomył', 'źle', 'poprawnie']
    mentions_count = sum(1 for word in error_words if word in think_content_lower)
    content_reward = min(0.5, mentions_count * 0.15)
    think_reward += content_reward
    
    # 3c. Specificity - mentions actual TYPO words (0-0.75)
    typo_pairs = find_typo_words(typo_text, correct_text)
    # DON'T filter by length - include ALL typos
    typo_words_only = [typo for typo, correct in typo_pairs if typo]
    
    # Check how many typos are mentioned
    mentioned_typos = sum(1 for typo in typo_words_only 
                         if typo.lower() in think_content_lower)
    
    # Reward PROPORTIONALLY - encourage mentioning ALL typos
    if typo_words_only:
        typo_coverage = mentioned_typos / len(typo_words_only)
        typo_reward = 0.75 * typo_coverage
    else:
        typo_reward = 0
    
    think_reward += typo_reward
    
    # 3d. BONUS: mentions CORRECTIONS (0-0.75) - NEW!
    correct_words_only = [correct for typo, correct in typo_pairs if correct]
    
    # Check how many corrections are mentioned
    mentioned_corrections = sum(1 for correct in correct_words_only 
                               if correct.lower() in think_content_lower)
    
    # Reward PROPORTIONALLY - encourage mentioning ALL corrections
    if correct_words_only:
        correction_coverage = mentioned_corrections / len(correct_words_only)
        correction_reward = 0.75 * correction_coverage
    else:
        correction_reward = 0
    
    think_reward += correction_reward
    
    components['think_len'] = think_len
    components['think_mentions'] = mentions_count
    components['typos_mentioned'] = mentioned_typos
    components['typos_total'] = len(typo_words_only)
    components['typo_coverage'] = mentioned_typos / len(typo_words_only) if typo_words_only else 0
    components['corrections_mentioned'] = mentioned_corrections
    components['corrections_total'] = len(correct_words_only)
    components['correction_coverage'] = mentioned_corrections / len(correct_words_only) if correct_words_only else 0
    
    return think_reward, components

def apply_final_penalties(total_reward: float, has_think: bool, proper_format: bool, components: dict) -> tuple[float, dict]:
    """Apply final penalties based on format requirements"""
    # CRITICAL: Enforce thinking requirement
    if not has_think:
        original_reward = total_reward
        total_reward = min(total_reward, 3.0)
        if original_reward > 3.0:
            components['think_penalty'] = original_reward - total_reward
    elif not proper_format:
        original_reward = total_reward
        total_reward = min(total_reward, 4.0)
        if original_reward > 4.0:
            components['format_penalty'] = original_reward - total_reward
    
    return total_reward, components

# ============================================================================
# Main Reward Function
# ============================================================================

def reward_function(prompts, completions, correct_text, typo_text, **kwargs):
    """
    Advanced multi-component reward function for typo correction with reasoning.
    
    Reward Components (total max ~8.0 points):
    1. Format (0-1): Proper structure with <think> and <answer> tags (REQUIRED!)
    2. Answer quality (0-4): Character and word-level accuracy
    3. Thinking quality (0-2.5): Quality and relevance of reasoning (REQUIRED!)
       - 0-0.5: Length appropriateness
       - 0-0.5: Error-related vocabulary
       - 0-0.75: Coverage of TYPO words (proportional to all typos found)
       - 0-0.75: Coverage of CORRECTION words (proportional to all corrections)
    4. Bonus rewards (0-0.5): Exact matches, etc.
    
    IMPORTANT: Without proper <think> tag, max reward is capped at ~3.0 (37% of max)
    
    Args:
        prompts: List of prompts
        completions: List of completions
        correct_text: List of correct texts (from dataset column)
        typo_text: List of typo texts (from dataset column)
        **kwargs: Additional columns from dataset
    """    
    # Convert completions to strings if they're not already
    # (GRPOTrainer may pass them as various formats)
    processed_completions = []
    for idx, comp in enumerate(completions):
        try:
            if isinstance(comp, str):
                processed_completions.append(comp)
            elif isinstance(comp, list):
                # If it's a list, join it or take first element
                if comp and isinstance(comp[0], dict) and 'content' in comp[0]:
                    # Messages format: [{"role": "assistant", "content": "..."}]
                    processed_completions.append(comp[0]['content'])
                else:
                    processed_completions.append(str(comp))
            else:
                processed_completions.append(str(comp))
        except Exception as e:
            print(f"⚠️  Error processing completion {idx}: {e}")
            print(f"   Type: {type(comp)}, Value: {comp}")
            # Add empty string as fallback to maintain list length
            processed_completions.append("")
    
    completions = processed_completions
    
    # CRITICAL: Handle dataset column replication
    # GRPOTrainer generates num_generations completions per prompt
    # but dataset columns (correct_text, typo_text) may NOT be automatically replicated
    num_completions = len(completions)
    num_dataset_rows = len(correct_text)
    
    rewards = []
    reward_components = []  # Track components for debugging
    
    for idx, (completion, correct_txt, typo_txt) in enumerate(zip(completions, correct_text, typo_text)):
        try:
            # Calculate all reward components using helper functions
            format_reward, format_comp = calculate_format_reward(completion)
            answer_reward, answer_comp = calculate_answer_reward(completion, correct_txt)
            think_reward, think_comp = calculate_thinking_reward(completion, typo_txt, correct_txt)
            
            # Combine components
            components = {**format_comp, **answer_comp, **think_comp}
            components['format'] = format_reward
            components['answer'] = answer_reward
            components['think'] = think_reward
            
            # Calculate total reward
            total_reward = format_reward + answer_reward + think_reward
            
            # Apply final penalties
            total_reward, components = apply_final_penalties(
                total_reward,
                format_comp.get('has_think', False),
                format_comp.get('proper_format', False),
                components
            )
            
            rewards.append(total_reward)
            reward_components.append(components)
            
            # Debug first few rewards
            if not hasattr(reward_function, '_rewards_printed') and idx < 2:
                proper_format = components.get('proper_format', False)
                has_think = components.get('has_think', False)
                cer = components.get('cer', 0)
                word_accuracy = components.get('word_acc', 0)
                
                print(f"\n  Sample {idx} reward breakdown:")
                print(f"    - Format: {format_reward:.2f} ({'✓' if proper_format else '✗'})")
                print(f"    - Answer: {answer_reward:.2f} (CER={cer:.3f}, WordAcc={word_accuracy:.3f})")
                if has_think:
                    typo_cov = components.get('typo_coverage', 0)
                    corr_cov = components.get('correction_coverage', 0)
                    typos_mentioned = components.get('typos_mentioned', 0)
                    typos_total = components.get('typos_total', 0)
                    corr_mentioned = components.get('corrections_mentioned', 0)
                    corr_total = components.get('corrections_total', 0)
                    print(f"    - Think: {think_reward:.2f}")
                    print(f"      • Length: {components.get('think_len', 0)} chars, mentions: {components.get('think_mentions', 0)}")
                    print(f"      • Typo coverage: {typos_mentioned}/{typos_total} ({typo_cov*100:.0f}%)")
                    print(f"      • Correction coverage: {corr_mentioned}/{corr_total} ({corr_cov*100:.0f}%)")
                else:
                    print(f"    - Think: 0.00 (❌ MISSING - reward capped at 3.0!)")
                if 'think_penalty' in components:
                    print(f"    - ⚠️  NO THINKING PENALTY: -{components['think_penalty']:.2f}")
                if 'format_penalty' in components:
                    print(f"    - ⚠️  BAD FORMAT PENALTY: -{components['format_penalty']:.2f}")
                if 'outside_penalty' in components:
                    print(f"    - ⚠️  TEXT OUTSIDE TAGS: {components['outside_ratio']*100:.1f}% (penalty: -{components['outside_penalty']:.2f})")
                print(f"    - TOTAL: {total_reward:.2f} / 8.0")
                print(f"    - Completion (first 200 chars): {completion[:200]}...")
        
        except Exception as e:
            # If anything goes wrong calculating reward, give 0 and continue
            print(f"\n⚠️  ERROR calculating reward for completion {idx}: {e}")
            print(f"   Completion type: {type(completion)}")
            print(f"   Completion (first 100 chars): {str(completion)[:100]}...")
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
    
    # Log detailed metrics to wandb on every call
    if hasattr(reward_function, '_call_count'):
        reward_function._call_count += 1
    else:
        reward_function._call_count = 1
    
    # Aggregate metrics from all completions in this batch
    batch_cers = [comp.get('cer', 0) for comp in reward_components if 'cer' in comp]
    batch_word_accs = [comp.get('word_acc', 0) for comp in reward_components if 'word_acc' in comp]
    batch_format_scores = [comp.get('format', 0) for comp in reward_components]
    batch_think_scores = [comp.get('think', 0) for comp in reward_components]
    batch_answer_scores = [comp.get('answer', 0) for comp in reward_components]
    
    # Calculate word error rate (inverse of word accuracy)
    batch_wers = [1 - acc for acc in batch_word_accs]
    
    # Log to wandb
    try:
        log_dict = {
            # Error rates
            'reward/character_error_rate': sum(batch_cers) / len(batch_cers) if batch_cers else 0,
            'reward/word_error_rate': sum(batch_wers) / len(batch_wers) if batch_wers else 0,
            'reward/word_accuracy': sum(batch_word_accs) / len(batch_word_accs) if batch_word_accs else 0,
            
            # Component scores
            'reward/format_score': sum(batch_format_scores) / len(batch_format_scores),
            'reward/thinking_score': sum(batch_think_scores) / len(batch_think_scores) if batch_think_scores else 0,
            'reward/answer_score': sum(batch_answer_scores) / len(batch_answer_scores),
            
            # Overall reward stats
            'reward/mean': sum(rewards) / len(rewards),
            'reward/min': min(rewards),
            'reward/max': max(rewards),
            'reward/std': (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5,
            
            # Format compliance
            'reward/has_think_ratio': sum(1 for comp in reward_components if comp.get('think', 0) > 0) / len(reward_components),
            'reward/exact_match_ratio': sum(1 for comp in reward_components if comp.get('exact_match', False)) / len(reward_components),
            
            # Penalties
            'reward/think_penalty_count': sum(1 for comp in reward_components if 'think_penalty' in comp),
            'reward/format_penalty_count': sum(1 for comp in reward_components if 'format_penalty' in comp),
            'reward/outside_penalty_count': sum(1 for comp in reward_components if 'outside_penalty' in comp),
            
            # Outside ratio (tekst poza tagami)
            'reward/mean_outside_ratio': sum(comp.get('outside_ratio', 0) for comp in reward_components) / len(reward_components),
            
            # Typo identification
            'reward/mean_typos_identified': sum(comp.get('typo_words_identified', 0) for comp in reward_components) / len(reward_components),
            'reward/mean_typos_mentioned': sum(comp.get('think_specificity', 0) for comp in reward_components) / len(reward_components),
            
            # Call tracking
            'reward/call_count': reward_function._call_count,
        }
        
        # Log example outputs every 10 calls
        if reward_function._call_count % 10 == 0:
            # Create table with examples (show first 3 completions)
            examples_table = wandb.Table(columns=[
                "Typo Text", 
                "Generated Completion", 
                "Extracted Answer",
                "Correct Text", 
                "Reward",
                "CER",
                "Format OK",
                "Has Think",
                "Outside %",
                "Typos Mentioned"
            ])
            
            num_examples = min(3, len(completions))
            for i in range(num_examples):
                comp = reward_components[i] if i < len(reward_components) else {}
                
                # Extract answer from completion
                completion_text = completions[i]
                has_think, has_answer, proper_format, outside_ratio = check_format(completion_text)
                
                if has_answer:
                    answer_text = extract_answer(completion_text)
                else:
                    answer_text = completion_text[:100] + "..." if len(completion_text) > 100 else completion_text
                
                # Format typo mention info
                typos_id = comp.get('typo_words_identified', 0)
                typos_mentioned = comp.get('think_specificity', 0)
                typo_info = f"{typos_mentioned}/{typos_id}" if typos_id > 0 else "N/A"
                
                examples_table.add_data(
                    typo_text[i][:200],  # Truncate for readability
                    completion_text[:300],  # Show first 300 chars
                    answer_text[:200],
                    correct_text[i][:200],
                    f"{rewards[i]:.2f}",
                    f"{comp.get('cer', 0):.3f}",
                    "✓" if proper_format else "✗",
                    "✓" if has_think else "✗",
                    f"{outside_ratio*100:.1f}%",
                    typo_info
                )
            
            log_dict['examples/completions'] = examples_table
        
        wandb.log(log_dict)
    except Exception as e:
        # Don't fail training if wandb logging fails
        print(f"Warning: Failed to log to wandb: {e}")
    
    return rewards

# Log reward function code to wandb for reproducibility
import inspect
reward_function_code = inspect.getsource(reward_function)
wandb.config.update({
    "reward_function_code": reward_function_code,
    "reward_max_score": 6.0,
    "reward_components": {
        "format": "0-1 (requires <think> and <answer>)",
        "answer_quality": "0-4 (char + word accuracy)",
        "thinking_quality": "0-1.5 (length + content + specificity)",
        "penalties": "caps at 3.0 without <think>, 4.0 with bad format"
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
train_dataset = train_dataset.select(range(100))
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
    num_train_epochs=5,  # Więcej epok
    
    # GRPO specific - ZOPTYMALIZOWANE NA STABILNOŚĆ
    num_generations=4,
    max_completion_length=1024,  # Zmniejszone z 1024 (wystarczy dla <think>+<answer>)
    temperature=0.7,  # Trochę więcej diversity
    
    # Generation parameters dla stabilności
    remove_unused_columns=False,  # CRITICAL: Keep all dataset columns for reward function!
    
    # Batch sizes - ZOPTYMALIZOWANE NA STABILNOŚĆ
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,  # Zwiększone żeby effective batch = 8
    per_device_eval_batch_size=8, 
    
    # Optimization
    learning_rate=1e-5,  # Niższy stabilny LR
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
    f"<think>Widzę kilka błędów w tekście</think><answer>{example['correct_text']}</answer>",
    f"<answer>{example['correct_text']}</answer>",
    f"{example['correct_text']}",
    f"<think>błąd</think><answer>zły tekst</answer>",
]
test_rewards = reward_function(
    prompts=[example['prompt']] * len(test_completions),
    completions=test_completions,
    correct_text=[example['correct_text']] * len(test_completions),
    typo_text=[example['typo_text']] * len(test_completions)
)
print("Example rewards for different formats:")
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
print(f"Final reward: {final_reward:.2f} / 5.0")

# Check format
has_think, has_answer, proper_format, outside_ratio = check_format(generated_response)
print(f"\nFormat check:")
print(f"  ✓ Has <think>: {has_think}")
print(f"  ✓ Has <answer>: {has_answer}")
print(f"  ✓ Proper order: {proper_format}")
print(f"  ✓ Outside ratio: {outside_ratio*100:.1f}% (text outside tags)")

if has_answer:
    predicted_answer = extract_answer(generated_response)
    cer = calculate_cer(predicted_answer, test_correct)
    print(f"\nAnswer quality:")
    print(f"  CER: {cer*100:.2f}%")
    print(f"  Char accuracy: {(1-cer)*100:.2f}%")

print("\n" + "="*80)
print("🎉 GRPO TRAINING COMPLETED!")
print("="*80)

wandb.finish()

