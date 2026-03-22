import torch
import random
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

from config import SimulationConfig, PersonalityConfig, ModelConfig
from questionbank import QuestionBank
from models.baseagent_api import create_api_client, BaseAgentAPI
from evaluation.metrics import AnswerEvaluator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_questions(question_bank, category, train_ratio=0.6, seed=42):
    all_ids = question_bank.get_all_ids_by_category(category)
    
    random.seed(seed)
    shuffled_ids = all_ids.copy()
    random.shuffle(shuffled_ids)
    
    split_point = int(len(shuffled_ids) * train_ratio)
    train_ids = shuffled_ids[:split_point]
    test_ids = shuffled_ids[split_point:]
    
    return train_ids, test_ids

def save_exam_results(exam_history, output_path):
    records = []
    for item in exam_history:
        records.append({
            'question_id': item['question_id'],
            'question': item['question'],
            'agent_answer': item['answer'],
            'extracted_answer': item['extracted_answer'],
            'raw_answer': item['raw_answer'],
            'correct': item['correct'],
            'exact_match': item['exact_match'],
            'f1': item['f1'],
            'precision': item['precision'],
            'recall': item['recall']
        })
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exam results saved to {output_path}")

def save_final_results(config, personality_name, exam_summary, micro_macro_metrics, output_path):
    result = {
        'personality': personality_name,
        'topic': config.exam_topic,
        'num_questions': config.exam_config.num_questions,
        'student_temperature': config.model_config.student_temperature,
        'model_type': 'test_only_no_learning',
        
        'correct_count': exam_summary['correct_count'],
        'accuracy': exam_summary['accuracy'],
        'empty_answer_count': exam_summary['empty_answer_count'],
        'empty_answer_percentage': exam_summary['empty_answer_percentage'],
        'exam_timestamp': exam_summary['final_timestamp'],
        
        'micro_f1': micro_macro_metrics['micro_f1'],
        'macro_f1': micro_macro_metrics['macro_f1'],
        'micro_precision': micro_macro_metrics['micro_precision'],
        'micro_recall': micro_macro_metrics['micro_recall'],
        'macro_precision': micro_macro_metrics['macro_precision'],
        'macro_recall': micro_macro_metrics['macro_recall']
    }
    
    df = pd.DataFrame([result])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Final results saved to {output_path}")

def run_test_only(config, personality_config):
    set_seed(config.random_seed)
    
    print("="*80)
    print(f"TEST-ONLY MODE (No Learning): {personality_config.name}, Topic: {config.exam_topic}")
    print("="*80)
    
    question_bank = QuestionBank()
    print(f"Loaded question bank with {len(question_bank.categories)} categories")
    
    train_ids, test_ids = split_questions(
        question_bank, 
        config.exam_topic, 
        train_ratio=0.6, 
        seed=config.random_seed
    )
    print(f"Using test set: {len(test_ids)} questions")
    
    print("Creating API client...")
    api_client = create_api_client(config.model_config)
    agent = BaseAgentAPI(api_client, config.model_config, config.model_config.student_temperature)
    
    print("\n" + "="*80)
    print("EXAM PHASE (Direct Testing)")
    print("="*80)
    
    exam_question_ids = test_ids[:config.exam_config.num_questions]
    exam_history = []
    total_timestamp = 0
    
    for idx, question_id in enumerate(exam_question_ids, 1):
        problem = question_bank.get_question_by_id(question_id)
        
        # Simple prompt without memory
        prompt = f"""[INST] <<SYS>>
{personality_config.description}
<</SYS>>

Solve this math problem:
{problem['question']}

Output format: {{"answer": "put your final answer here"}}[/INST]"""
        
        response = agent.generate(prompt, max_new_tokens=500)
        total_timestamp += 1
        
        # Extract answer
        answer = agent.extract_content(response, 'answer')
        if not answer:
            answer = response
        
        # Evaluate
        is_correct_simple = bool(answer and answer.strip())
        metrics = AnswerEvaluator.evaluate_answer(answer, problem['extracted_answer'], problem['raw_answer'])
        
        is_correct_final = is_correct_simple and (metrics['exact_match'] or bool(answer.strip()))
        
        exam_history.append({
            'question_id': question_id,
            'question': problem['question'],
            'answer': answer,
            'extracted_answer': problem['extracted_answer'],
            'raw_answer': problem['raw_answer'],
            'correct': is_correct_final,
            'exact_match': metrics['exact_match'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
        
        print(f"Question {idx}/{len(exam_question_ids)} - Correct: {is_correct_final}")
    
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)
    
    # Calculate statistics
    correct_count = sum(1 for item in exam_history if item['correct'])
    empty_count = sum(1 for item in exam_history if not item['answer'] or item['answer'].strip() == "")
    accuracy = correct_count / len(exam_history) if exam_history else 0.0
    
    exam_summary = {
        'total_questions': len(exam_history),
        'correct_count': correct_count,
        'accuracy': accuracy,
        'empty_answer_count': empty_count,
        'empty_answer_percentage': (empty_count / len(exam_history) * 100) if exam_history else 0.0,
        'final_timestamp': total_timestamp
    }
    
    # Calculate micro/macro F1
    micro_macro_metrics = AnswerEvaluator.calculate_micro_macro_f1(exam_history)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Micro F1: {micro_macro_metrics['micro_f1']:.4f}")
    print(f"Macro F1: {micro_macro_metrics['macro_f1']:.4f}")
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save results
    exam_output_path = f"output/{config.exam_topic}/test_only/agent_{personality_config.name}_nolearning.csv"
    save_exam_results(exam_history, exam_output_path)
    
    result_output_path = f"output/{config.exam_topic}/test_only/result_nolearning.csv"
    save_final_results(config, personality_config.name, exam_summary, micro_macro_metrics, result_output_path)
    
    print("\n" + "="*80)
    print("TEST-ONLY COMPLETED")
    print("="*80)
    
    return {
        'exam_summary': exam_summary,
        'metrics': micro_macro_metrics
    }

if __name__ == "__main__":
    config = SimulationConfig(
        random_seed=42,
        exam_topic="Algebra",
        model_config=ModelConfig(model_type="api")
    )
    
    config.exam_config.num_questions = 20
    config.personality = PersonalityConfig.get_high_openness()
    
    results = run_test_only(config, config.personality)