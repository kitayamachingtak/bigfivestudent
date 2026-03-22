import torch
import random
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

from config import SimulationConfig, PersonalityConfig
from questionbank import QuestionBank
from retriever import RetrieverModel, QuestionRetriever
from storememory import MemoryStore
from timestamp import TimestampManager
from models.teacheragent import TeacherAgent
from models.studentlearning import StudentLearningAgent
from models.studentexam import StudentExamAgent
from environments.learningloop import LearningLoop
from environments.examloop import ExamLoop
from evaluation.metrics import AnswerEvaluator
from models.basemodel import create_shared_generator
from models.agent_factory import create_base_agent

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
            'correct_simple': item['correct_simple'],    # 简单匹配
            'exact_match': item['exact_match'],          # 精确匹配
            'correct_final': item['correct_final'],      # 综合判断（用于accuracy）
            'f1': item['metrics']['f1'],
            'precision': item['metrics']['precision'],
            'recall': item['metrics']['recall']
        })
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exam results saved to {output_path}")

def save_memory(memory_store, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    full_memory_data = {
        'memories': memory_store.get_all_memories(),
        'interaction_logs': memory_store.get_all_interaction_logs()
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_memory_data, f, indent=2)
    print(f"Memory and interaction logs saved to {output_path}")

def save_final_results(config, personality_name, learning_summary, exam_summary, 
                       micro_macro_metrics, output_path):
    result = {
        'personality': personality_name,
        'topic': config.exam_topic,
        'learning_rounds': config.learning_config.learning_rounds,
        'num_questions': config.exam_config.num_questions,
        'student_temperature': config.model_config.student_temperature,
        'teacher_temperature': config.model_config.teacher_temperature,
        
        'self_study_count': learning_summary['action_counts']['self_study'],
        'ask_teacher_count': learning_summary['action_counts']['ask_teacher'],
        'skip_count': learning_summary['action_counts']['skip'],
        'total_memories': learning_summary['total_memories'],
        'learning_timestamp': learning_summary['final_timestamp'],
        
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
        'macro_recall': micro_macro_metrics['macro_recall'],
        
        'total_timestamp': exam_summary['final_timestamp']
    }
    
    df = pd.DataFrame([result])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Final results saved to {output_path}")

def run_simulation(config, personality_config):
    set_seed(config.random_seed)
    
    # Print model type
    if config.model_config.model_type == "local":
        num_gpus = torch.cuda.device_count()
        print(f"Using LOCAL model")
        print(f"Detected {num_gpus} GPU(s)")
    else:
        print(f"Using API model: {config.model_config.api_model_name}")
    
    print("="*80)
    print(f"Starting simulation: {personality_config.name}, Topic: {config.exam_topic}")
    print("="*80)
    
    question_bank = QuestionBank()
    print(f"Loaded question bank with {len(question_bank.categories)} categories")
    
    train_ids, test_ids = split_questions(
        question_bank, 
        config.exam_topic, 
        train_ratio=0.6, 
        seed=config.random_seed
    )
    print(f"Split questions: {len(train_ids)} training, {len(test_ids)} testing")
    
    print("Initializing retriever model...")
    retriever_model = RetrieverModel(
        model_path="infly/inf-retriever-v1-1.5b",
        device=config.model_config.retriever_device
    )
    
    print("Building question retriever...")
    question_retriever = QuestionRetriever(
        question_bank=question_bank,
        retriever_model=retriever_model,
        category=config.exam_topic
    )
    

    print("Loading LLM...")
    from models.agent_factory import create_base_agent

    # Create ONE base agent and reuse it for all agents
    base_agent, shared_resource, tokenizer = create_base_agent(
        config.model_config, 
        config.model_config.student_temperature
    )

    print("Initializing teacher agent...")
    teacher_agent = TeacherAgent(
        base_agent=base_agent,
        model_config=config.model_config,
        question_retriever=question_retriever
    )
    print("Initializing student memory and timestamp...")
    memory_store = MemoryStore(retriever_model)
    timestamp_manager = TimestampManager(config.timestamp_config)

    print("Initializing student agents...")
    student_learning = StudentLearningAgent(
        base_agent=base_agent,
        model_config=config.model_config,
        personality=personality_config,
        question_retriever=question_retriever,
        memory_store=memory_store,
        timestamp_manager=timestamp_manager
    )

    student_exam = StudentExamAgent(
        base_agent=base_agent,
        model_config=config.model_config,
        personality=personality_config,
        memory_store=memory_store,
        timestamp_manager=timestamp_manager
    )

    print("\n" + "="*80)
    print("LEARNING PHASE")
    print("="*80)
    learning_loop = LearningLoop(student_learning, teacher_agent, config)
    learning_results = learning_loop.run()
    learning_summary = learning_loop.get_summary()
    
    print("\n" + "="*80)
    print("EXAM PHASE")
    print("="*80)
    exam_loop = ExamLoop(student_exam, question_bank, test_ids, config)
    exam_results = exam_loop.run()
    exam_summary = exam_loop.get_summary()
    
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)
    micro_macro_metrics = AnswerEvaluator.calculate_micro_macro_f1(
        student_exam.get_exam_history()
    )
    
    print(f"\nAccuracy: {exam_summary['accuracy']:.4f}")
    print(f"Micro F1: {micro_macro_metrics['micro_f1']:.4f}")
    print(f"Macro F1: {micro_macro_metrics['macro_f1']:.4f}")
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    lr = config.learning_config.learning_rounds
    
    exam_output_path = f"output/{config.exam_topic}/agent_{personality_config.name}_lr{lr}.csv"
    save_exam_results(student_exam.get_exam_history(), exam_output_path)
    
    memory_output_path = f"memory/{config.exam_topic}/agent_{personality_config.name}_lr{lr}.json"
    save_memory(memory_store, memory_output_path)  

    result_output_path = f"results/{config.exam_topic}/result_lr{lr}.csv"
    save_final_results(
        config, 
        personality_config.name,
        learning_summary,
        exam_summary,
        micro_macro_metrics,
        result_output_path
    )
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETED")
    print("="*80)
    
    return {
        'learning_summary': learning_summary,
        'exam_summary': exam_summary,
        'metrics': micro_macro_metrics
    }

if __name__ == "__main__":
    config = SimulationConfig(
        random_seed=42,
        exam_topic="Algebra",
        personality=PersonalityConfig.get_high_openness()
    )
    
    results = run_simulation(config, config.personality)