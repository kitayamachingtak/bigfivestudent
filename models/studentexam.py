from typing import Dict, List
import re
from prompts import PromptTemplates
from storememory import MemoryStore
from timestamp import TimestampManager
from evaluation.metrics import AnswerEvaluator

class StudentExamAgent:
    def __init__(self, base_agent, model_config, personality, 
                 memory_store, timestamp_manager):
        self.agent = base_agent
        self.model_config = model_config
        self.personality = personality
        self.memory_store = memory_store
        self.timestamp_manager = timestamp_manager
        
        self.exam_history = []
        self.previous_exam_accuracy = None
    
    def answer_question(self, question: str, question_id: int, topic: str, 
                   extracted_answer: str, raw_answer: str) -> Dict:

        recall_result = self._recall_memory(question, question_id)
        query = recall_result['query']
        recalled_memory = recall_result['recalled_memory']

        answer_result = self._answer_with_memory(question, question_id, recalled_memory)
        answer = answer_result['answer']

        is_correct_simple = self._check_answer(answer, extracted_answer, raw_answer)

        metrics = AnswerEvaluator.evaluate_answer(answer, extracted_answer, raw_answer)

        is_correct_final = is_correct_simple or metrics['exact_match']

        self.exam_history.append({
            'question_id': question_id,
            'question': question,
            'recall_query': query,
            'recalled_memory': recalled_memory,
            'answer': answer,
            'correct_simple': is_correct_simple,    
            'exact_match': metrics['exact_match'],  
            'correct_final': is_correct_final,      
            'extracted_answer': extracted_answer,
            'raw_answer': raw_answer,
            'metrics': metrics,
            'timestamp': self.timestamp_manager.get_total_time()
        })

        return {
            'question_id': question_id,
            'answer': answer,
            'correct': is_correct_final, 
            'metrics': metrics,
            'timestamp': self.timestamp_manager.get_total_time()
        }
    def _recall_memory(self, question: str, question_id: int) -> Dict:
        prompt = PromptTemplates.get_exam_memory_query_prompt(
            self.personality.description, question
        )
        response = self.agent.generate(prompt)
        
        self.memory_store.log_interaction(
            round_num=question_id,
            timestamp=self.timestamp_manager.get_total_time(),
            interaction_type='exam_memory_query',
            prompt=prompt,
            response=response,
            metadata={'question': question}
        )
        
        self.timestamp_manager.consume('exam_recall_query', f"Question {question_id} recall query")
        
        query = self.agent.extract_content(response, 'query')
        if not query:
            query = question[:100]
        
        retrieved_memories = self.memory_store.retrieve(
            query,
            top_k=3,
            threshold=0.0
        )
        
        recalled_memory = "\n\n".join(retrieved_memories) if retrieved_memories else "No relevant memory found."
        
        return {
            'query': query,
            'recalled_memory': recalled_memory
        }
    
    def _answer_with_memory(self, question: str, question_id: int, recalled_memory: str) -> Dict:
        prompt = PromptTemplates.get_exam_recall_and_answer_prompt(
            self.personality.description,
            question,
            recalled_memory
        )
        response = self.agent.generate(prompt)
        
        self.memory_store.log_interaction(
            round_num=question_id,
            timestamp=self.timestamp_manager.get_total_time(),
            interaction_type='exam_answer_with_memory',
            prompt=prompt,
            response=response,
            metadata={
                'question': question,
                'recalled_memory': recalled_memory
            }
        )
        
        self.timestamp_manager.consume('exam_answer', f"Question {question_id} answer")
        
        answer = self.agent.extract_content(response, 'answer')
        if not answer:
            answer = response
        
        return {'answer': answer}
    
    def _check_answer(self, student_answer: str, extracted_answer: str, raw_answer: str) -> bool:
        if not student_answer or student_answer.strip() == "":
            return False

        student_lower = student_answer.lower().strip()
        extracted_lower = str(extracted_answer).lower().strip()
        raw_lower = str(raw_answer).lower().strip()

        if extracted_lower in student_lower or student_lower in extracted_lower:
            return True
        if raw_lower in student_lower or student_lower in raw_lower:
            return True

        return False

    def calculate_accuracy(self) -> float:
        if not self.exam_history:
            return 0.0

        correct_count = sum(1 for item in self.exam_history if item['correct_final'])
        accuracy = correct_count / len(self.exam_history)
        self.previous_exam_accuracy = accuracy
        return accuracy

    def get_empty_answer_stats(self) -> dict:
        if not self.exam_history:
            return {
                'total_questions': 0,
                'empty_count': 0,
                'empty_percentage': 0.0
            }

        total = len(self.exam_history)
        empty_count = sum(1 for item in self.exam_history 
                         if not item['answer'] or item['answer'].strip() == "")

        return {
            'total_questions': total,
            'empty_count': empty_count,
            'empty_percentage': (empty_count / total * 100) if total > 0 else 0.0
        }
    
    def get_exam_history(self) -> List[Dict]:
        return self.exam_history
    
    def reset_exam(self):
        self.exam_history = []