from typing import Dict, List

class ExamLoop:
    def __init__(self, student_exam_agent, question_bank, test_question_ids, config):
        self.student = student_exam_agent
        self.question_bank = question_bank
        self.test_question_ids = test_question_ids
        self.config = config
        self.num_questions = config.exam_config.num_questions
        self.topic = config.exam_topic
    
    def run(self) -> List[Dict]:
        exam_question_ids = self.test_question_ids[:self.num_questions]
        results = []
        
        for idx, question_id in enumerate(exam_question_ids, 1):
            problem = self.question_bank.get_question_by_id(question_id)
            
            result = self.student.answer_question(
                question=problem['question'],
                question_id=question_id,
                topic=self.topic,
                extracted_answer=problem['extracted_answer'],
                raw_answer=problem['raw_answer']
            )
            
            results.append(result)
            
            print(f"Question {idx}/{len(exam_question_ids)} - "
                  f"Correct: {result['correct']}, Timestamp: {result['timestamp']}")
        
        return results
    
    def get_summary(self) -> Dict:
        history = self.student.get_exam_history()
        accuracy = self.student.calculate_accuracy()
        empty_stats = self.student.get_empty_answer_stats()

        correct_count = sum(1 for record in history if record['correct_final'])

        correct_simple_count = sum(1 for record in history if record['correct_simple'])
        exact_match_count = sum(1 for record in history if record['exact_match'])

        return {
            'total_questions': len(history),
            'correct_count': correct_count,             
            'correct_simple_count': correct_simple_count,  
            'exact_match_count': exact_match_count,    
            'accuracy': accuracy,                     
            'empty_answer_count': empty_stats['empty_count'],
            'empty_answer_percentage': empty_stats['empty_percentage'],
            'final_timestamp': self.student.timestamp_manager.get_total_time(),
            'exam_history': history
        }