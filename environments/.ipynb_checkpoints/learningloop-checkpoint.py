from typing import Dict, List

class LearningLoop:
    def __init__(self, student_learning_agent, teacher_agent, config):
        self.student = student_learning_agent
        self.teacher = teacher_agent
        self.config = config
        self.learning_rounds = config.learning_config.learning_rounds
        self.topic = config.exam_topic
    
    def run(self) -> List[Dict]:
        results = []
        
        for round_num in range(1, self.learning_rounds + 1):
            round_result = self.student.learning_round(
                round_num=round_num,
                total_rounds=self.learning_rounds,
                topic=self.topic,
                teacher_agent=self.teacher
            )
            
            results.append(round_result)
            
            print(f"Round {round_num}/{self.learning_rounds} - Action: {round_result['action']}, "
                  f"Timestamp: {round_result['timestamp']}")
        
        return results
    
    def get_summary(self) -> Dict:
        history = self.student.get_learning_history()
        
        action_counts = {
            'self_study': 0,
            'ask_teacher': 0,
            'skip': 0
        }
        
        for record in history:
            action = record['action']
            if action in action_counts:
                action_counts[action] += 1
        
        return {
            'total_rounds': len(history),
            'action_counts': action_counts,
            'final_timestamp': self.student.timestamp_manager.get_total_time(),
            'total_memories': len(self.student.memory_store.memories),
            'learning_history': history
        }