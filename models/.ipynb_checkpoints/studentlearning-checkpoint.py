from typing import Dict, List
import re
from models.basemodel import BaseAgent
from prompts import PromptTemplates
from storememory import MemoryStore
from timestamp import TimestampManager

class StudentLearningAgent(BaseAgent):
    def __init__(self, shared_generator, tokenizer, model_config, personality, 
                 question_retriever, memory_store, timestamp_manager):
        super().__init__(shared_generator, tokenizer, model_config, 
                        temperature=model_config.student_temperature)
        self.personality = personality
        self.question_retriever = question_retriever
        self.memory_store = memory_store
        self.timestamp_manager = timestamp_manager
        
        self.learning_history = []
        self.previous_action = None
    
    def learning_round(self, round_num: int, total_rounds: int, topic: str, 
                  teacher_agent=None) -> Dict:
        context = {
            'current_round': round_num,
            'total_rounds': total_rounds,
            'topic': topic,
            'memory_count': len(self.memory_store.memories),
            'previous_action': self.previous_action
        }

        decision = self._make_decision(context, round_num)  # 添加 round_num 参数
        self.timestamp_manager.consume('decision', f"Round {round_num} decision")

        
        action = decision['action']
        reasoning = decision['reasoning']
        
        if action == 'self_study':
            result = self._execute_self_study(context, round_num)
        elif action == 'ask_teacher':
            result = self._execute_ask_teacher(context, round_num, teacher_agent)
        else:
            result = self._execute_skip(round_num)
        
        self.previous_action = {
            'action': action,
            'content': result.get('content', ''),
            'reasoning': reasoning
        }
        
        self.learning_history.append({
            'round': round_num,
            'action': action,
            'reasoning': reasoning,
            'timestamp': self.timestamp_manager.get_total_time(),
            'result': result
        })
        
        return {
            'action': action,
            'reasoning': reasoning,
            'timestamp': self.timestamp_manager.get_total_time(),
            'result': result
        }
    
    def _make_decision(self, context: Dict, round_num: int) -> Dict:
        prompt = PromptTemplates.get_learning_decision_prompt(
            self.personality.description, context
        )
        response = self.generate(prompt)

        # Log this interaction
        self.memory_store.log_interaction(
            round_num=round_num,
            timestamp=self.timestamp_manager.get_total_time(),
            interaction_type='learning_decision',
            prompt=prompt,
            response=response,
            metadata={'context': context}
        )

        # Better action parsing
        response_lower = response.lower()
        if 'self_study' in response_lower or 'self study' in response_lower or 'review' in response_lower or 'study' in response_lower:
            action = 'self_study'
        elif 'ask_teacher' in response_lower or 'ask teacher' in response_lower or 'teacher' in response_lower:
            action = 'ask_teacher'
        elif 'skip' in response_lower:
            action = 'skip'
        else:
            action = 'skip'  # default

        reasoning_patterns = [
            r'"reasoning":\s*"([^"]*)"',
            r'"reasoning":\s*\'([^\']*)\'',
            r'reasoning:\s*"([^"]*)"',
            r'because\s+(.+?)(?:\.|$)',
            r'reason:\s*(.+?)(?:\.|$)'
        ]

        reasoning = ""
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                break

        if not reasoning:
            reasoning = response[:100]

        return {'action': action, 'reasoning': reasoning}
    
    def _execute_self_study(self, context: Dict, round_num: int) -> Dict:
        prompt = PromptTemplates.get_self_study_content_prompt(
            self.personality.description, context
        )
        response = self.generate(prompt)

        # Log this interaction
        self.memory_store.log_interaction(
            round_num=round_num,
            timestamp=self.timestamp_manager.get_total_time(),
            interaction_type='self_study_content_decision',
            prompt=prompt,
            response=response,
            metadata={'context': context}
        )

        self.timestamp_manager.consume('self_study_content', f"Round {round_num} decide content")

        content = self.extract_content(response, 'content')
        if not content:
            content = response[:50]

        retrieved_ids = self.question_retriever.retrieve(
            content,
            top_k=1,
            task_description="Given a student's study topic, retrieve a relevant practice problem"
        )

        if not retrieved_ids:
            return {'content': content, 'learned': ''}

        question_id = retrieved_ids[0]
        problem = self.question_retriever.question_bank.get_question_by_id(question_id)

        learned_content = f"Problem: {problem['question']}\n\nSolution: {problem['solution']}"

        self.memory_store.add_memory(
            content=learned_content,
            source='self_study',
            timestamp=self.timestamp_manager.get_total_time(),
            round_num=round_num,
            decision_info={'query': content}
        )

        return {
            'content': content,
            'learned': learned_content,
            'question_id': question_id
        }
    
    def _execute_ask_teacher(self, context: Dict, round_num: int, teacher_agent) -> Dict:
        if teacher_agent is None:
            return {'content': '', 'learned': 'No teacher available'}

        prompt = PromptTemplates.get_ask_teacher_content_prompt(
            self.personality.description, context
        )
        response = self.generate(prompt)

        # Log student's question formulation
        self.memory_store.log_interaction(
            round_num=round_num,
            timestamp=self.timestamp_manager.get_total_time(),
            interaction_type='ask_teacher_question_formulation',
            prompt=prompt,
            response=response,
            metadata={'context': context}
        )

        self.timestamp_manager.consume('ask_teacher_content', f"Round {round_num} decide question")

        question = self.extract_content(response, 'content')
        if not question:
            question = response[:50]

        # Teacher generates explanation (teacher will log its own interaction internally)
        explanation = teacher_agent.teach(
            student_personality=self.personality.description,
            student_query=question,
            round_num=round_num,
            memory_store=self.memory_store
        )
        self.timestamp_manager.consume('teacher_response', f"Round {round_num} teacher explanation")

        self.memory_store.add_memory(
            content=explanation,
            source='ask_teacher',
            timestamp=self.timestamp_manager.get_total_time(),
            round_num=round_num,
            decision_info={'question': question}
        )

        return {
            'content': question,
            'learned': explanation
        }
    
    def _execute_skip(self, round_num: int) -> Dict:
        self.memory_store.add_memory(
            content='',
            source='skip',
            timestamp=self.timestamp_manager.get_total_time(),
            round_num=round_num,
            decision_info={}
        )
        
        return {'content': '', 'learned': ''}
    
    def get_learning_history(self) -> list:
        return self.learning_history