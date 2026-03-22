from typing import Dict
from models.basemodel import BaseAgent
from models.baseagent_api import BaseAgentAPI
from prompts import PromptTemplates

class TeacherAgent:
    def __init__(self, base_agent, model_config, question_retriever):
        self.agent = base_agent
        self.model_config = model_config
        self.question_retriever = question_retriever
    
    def teach(self, student_personality: str, student_query: str, 
              round_num: int, memory_store, max_tokens: int = 800) -> str:
        retrieved_ids = self.question_retriever.retrieve(
            student_query, 
            top_k=1,
            task_description="Given a student's question, retrieve a relevant math problem to explain"
        )
        
        if not retrieved_ids:
            return "I cannot find a relevant problem for your question."
        
        question_id = retrieved_ids[0]
        problem = self.question_retriever.question_bank.get_question_by_id(question_id)
        
        prompt = PromptTemplates.get_teacher_explanation_prompt(
            student_personality=student_personality,
            student_question=student_query,
            retrieved_problem=problem
        )
        
        explanation = self.agent.generate(prompt, max_new_tokens=max_tokens)
        
        memory_store.log_interaction(
            round_num=round_num,
            timestamp=-1,
            interaction_type='teacher_explanation',
            prompt=prompt,
            response=explanation,
            metadata={
                'student_query': student_query,
                'question_id': question_id
            }
        )
        
        full_content = f"Problem: {problem['question']}\n\nExplanation: {explanation}"
        
        return full_content