from models.basemodel import BaseAgent
from prompts import PromptTemplates

class TeacherAgent(BaseAgent):
    def __init__(self, model_config, question_retriever):
        super().__init__(model_config, temperature=model_config.teacher_temperature)
        self.question_retriever = question_retriever
    
    def teach(self, student_personality: str, student_query: str, 
              max_tokens: int = 800) -> str:
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
        
        explanation = self.generate(prompt, max_new_tokens=max_tokens)
        
        full_content = f"Problem: {problem['question']}\n\nExplanation: {explanation}"
        
        return full_content