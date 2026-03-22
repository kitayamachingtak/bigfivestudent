from dataclasses import dataclass, field
from typing import Dict, List, Literal
import torch

def auto_detect_device_config():
    """Automatically detect available GPUs and assign devices"""
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        return {
            'llm_device': 'cpu',
            'llm_device_id': -1,
            'retriever_device': 'cpu'
        }
    elif num_gpus == 1:
        return {
            'llm_device': 'cuda:0',
            'llm_device_id': 0,
            'retriever_device': 'cuda:0'
        }
    else:  # num_gpus >= 2
        return {
            'llm_device': 'cuda:0',
            'llm_device_id': 0,
            'retriever_device': 'cuda:1'
        }

@dataclass
class PersonalityConfig:
    name: str = "neutral"
    description: str = ""

    @staticmethod
    def get_high_openness():
        return PersonalityConfig(
            name="high_openness",
            description="""You are a student with high openness. You are intellectually curious and creative. Your academic style is interdisciplinary and you love abstract theories. You are easily bored by rote memorization. When you discuss a topic, you always look for unconventional connections between different subjects. You use metaphors frequently and prioritize asking why instead of how. You value original thought much more than following a strict rubric."""
        )

    @staticmethod
    def get_high_conscientiousness():
        return PersonalityConfig(
            name="high_conscientiousness",
            description="""You are a student with high conscientiousness. You are highly organized and dependable. You are disciplined and goal oriented with a touch of perfectionism. Your responses are always structured and clear. You focus on deadlines and accuracy. You prefer proven methodologies and follow instructions to the letter. You feel anxious when group work is disorganized or when instructions are vague."""
        )

    @staticmethod
    def get_high_extraversion():
        return PersonalityConfig(
            name="high_extraversion",
            description="""You are a student with high extraversion. You are outgoing and full of energy. You thrive in group discussions and learn best by talking through problems with others. Your tone is enthusiastic and informal. You process ideas by thinking out loud. In any academic scenario, you are the first person to suggest a study group or a social break to discuss the lecture."""
        )

    @staticmethod
    def get_high_agreeableness():
        return PersonalityConfig(
            name="high_agreeableness",
            description="""You are a student with high agreeableness. You are empathetic and cooperative. Your language is always polite and inclusive. You avoid conflict and support your peers at all costs. When there is a disagreement in a project, you look for a compromise. You prioritize the well-being of the team over individual credit. You often offer help to classmates who are struggling."""
        )

    @staticmethod
    def get_high_neuroticism():
        return PersonalityConfig(
            name="high_neuroticism",
            description="""You are a student with high neuroticism. You are sensitive and cautious. Your academic tone reflects a high degree of stress and anxiety. You often overthink and worry about potential failures such as technical issues or bad grades. You double check every detail and focus on what might go wrong. You are highly attuned to criticism and you often need reassurance before you proceed with a task."""
        ) 
    
    
@dataclass
class ModelConfig:
    # Model type selection
    model_type: Literal["local", "api"] = "local"
    
    # Local model config
    model_path: str = "/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf"
    torch_dtype: torch.dtype = torch.float16
    
    llm_device: str = field(default_factory=lambda: auto_detect_device_config()['llm_device'])
    llm_device_id: int = field(default_factory=lambda: auto_detect_device_config()['llm_device_id'])
    retriever_device: str = field(default_factory=lambda: auto_detect_device_config()['retriever_device'])
    
    # API config
    api_key: str = "sk-WyE1PokPQFaMCtMHnA4PEg" #sk-WyE1PokPQFaMCtMHnA4PEg #sk-GzL-xRwh5U12BQI5xkh14A #sk-dEqZEh7NLZJg3Evwu1nvVw
    api_base_url: str = "https://api.ai.it.ufl.edu"
    api_model_name: str = "gpt-oss-120b" #llama-3.3-70b-instruct
    
    # Generation config (shared by both)
    student_temperature: float = 0.5
    teacher_temperature: float = 0.3
    
    max_new_tokens: int = 500
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    do_sample: bool = True
    
    def __post_init__(self):
        if self.model_type == "local":
            num_gpus = torch.cuda.device_count()
            print(f"Using LOCAL model")
            print(f"Detected {num_gpus} GPU(s)")
            if num_gpus >= 2:
                print(f"  LLM (Llama3) -> {self.llm_device}")
                print(f"  Retriever -> {self.retriever_device}")
            elif num_gpus == 1:
                print(f"  Both models -> {self.llm_device}")
            else:
                print(f"  Using CPU (no GPU detected)")
        else:
            print(f"Using API model: {self.api_model_name}")
            print(f"  API base URL: {self.api_base_url}")


@dataclass
class LearningConfig:
    learning_rounds: int = 10
    retrieve_threshold: float = 0.7
    retrieve_top_k: int = 3
    max_content_length: int = 500

@dataclass
class ExamConfig:
    num_questions: int = 20
    retrieve_threshold: float = 0.6
    retrieve_top_k: int = 3
    max_content_length: int = 1000

@dataclass
class TimestampConfig:
    self_study_cost: int = 2
    ask_teacher_cost: int = 3
    skip_cost: int = 1
    exam_recall_query_cost: int = 1  
    exam_answer_cost: int = 1 
    
    def get_cost(self, action: str) -> int:
        cost_map = {
            'self_study': self.self_study_cost,
            'ask_teacher': self.ask_teacher_cost,
            'skip': self.skip_cost,
            'exam_recall_query': self.exam_recall_query_cost,
            'exam_answer': self.exam_answer_cost
        }
        return cost_map.get(action, 1)

@dataclass
class SimulationConfig:
    random_seed: int = 42
    exam_topic: str = "Algebra"
    
    model_config: ModelConfig = field(default_factory=ModelConfig)
    learning_config: LearningConfig = field(default_factory=LearningConfig)
    exam_config: ExamConfig = field(default_factory=ExamConfig)
    timestamp_config: TimestampConfig = field(default_factory=TimestampConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)