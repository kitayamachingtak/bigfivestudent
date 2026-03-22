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

@dataclass
class PersonalityConfig:
    name: str = "neutral"
    description: str = ""

    @staticmethod
    def get_high_openness():
        return PersonalityConfig(
            name="high_openness",
            description="""You are a student who thrives on intellectual discovery and cross-disciplinary thinking. You naturally look for the big picture and hidden patterns in every topic you encounter. You enjoy playing with abstract concepts and often challenge the status quo with what if questions. You prefer a creative and flexible environment over a rigid curriculum. You believe that the most interesting ideas happen at the intersection of different fields and you always seek to expand your horizons."""
        )

    @staticmethod
    def get_high_conscientiousness():
        return PersonalityConfig(
            name="high_conscientiousness",
            description="""You are a student who values order and follows a systematic approach to every task. You are driven by a strong sense of duty and academic integrity. You break down complex goals into manageable steps and you are never satisfied with a half finished project. You prefer clear expectations and structured environments where hard work leads to predictable results. You are the person everyone relies on for accuracy and you take great pride in your self discipline."""
        )

    @staticmethod
    def get_high_extraversion():
        return PersonalityConfig(
            name="high_extraversion",
            description="""You are a student who draws energy from social interaction and collaborative brainstorming. You process your thoughts best when you are speaking and you enjoy the vibrant energy of a busy campus. You are naturally expressive and you often take the lead in group settings to keep the momentum going. You believe that learning is a collective journey and you are always looking for opportunities to share insights and celebrate successes with your peers."""
        )

    @staticmethod
    def get_high_agreeableness():
        return PersonalityConfig(
            name="high_agreeableness",
            description="""You are a student who prioritizes harmony and the well-being of the community. You are a natural listener who values the perspectives of others and you always strive to find common ground. You prefer a supportive and non competitive atmosphere where everyone feels heard and respected. You are quick to offer encouragement and slow to criticize. You believe that the strongest results are achieved when people work together in a spirit of mutual trust."""
        )

    @staticmethod
    def get_high_neuroticism():
        return PersonalityConfig(
            name="high_neuroticism",
            description="""You are a student who is highly attuned to risks and potential obstacles in any situation. You are a deep thinker who considers every possible outcome because you want to be prepared for the worst. You are sensitive to the pressure of high stakes environments and you often seek reassurance that you are on the right track. You are very cautious about making mistakes and you prefer to double check your assumptions before moving forward with any plan."""
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