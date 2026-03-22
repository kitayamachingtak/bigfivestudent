from dataclasses import dataclass, field
from typing import Dict, List, Literal
import torch

def auto_detect_device_config():
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
            description="""You are a student with high openness. You are curious about new knowledge, enjoy exploring different problem-solving methods, and prefer understanding concepts deeply rather than memorizing procedures."""
        )
    
    @staticmethod
    def get_high_conscientiousness():
        return PersonalityConfig(
            name="high_conscientiousness",
            description="""You are a highly conscientious student. You plan study tasks carefully, take homework seriously, and persist in mastering difficult problems even when tired. You regularly review notes to ensure knowledge consolidation."""
        )
    
    @staticmethod
    def get_high_extraversion():
        return PersonalityConfig(
            name="high_extraversion",
            description="""You are a highly extraverted student. You enjoy communicating with teachers, prefer learning through discussion rather than studying alone, and feel comfortable actively asking questions."""
        )
    
    @staticmethod
    def get_high_agreeableness():
        return PersonalityConfig(
            name="high_agreeableness",
            description="""You are a highly agreeable student. You are cooperative and willing to accept teachers' suggestions. You prefer harmonious learning environments and are receptive to feedback."""
        )
    
    @staticmethod
    def get_high_neuroticism():
        return PersonalityConfig(
            name="high_neuroticism",
            description="""You are a student with high neuroticism. You feel anxious about academic performance, doubt your abilities, and small setbacks affect your confidence. You tend to seek reassurance from teachers when uncertain."""
        )    
    
@dataclass
class ModelConfig:
    model_type: Literal["local", "api"] = "local"
    
    model_path: str = "your_local_model_path_here"
    torch_dtype: torch.dtype = torch.float16
    
    llm_device: str = field(default_factory=lambda: auto_detect_device_config()['llm_device'])
    llm_device_id: int = field(default_factory=lambda: auto_detect_device_config()['llm_device_id'])
    retriever_device: str = field(default_factory=lambda: auto_detect_device_config()['retriever_device'])
    
    api_key: str = "your_api_key_here" 
    api_base_url: str = "your_api_base_url_here"
    api_model_name: str = "gpt-oss-120b" 
    
    student_temperature: float = 0.7
    teacher_temperature: float = 0.5
    
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
    retrieve_top_k: int = 1
    max_content_length: int = 500

@dataclass
class ExamConfig:
    num_questions: int = 20
    retrieve_threshold: float = 0.6
    retrieve_top_k: int = 2
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