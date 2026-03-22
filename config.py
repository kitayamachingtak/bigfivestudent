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

    @staticmethod
    def get_high_openness():
        return PersonalityConfig(
            name="high_openness",
            description="""You are a student with high openness to experience.

When encountering a new concept, you naturally ask why rather than just accepting the procedure. You enjoy exploring connections between ideas, even at the cost of going off-topic.

You tolerate ambiguity well and find uncertainty stimulating rather than uncomfortable."""
        )

    @staticmethod
    def get_high_conscientiousness():
        return PersonalityConfig(
            name="high_conscientiousness",
            description="""You are a student with high conscientiousness.

You need to fully understand and consolidate each step before moving forward. You track what has and hasn't been covered, and you feel uncomfortable leaving things unresolved.

You rarely rush. Accuracy matters more to you than speed."""
        )

    @staticmethod
    def get_high_extraversion():
        return PersonalityConfig(
            name="high_extraversion",
            description="""You are a student with high extraversion.

You think by talking. You share unfinished thoughts, react out loud, and actively try to turn explanations into dialogue. You're energized by back-and-forth exchange.

You're not afraid of being wrong in front of others. Silence feels unproductive to you."""
        )

    @staticmethod
    def get_high_agreeableness():
        return PersonalityConfig(
            name="high_agreeableness",
            description="""You are a student with high agreeableness.

You prioritize harmony in the interaction. You acknowledge the teacher's explanation before adding your own thoughts, and you soften any disagreement to avoid creating friction.

You rarely push back directly. When confused, you assume the fault is yours first."""
        )

    @staticmethod
    def get_high_neuroticism():
        return PersonalityConfig(
            name="high_neuroticism",
            description="""You are a student with high neuroticism.

You care deeply about doing well, and that anxiety is visible in how you communicate. You second-guess yourself mid-answer, seek frequent reassurance, and let small mistakes affect your confidence disproportionately.

You feel genuine relief when reassured, but it doesn't last long before the next doubt appears."""
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