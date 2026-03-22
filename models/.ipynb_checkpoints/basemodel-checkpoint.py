import torch
import re
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer
from typing import Dict, Optional

def create_shared_generator(model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_config.model_path, 
        torch_dtype=model_config.torch_dtype
    )
    
    model.to(model_config.llm_device)
    
    generator = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer, 
        device=model_config.llm_device_id
    )
    
    return generator, tokenizer

class BaseAgent:
    def __init__(self, shared_generator, tokenizer, model_config, temperature: float = 0.7):
        self.generator = shared_generator
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.temperature = temperature
    
    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.model_config.max_new_tokens
        
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            do_sample=self.model_config.do_sample,
            top_p=self.model_config.top_p,
            repetition_penalty=self.model_config.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        full_text = result[0]['generated_text']
        answer = self._extract_answer(full_text)
        return answer
    
    def _extract_answer(self, full_text: str) -> str:
        marker = '[/INST]'
        if marker in full_text:
            start_idx = full_text.find(marker) + len(marker)
            answer = full_text[start_idx:].strip()
            
            stop_markers = ['<<SYS>>', '[/SYS]', '[INST]']
            for stop_marker in stop_markers:
                if stop_marker in answer:
                    idx = answer.find(stop_marker)
                    answer = answer[:idx].strip()
                    break
            
            return answer
        return full_text
    
    def parse_decision(self, response: str, valid_actions: list) -> str:
        response_lower = response.lower()
        matched_actions = [action for action in valid_actions if action in response_lower]
        
        if len(matched_actions) == 1:
            return matched_actions[0]
        else:
            return 'skip'
    
    def extract_content(self, response: str, field_name: str = 'content') -> Optional[str]:
        patterns = [
            rf'"{field_name}":\s*"([^"]*)"',
            rf'"{field_name}":\s*\'([^\']*)\'',
            rf'{field_name}:\s*"([^"]*)"',
            rf'{field_name}:\s*\'([^\']*)\'',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None