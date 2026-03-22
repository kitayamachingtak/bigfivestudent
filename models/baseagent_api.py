import openai
import re
from typing import Dict, Optional

def create_api_client(model_config):
    client = openai.OpenAI(
        api_key=model_config.api_key,
        base_url=model_config.api_base_url
    )
    return client

class BaseAgentAPI:
    def __init__(self, api_client, model_config, temperature: float = 0.5):
        self.client = api_client
        self.model_config = model_config
        self.temperature = temperature
    
    def generate(self, prompt: str, max_new_tokens: int = None, max_retries: int = 1) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.model_config.max_new_tokens

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config.api_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=max_new_tokens
                )

                if response and response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content

                    if content is not None:
                        answer = content.strip()
                        if answer:  
                            return answer
                        else:
                            print(f"Warning: API returned empty string (attempt {attempt + 1}/{max_retries})")
                    else:
                        print(f"Warning: API returned None content (attempt {attempt + 1}/{max_retries})")
                else:
                    print(f"Warning: Invalid response structure (attempt {attempt + 1}/{max_retries})")

                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                else:
                    print(f"Error: Failed to get valid response after {max_retries} attempts")
                    return ""

            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg and attempt < max_retries - 1:
                    print(f"API Error (attempt {attempt + 1}/{max_retries}): 404, retrying...")
                    import time
                    time.sleep(1)
                    continue
                elif attempt == max_retries - 1:
                    print(f"API Error: Failed after {max_retries} attempts: {e}")
                    return ""
                else:
                    print(f"API Error: {e}")
                    return ""

        return ""
    
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