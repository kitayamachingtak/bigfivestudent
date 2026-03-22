from models.basemodel import create_shared_generator, BaseAgent
from models.baseagent_api import create_api_client, BaseAgentAPI

def create_base_agent(model_config, temperature):
    if model_config.model_type == "local":
        shared_generator, tokenizer = create_shared_generator(model_config)
        return BaseAgent(shared_generator, tokenizer, model_config, temperature), shared_generator, tokenizer
    else:  # API mode
        api_client = create_api_client(model_config)
        return BaseAgentAPI(api_client, model_config, temperature), api_client, None

def get_agent_init_params(model_config, base_agent_instance, shared_resource):
    if model_config.model_type == "local":
        shared_generator, tokenizer = shared_resource
        return {
            'shared_generator': shared_generator,
            'tokenizer': tokenizer
        }
    else:
        api_client = shared_resource
        return {
            'api_client': api_client
        }