from typing import List
import numpy as np
from ray import serve
from ray.serve.handle import DeploymentHandle
import yaml
from ray import serve
from google.protobuf.json_format import MessageToDict
import sys
import uuid
from mii_custom import pipeline

def random_uuid():
    return str(uuid.uuid4())

CONFIG_FILE = "ray_config.yaml"
with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)
print(CONFIG)

@serve.deployment(ray_actor_options={"num_gpus": 1})
class Model:
    def __init__(self):
        self.pipeline = pipeline(CONFIG['MODEL_NAME'])
    @serve.batch(max_batch_size=CONFIG['MAX_BATCH_SIZE'], batch_wait_timeout_s=CONFIG['BATCH_WAIT_TIMEOUT_S'])
    async def __call__(self, requests: List[dict]) -> List[dict]:
        requests = [await request.json() for request in requests]
        print("Batch size: ", len(requests))
        prompts = [request['prompt'] for request in requests]
        params = [{
            'max_length': request['max_tokens'],
            'postprocess_config': {
                    'logit_processor': {
                    'name': 'Temperature',
                    'args': {
                        'temperature': request['temperature']
                    }
                }
            }
        } for request in requests]
        outputs = self.pipeline(inputs=prompts, params=params)
        print(outputs)
        responses = [output.get_msg() for output in outputs]
        return responses

app = Model.bind()
