import os
import json
from typing import Dict
from pydantic import BaseModel
from mikazuki.models import TaggerInterrogateRequest, TrainingInfo


class TrainingInfoManager:
    def __init__(self, storage_dir='./trainInfo'):
        self.storage_dir = storage_dir
        self.training_info_list: Dict[str, TrainingInfo] = {} 
        self.load_training_info()

    def load_training_info(self):
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                task_id = filename[:-5]
                with open(os.path.join(self.storage_dir, filename), 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    self.training_info_list[task_id] = TrainingInfo.parse_obj(data)

    def save_training_info(self):
        for task_id, info in self.training_info_list.items():
            with open(os.path.join(self.storage_dir, f'{task_id}.json'), 'w', encoding="utf-8") as f:
                json.dump(info.dict(), f, indent=4)


train_info_manager = TrainingInfoManager()