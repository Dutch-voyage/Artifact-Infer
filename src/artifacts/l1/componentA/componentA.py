from src.core.artifact_base import Artifact

class ComponentAArtifact(Artifact):
    def __init__(self):
        super().__init__()
        self.dict_A = {}
    
    @property
    def path(self):
        return __file__
    
    @property
    def name(self):
        return "ComponentAArtifact"
    
    def register(self, service):
        self._register_obj("dict_A", service)
    
    def create_dict_A_with_initial_value(self, keys, values):
        for key, value in zip(keys, values):
            self.append(key, value)
        