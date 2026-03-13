from src.core.artifact_base import Artifact

class ComponentBArtifact(Artifact):
    def __init__(self):
        super().__init__()
    
    @property
    def path(self):
        return __file__
    
    @property
    def name(self):
        return "ComponentBAritfact"
    
    def register(self, service):
        methods_to_register = ["append", "print_dict"]
        for method in methods_to_register:
            self._register_method(method, service)
    
    def append(self, key, value):
        self.dict_A[key] = value
    
    def print_dict(self):
        print(self.dict_A)