from src.core.service_base import BaseService
from src.artifacts.l1.componentA import ComponentAArtifact
from src.artifacts.l1.componentB import ComponentBArtifact
import dataclasses

@dataclasses.dataclass
class L1Artifacts:
    componentA: ComponentAArtifact
    componentB: ComponentBArtifact

    @classmethod
    def init_new(cls):
        return cls(
            componentA = ComponentAArtifact(), 
            componentB = ComponentBArtifact()
        )
    
    def register(self, service):
        self.componentA.register(service)
        self.componentB.register(service)

class L1Service(BaseService):
    def __init__(self, ):
        super().__init__()
        self.artifacts = L1Artifacts.init_new()
        self.artifacts.register(self)
    
    @property
    def path(self):
        return __file__
    
    def run(self):
        keys = ["A", "B", "C"]
        values = [12, 33, 142]
        self.create_with_initial_values(keys, values)
        self.dict_A["A"] = 7
        self.print_dict()