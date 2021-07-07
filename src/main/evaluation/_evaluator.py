from abc import ABC, abstractmethod


class _Evaluator(ABC):

    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
