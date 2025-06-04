from abc import abstractmethod


class BaseController:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def step() -> None:
        raise NotImplementedError("Method 'step' must be implemented in controller.")
