import abc
from typing import Dict, Union

# TODO: update Dict to TypedDict in python 3.8
Schedulable = Union[Dict, float]

def get_scheduler(scheduler_cfg):
    if isinstance(scheduler_cfg, dict):
        return scheduler_cfg["scheduler_fn"](**scheduler_cfg["scheduler_kwargs"])
    else:
        return ConstantScheduler(value=scheduler_cfg)

class Scheduler(object):
    """
    An abstract class for scheduler.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def value(self, step: int) -> float:
        """Return the value of the schedule for a given timestep

        Args:
            step: The timestep.

        Returns:
            the output value for the given timestep

        """
        raise NotImplementedError


class ConstantScheduler(Scheduler):
    """
    A constant scheduler.
    """

    def __init__(self, value):
        self._value = value

    def value(self, step):
        return self._value


class LinearScheduler(Scheduler):
    """
    A linear scheduler which linearly interpolates between initial_value and final_value
    over schedule_steps. After this many steps, final_value is returned.
    """

    def __init__(self, value, schedule_steps, final_value):
        self.schedule_steps = schedule_steps
        self.initial_value = value
        self.final_value = final_value

    def value(self, step):
        fraction = min(float(step) / self.schedule_steps, 1.0)
        return self.initial_value + fraction * (self.final_value - self.initial_value)
