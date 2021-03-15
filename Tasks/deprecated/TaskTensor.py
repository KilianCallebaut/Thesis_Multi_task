from torch import Tensor


class TaskTensor(Tensor):

    @staticmethod
    def __new__(cls, data, task_name, *args, **kwargs):
        return super().__new__(cls, data, *args, **kwargs)

    def __init__(self, data, task_name):
        self.task_name = task_name

    def clone(self, *args, **kwargs):
        return TaskTensor(super().clone(*args, **kwargs), self.task_name)

    def to(self, *args, **kwargs):
        new_obj = TaskTensor([], self.task_name)
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return(new_obj)