from Training.Results import Results


class TrainingUtils:

    def combine_loss(self, list_of_losses):
        return sum(list_of_losses)

    def early_stop(self, results: Results, epoch: int):
        if epoch == 0:
            return False
        if (results.training_curve[epoch - 1] - results.training_curve[epoch]) < 0.1:
            return True

        return False

    def extra_operation(self, **kwargs):
        pass
