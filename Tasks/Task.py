class Task:

    def __init__(
            self,
            name: str,
            output_labels: list,
            output_module='softmax' #'sigmoid'

    ):
        super().__init__()
        self.output_labels = output_labels
        self.name = name
        self.output_module = output_module
