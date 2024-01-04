"""
Deterministic Neural Network model.
"""

from torch import nn

class NN(nn.Module):

    def __init__(self, dataset_name, input_shape, output_size, hidden_size, activation,
                 architecture, lr, epochs):
        super(NN, self).__init__()
        self.dataset_name = dataset_name
        self.loss_func = nn.CrossEntropyLoss()
        self.architecture = architecture
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.lr, self.epochs = lr, epochs

        self.name = self.get_name(dataset_name, hidden_size, activation, architecture, lr, epochs)
        self.set_model(architecture, activation, input_shape, output_size, hidden_size)

    def get_name(self, dataset_name, hidden_size, activation, architecture, lr, epochs):
        return str(dataset_name)+"_nn_hid="+str(hidden_size)+"_act="+str(activation)+\
               "_arch="+str(architecture)+"_ep="+str(epochs)+"_lr="+str(lr)

    def set_model(self, architecture, activation, input_shape, output_size, hidden_size):
        if type(input_shape) is not int:
            input_size = input_shape[0]*input_shape[1]*input_shape[2]
        else:
            input_size = input_shape

        if activation == "relu":
            activ = nn.ReLU
        elif activation == "leaky":
            activ = nn.LeakyReLU
        elif activation == "sigm":
            activ = nn.Sigmoid
        elif activation == "tanh":
            activ = nn.Tanh
        else:
            raise AssertionError("\nWrong activation name.")

        if architecture == "fc1":
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, output_size))

        elif architecture == "fc2":
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, output_size))
        else:
            raise NotImplementedError()

    def forward(self, inputs, device=None, *args, **kwargs):
        device = self.device if device is None else device

        self.to(device)
        inputs = inputs.to(device)

        x = self.model(inputs)

        return x