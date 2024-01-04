"""
Bayesian Neural Network model.
"""

import random
import time
import os
import torch
import torch.nn.functional as nnf
import pyro
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
import pyro.optim as pyroopt
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform
from pyro.nn import PyroModule

from examples.utils import plot_loss_accuracy, data_loaders, execution_time
from examples.nn import NN

softplus = torch.nn.Softplus()


class BNN(PyroModule):
    def __init__(self, dataset_name, hidden_size, activation, architecture,
                 epochs, lr, n_samples, warmup, input_shape, output_size, trained_by,
                 step_size=0.005, num_steps=10, sigma=0.2, scale_prior=1.):
        super(BNN, self).__init__()

        # self.scale_prior = scale_prior
        self.sigma = 0.1
        self.classification = False
        if dataset_name in ['mnist', 'half_moons', 'cifar', 'fashion_mnist']:
            self.classification = True

        self.dataset_name = dataset_name
        self.architecture = architecture
        self.trained_by = trained_by
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.warmup = warmup
        self.step_size = step_size
        self.num_steps = num_steps
        self.basenet = NN(dataset_name=dataset_name, input_shape=input_shape, output_size=output_size,
                          hidden_size=hidden_size, activation=activation, architecture=architecture,
                          epochs=epochs, lr=lr)
        self.name = self.get_name()

    def get_name(self, n_inputs=None):
        name = str(self.dataset_name) + "_bnn_hid=" + str(self.basenet.hidden_size) + "_act=" + str(
            self.basenet.activation) + "_arch=" + str(self.basenet.architecture)

        if n_inputs:
            name = name + "_inp=" + str(n_inputs)

        return name + "_" + str(self.trained_by)

    def model(self, x_data, y_data):
        priors = {}
        for key, value in self.basenet.state_dict().items():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value) * 0.01
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key): prior})

        lifted_module = pyro.random_module("module", self.basenet, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            if self.classification:
                lhat = nnf.log_softmax(logits, dim=-1)
                # lhat = nnf.softmax(logits, dim=-1)
                obs = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
            else:
                obs = pyro.sample("obs", Normal(logits, self.sigma), obs=y_data)

    def guide(self, x_data, y_data=None):
        dists = {}
        for key, value in self.basenet.state_dict().items():
            loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value))
            scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))

            if self.trained_by == 'oxford':
                distr = Normal(loc=loc, scale=scale)
            else:
                distr = Normal(loc=loc, scale=softplus(scale))

            dists.update({str(key): distr})

        lifted_module = pyro.random_module("module", self.basenet, dists)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            if self.classification:
                preds = nnf.softmax(logits, dim=-1)
            else:
                preds = logits

        return preds

    def save(self, path):
        filename = self.name + "_weights"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        print(f"\nSaving {path}{filename}")

        self.basenet.to("cpu")
        self.to("cpu")

        param_store = pyro.get_param_store()
        print("\nSaving: ", path + filename + ".pt")
        print(f"\nlearned params = {param_store.get_all_param_names()}")
        param_store.save("{}\{}{}".format(path, filename, ".pt"))

    def available(self, path):
        filename = self.name + "_weights"
        return os.path.exists("{}\{}{}".format(path, filename, ".pt"))

    def load(self, device, path):
        filename = self.name + "_weights"

        self.device = device
        self.basenet.device = device

        param_store = pyro.get_param_store()
        param_store.load("{}\{}{}".format(path, filename, ".pt"))
        for key, value in param_store.items():
            param_store.replace_param(key, value.to(device), value)

        print("\nLoading ", path + filename + ".pt\n")

        self.to(device)
        self.basenet.to(device)

    def forward(self, inputs, n_samples=10, avg_posterior=False, seeds=None, dist=False):
        if seeds:
            if len(seeds) != n_samples:
                raise ValueError("Number of seeds should match number of samples.")

        if avg_posterior is True:

            guide_trace = poutine.trace(self.guide).get_trace(inputs)

            avg_state_dict = {}
            for key in self.basenet.state_dict().keys():
                avg_weights = guide_trace.nodes[str(key) + "_loc"]['value']
                avg_state_dict.update({str(key): avg_weights})

            self.basenet.load_state_dict(avg_state_dict)
            preds = [self.basenet.model(inputs)]

        else:

            preds = []

            if seeds:
                for seed in seeds:
                    pyro.set_rng_seed(seed)
                    guide_trace = poutine.trace(self.guide).get_trace(inputs)
                    preds.append(guide_trace.nodes['_RETURN']['value'])

            else:

                for _ in range(n_samples):
                    guide_trace = poutine.trace(self.guide).get_trace(inputs)
                    preds.append(guide_trace.nodes['_RETURN']['value'])

        if dist:
            return torch.stack(preds)
        else:
            return torch.stack(preds).mean(0)

    def _train_svi(self, train_loader, epochs, lr, device, path):
        self.device = device

        print("\n == SVI training ==")

        optimizer = pyro.optim.Adam({"lr": lr})
        # elbo = TraceMeanField_ELBO()
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        loss_list = []
        accuracy_list = []

        start = time.time()
        for epoch in range(epochs):
            loss = 0.0
            correct_predictions = 0.0

            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if self.classification:
                    labels = y_batch.argmax(-1)
                    loss += svi.step(x_data=x_batch, y_data=labels)

                    outputs = self.forward(x_batch, n_samples=10)
                    predictions = outputs.argmax(dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                else:
                    # loss += svi.step(x_data=x_batch, y_data=y_batch)
                    loss += svi.step(x_data=x_batch, y_data=y_batch[:, 0])

            # total_loss = loss / len(train_loader.dataset)
            total_loss = loss
            # accuracy = 100 * correct_predictions / len(train_loader.dataset)
            accuracy = 100 * correct_predictions

            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}",
                  end="\t")

            loss_list.append(loss)
            accuracy_list.append(accuracy)

        execution_time(start=start, end=time.time())
        self.save(path=path)

    def train(self, train_loader, device, path, filename=None):
        self.device = device
        self.basenet.device = device

        self.to(device)
        self.basenet.to(device)

        random.seed(0)
        pyro.set_rng_seed(0)

        self._train_svi(train_loader, self.epochs, self.lr, device, path=path)

    def evaluate(self, test_loader, device, n_samples=10, seeds_list=None):
        self.device = device
        self.basenet.device = device
        self.to(device)
        self.basenet.to(device)

        random.seed(0)
        pyro.set_rng_seed(0)

        bnn_seeds = list(range(n_samples)) if seeds_list is None else seeds_list

        with torch.no_grad():
            correct_predictions = 0.0
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                outputs = self.forward(x_batch, n_samples=n_samples, seeds=bnn_seeds)
                predictions = outputs.argmax(-1)
                labels = y_batch.to(device).argmax(-1)
                correct_predictions += (predictions == labels).sum().item()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("Accuracy: %.2f%%" % (accuracy))
            return accuracy
