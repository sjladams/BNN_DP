import torch
import copy
from pyro.distributions import Normal, Categorical
from torch import nn, distributions
import torch.nn.functional as F

from pyro.nn import PyroModule, PyroSample
from pyro.infer import Predictive
import pyro.distributions as dist
import pyro
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO
from bound_propagation import HyperRectangle, BoundModelFactory
import torchattacks
import matplotlib.pyplot as plt

from bound import bound_softmax

factory = BoundModelFactory()


class PyroBatchLinear(nn.Linear, PyroModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None,
                 weight_prior=None, bias_prior=None, weight_prior_scale=1., bias_prior_scale=1.) -> None:
        # While calling super().__init__() creates the weights and we overwrite them later, it's just easier this way,
        # and we get to inherit from nn.Linear. It shouldn't be too much of an issue considering that you usually
        # only instantiate a model once during execution.
        super(PyroBatchLinear, self).__init__(in_features, out_features, bias, device, dtype)

        if weight_prior is None:
            weight_prior = Normal(torch.as_tensor(0.0, device=device),
                                  torch.as_tensor(weight_prior_scale, device=device)) \
                .expand(self.weight.shape) \
                .to_event(self.weight.dim())

        if bias and bias_prior is None:
            bias_prior = Normal(torch.as_tensor(0.0, device=device),
                                torch.as_tensor(bias_prior_scale, device=device)).expand(self.bias.shape).to_event(
                self.bias.dim())

        self.weight_prior = weight_prior
        self.weight = PyroSample(prior=self.weight_prior)
        if bias:
            self.bias_prior = bias_prior
            self.bias = PyroSample(prior=self.bias_prior)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        weight = self.weight

        if weight.dim() == 2:
            return F.linear(input, weight, bias)
        else:
            if input.dim() == 2:
                input = input.unsqueeze(0).expand(weight.size(0), *input.size())

            return torch.baddbmm(bias.unsqueeze(1), input, weight.transpose(-1, -2))

    def to(self, *args, **kwargs):
        self.dist_to(self.weight_prior, *args, **kwargs)
        if self.bias is not None:
            self.dist_to(self.bias_prior, *args, **kwargs)

        return super().to(*args, *kwargs)

    def dist_to(self, dist, *args, **kwargs):
        for key, value in dist.__dict__.items():
            if torch.is_tensor(value):
                dist.__dict__[key] = value.to(*args, **kwargs)
            elif isinstance(value, distributions.Distribution):
                self.dist_to(value, *args, **kwargs)


class PyroLogSoftMax(nn.LogSoftmax, PyroModule):
    def __init__(self):
        super().__init__(dim=-1)


class PyroSoftMax(nn.Softmax, PyroModule):
    def __init__(self):
        super().__init__(dim=-1)


class PyroSigmoid(nn.Sigmoid, PyroModule):
    pass


class PyroTanh(nn.Tanh, PyroModule):
    pass


class PyroReLU(nn.ReLU, PyroModule):
    pass


class BNNNotSampledError(Exception):
    pass

class PyroVI(nn.Sequential, PyroModule):
    def __init__(self, *args, sigma=1.0, step_size=1e-6, num_steps=200, classification=False):
        # if PyroSoftMax in [type(item) for item in args]:
        #     self.classification = True
        # else:
        #     self.classification = False
        self.classification = classification
        super().__init__(*args)

        self.sigma = sigma

    def forward(self, X, y=None):
        # mean = pyro.deterministic('mean', super().forward(X.flatten(start_dim=1, end_dim=-1)))
        mean = pyro.deterministic('mean', super().forward(X.flatten(start_dim=1, end_dim=-1)))
        # std = pyro.sample('sigma', dist.LogNormal(0., 1))

        if y is not None:
            with pyro.plate("data", X.shape[0], device=X.device):
                if self.classification:
                    # lhat = F.log_softmax(mean, dim=-1)
                    # lhat = F.softmax(mean, dim=-1)
                    obs = pyro.sample("obs", Categorical(logits=mean), obs=y)
                else:
                    obs = pyro.sample("obs", dist.Normal(mean.squeeze(-1), self.sigma), obs=y)
                    # obs = pyro.sample("obs", dist.Normal(mean.squeeze(-1), std*self.sigma), obs=y)
        return mean

    def vi(self, train_loader, args):
        guide = AutoNormal(self)

        adam = pyro.optim.Adam({"lr": args.lr})
        svi = SVI(self, guide, adam, loss=Trace_ELBO())

        pyro.clear_param_store()

        store_loss = []
        for epoch in range(args.epochs):
            loss = 0.
            corr_pred = 0.

            for i, (x_batch, y_batch) in enumerate(train_loader):
                if self.classification:
                    loss += svi.step(x_batch, y_batch.argmax(dim=-1))
                    pred = self.predict_mean(guide, x_batch, num_samples=10)
                    corr_pred += (pred.argmax(dim=-1) == y_batch.argmax(dim=-1)).sum().item()
                else:
                    loss += svi.step(x_batch, y_batch[:, 0])

            if epoch % 1 == 0:
                if self.classification:
                    print(f"""\n[Epoch {epoch + 1}]\t loss: {loss / len(train_loader.dataset):.2f} \t  accuracy: {100 * corr_pred / len(train_loader.dataset):.2f}""",
                          end="\t")
                else:
                    print(f"""\n[Epoch {epoch + 1}]\t loss: {loss / len(train_loader.dataset):.2f}""", end="\t")

            store_loss.append(loss / len(train_loader.dataset))


        # num_iterations = 1500 # 1500
        # X, y = train_loader
        # # for i, train_loader in enumerate(train_loaders):
        # for j in range(num_iterations):
        #     # calculate the loss and take a gradient step
        #     loss = svi.step(X, y[:,0])
        #     if j % 100 == 0:
        #         print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(y)))

        #     # loss = 0
        #     # for batch_id, data in enumerate(train_loader):
        #     #     # calculate the loss and take a gradient step
        #     #     # X_test = data[0].view(-1,28*28)
        #     #     # test = self.predict_dist(guide, X_test[0], num_samples=1)
        #     #     loss += svi.step(data[0].view(-1,28*28), data[1])
        #     #
        #     #     # if j % 100 == 0:
        #     #     #     print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data[1])))
        #     #
        #     # normalizer_train = len(train_loader.dataset)
        #     # total_epoch_loss_train = loss / normalizer_train
        #     #
        #     # print("Train Loader: {}, Epoch: {}, Loss: {}".format(i,j, total_epoch_loss_train))

        plt.plot(range(args.epochs), store_loss)
        plt.show()
        return guide

    def predict_inter_layer(self, guide, X, num_samples=10):
        for i in range(num_samples):
            nn = torch.nn.Sequential(*list(self.sample_nn(guide).children())[:-1])

    def predict_dist(self, guide, X, num_samples=None):
        predictive = Predictive(self, guide=guide, num_samples=num_samples,
                                return_sites=("linear.weight", "obs", "_RETURN"))
        y = predictive(X)['_RETURN']
        return y

    def predict_mean(self, guide, X, num_samples=None):
        y = self.predict_dist(guide, X, num_samples=num_samples)
        return y.mean(0)

    def evaluate(self, guide, test_loader, num_samples=10):
        pyro.set_rng_seed(0)

        with torch.no_grad():
            corr_preds = 0.
            for x_batch, y_batch in test_loader:
                outputs = self.predict_mean(guide, x_batch, num_samples=num_samples)
                predictions = outputs.argmax(-1)
                labels = y_batch.argmax(-1)
                corr_preds += (predictions == labels).sum().item()

            accuracy = 100 * corr_preds / len(test_loader.dataset)
            print("Accuracy: %.2f%%" % (accuracy))
            return accuracy

    def ibp(self, guide, x_data, epsilon, n_samples=10):
        input_bounds = HyperRectangle.from_eps(x_data.flatten(), epsilon)
        input_bounds.lower = torch.clip(input_bounds.lower, 0, torch.inf)

        lin_idxs = torch.tensor(
            [int(i) for i in list(self._modules) if type(self._modules[i]) == PyroBatchLinear])
        relu_idxs = torch.tensor(
            [int(i) for i in list(self._modules) if type(self._modules[i]) == PyroReLU])
        softmax_idxs = torch.tensor(
            [int(i) for i in list(self._modules) if type(self._modules[i]) == PyroSoftMax])
        closest_lin_layer = torch.tensor([lin_idxs[torch.where(lin_idxs - i <= 0)[0][-1]] for i in range(len(self._modules))])
        nr_outputs = [self._modules[f"{i}"].out_features for i in closest_lin_layer]

        store = {int(i): torch.zeros((n_samples, 2, nr_outputs[i])) for i in relu_idxs}

        for relu_idx in relu_idxs:
            for i in range(n_samples):
                nn = torch.nn.Sequential(*list(self.sample_nn(guide).children())[:relu_idx+1])
                nn_fact = factory.build(nn)
                ibs = nn_fact.ibp(input_bounds)
                store[int(relu_idx)][i, 0], store[int(relu_idx)][i, 1] = ibs.lower, ibs.upper

        if len(softmax_idxs) > 1:
            raise NotImplementedError

        for softmax_idx in softmax_idxs:
            store['logits'] = torch.zeros((n_samples, 2, nr_outputs[softmax_idx - 1]))
            store['probs'] = torch.zeros((2, nr_outputs[softmax_idx]))

            for i in range(n_samples):
                nn = torch.nn.Sequential(*list(self.sample_nn(guide).children())[:softmax_idx])
                nn_fact = factory.build(nn)
                ibs_logits = nn_fact.ibp(input_bounds)
                store['logits'][i, 0], store['logits'][i, 1] = ibs_logits.lower, ibs_logits.upper

        out = {i: torch.mean(store[i], dim=0) for i in store}

        if len(softmax_idxs) > 0:
            ibs_probs = bound_softmax(torch.cat((out['logits'][0][None], out['logits'][1][None])))
            out['probs'] = torch.vstack((ibs_probs['b_L'], ibs_probs['b_U']))

        return out

    def sample_nn(self, guide):
        sample = guide()
        layers = []
        for layer_idx in self._modules:
            if self._modules[layer_idx].__class__.__bases__[0] == torch.nn.Linear:
                lin_layer = self._modules[layer_idx].__class__.__bases__[0](in_features=self._modules[layer_idx].in_features,
                                                                           out_features=self._modules[layer_idx].out_features)
                lin_layer.weight = torch.nn.Parameter(sample[f"{layer_idx}.weight"])
                lin_layer.bias = torch.nn.Parameter(sample[f"{layer_idx}.bias"])
                layers += [lin_layer]
            else:
                layers += [self._modules[layer_idx].__class__.__bases__[0]()]
        nn = torch.nn.Sequential(*layers)
        return nn

    def fgsm_attack(self, guide, x_data, true_pred, epsilon, n_samples=10):
        nr_classes = true_pred.shape[0]
        true_label = torch.argmax(true_pred)

        data = x_data.flatten()
        data.requires_grad = True

        seeds = list(range(n_samples))

        final_logits_store = []
        final_probs_store = []
        final_labels_store = []

        for seed in seeds:
            pyro.set_rng_seed(seed)

            # nn = torch.nn.Sequential(*list(self.sample_nn(guide).children())[:-1])
            nn = torch.nn.Sequential(*list(self.sample_nn(guide).children()))
            attack = torchattacks.FGSM(nn, epsilon)
            data_perturbed = attack(data, true_label)

            # test_before = nn(data).detach().numpy()
            # test_after = nn(data_perturbed).detach().numpy()

            # ## old approach:
            # init_logits = nn(data)
            # init_pred = torch.nn.functional.softmax(init_logits, dim=-1).flatten()          # print(init_pred.round(decimals=2).detach().numpy())
            # init_label = torch.argmax(init_pred)
            #
            # # Calculate the loss
            # loss = torch.nn.functional.nll_loss(init_pred, true_pred)   # or use torch.nn.CrossEntropyLoss()
            #
            # # Zero all existing gradients
            # nn.zero_grad()
            #
            # # Calculate gradients of model in backward pass
            # loss.backward()
            #
            # # Collect datagrad
            # data_grad = data.grad.data
            #
            # # Call FGSM Attack
            # data_perturbed = fgsm_attack(data, epsilon, data_grad)

            # Re-classify the perturbed image
            final_probs_sampled = nn(data_perturbed)
            final_probs_store.append(final_probs_sampled)

        final_probs_store = torch.stack(final_probs_store)
        robust = torch.all(torch.max(final_probs_store, dim=1).indices == true_label)
        probs_min = torch.min(final_probs_store, dim=0).values
        return robust, probs_min


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image