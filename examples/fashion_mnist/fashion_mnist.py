import os
from examples.vi import PyroVI, PyroBatchLinear, PyroReLU, PyroSoftMax, PyroLogSoftMax
import examples.utils as utils
import pyro
from support import pickle_load
import torch

#########################################################################################
# An example of Variational Inference BNN for classification on the fashion-mnist dataset
#########################################################################################

PATH = os.getcwd() + "/examples/fashion_mnist"
PATH_DATA = PATH + "/data"


def get_net(args, train=False):
    prior_info = {'weight_prior_scale': args.prior_weight_scale,
                  'bias_prior_scale': args.prior_bias_scale}
    if train:
        last_layer = PyroLogSoftMax()
    else:
        last_layer = PyroSoftMax()

    if args.layers == 1:
        net = PyroVI(
            PyroBatchLinear(28*28, args.width, **prior_info),
            PyroReLU(),
            PyroBatchLinear(args.width, 10, **prior_info),
            last_layer,
            sigma=args.sigma,
            classification=True
        )
    else:
        hidden_layers = [item for _ in range(args.layers - 1) for item in
                         [PyroReLU(), PyroBatchLinear(args.width, args.width, **prior_info)]]
        net = PyroVI(
            PyroBatchLinear(28*28, args.width, **prior_info),
            *hidden_layers,
            PyroReLU(),
            PyroBatchLinear(args.width, 10, **prior_info),
            last_layer,
            sigma=args.sigma,
            classification=True
        )
    return net



def load_vogn_bnn(net, args, train_loader):
    params = pickle_load(f"{PATH_DATA}/VOGN/FASHION_MNIST_{args.layers}_{args.width}")

    ## set params (improvised method. Alternative elagant way to initialize the guide?)
    args.epochs = 1
    guide = net.vi(train_loader, args)

    param_store = pyro.get_param_store()
    for key, value in param_store.items():
        if 'weight' in key:
            new_value = torch.from_numpy(params[key]).moveaxis(0, 1)
        else:
            new_value = torch.from_numpy(params[key])

        value.data = new_value
        param_store.replace_param(key, value.to(args.device), value)

    utils.save(net, guide, PATH_DATA, args)

    return guide


def test(net, guide, test_loader, args):
    net.evaluate(guide, test_loader, num_samples=10)


def get_bnn(args):
    train_loader, test_loader, inp_shape, out_shape = \
        utils.data_loaders(dataset_name=args.example, batch_size_train=128, n_inputs=args.n_inputs, batch_size_test=128,
                           shuffle=False)

    # utils.plot_grid_images(next(iter(test_loader))[0], next(iter(test_loader))[1])
    # torch.argmax(next(iter(test_loader))[1], dim=1)
    bnn = get_net(args)

    if utils.available(PATH_DATA, args) and not args.train:
        bnn, guide = utils.load(bnn, PATH_DATA, args)
    elif args.trained_by == 'vogn':
        guide = load_vogn_bnn(bnn, args, train_loader)
        args.test = True
    else:
        bnn = get_net(args, train=True)
        guide = bnn.vi(train_loader, args)
        bnn = get_net(args)
        utils.save(bnn, guide, PATH_DATA, args)
        args.test = True

    if args.test:
        test(bnn, guide, test_loader=test_loader, args=args)

    return bnn, guide, test_loader