import os
import torch
import matplotlib.pyplot as plt
import pyro

from examples.vi import PyroVI, PyroBatchLinear, PyroReLU
import examples.utils as utils
import support


###################################################################################################
# An example of Variational Inference BNN for regression on the Kin8nm dataset
###################################################################################################

PATH = os.getcwd() + "/examples/kin8nm"
PATH_DATA = PATH + "/data"


def get_net(args):
    if args.layers == 1:
        net = PyroVI(
            PyroBatchLinear(8, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    else:
        net = PyroVI(
            PyroBatchLinear(8, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    return net


@torch.no_grad()
def test(net, guide, test_loader, train_loader=None, lin_bound_info=None, view=2, dim=0, loc=None):
    if view == 3:
        raise NotImplementedError
    elif view == 2:
        # X_test, y_test_true = next(iter(test_loader))
        # y_test_dist = net.predict_dist(guide, X_test, num_samples=100)
        # y_test_mean = y_test_dist.mean(0)
        # y_test_std = y_test_dist.std(0)
        #
        # plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        # plt.scatter(X_test[:, dim], y_test_mean, s=y_test_std * 3)
        # plt.title('std')
        # plt.show()
        #
        # plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        # plt.plot(X_test[:, dim], y_test_mean, 'r*', label='Predictive mean - Test Data')
        # plt.plot(X_test[:, dim], y_test_true, 'y+', label='Measurements - Test Data')
        #
        # # settings
        # plt.title('Mean-field Gaussian VI BNN prediction')
        # plt.legend()
        # plt.xlabel('focus dim {} - other dims locket at {}'.format(dim, loc))
        # plt.show()

        dims = next(iter(test_loader))[0].shape[1]
        if lin_bound_info is None:
            interval = torch.vstack((torch.zeros(1, dims), torch.ones(1, dims)))
        else:
            interval = torch.vstack((lin_bound_info['x_center'] - lin_bound_info['epsilon'],
                                     lin_bound_info['x_center'] + lin_bound_info['epsilon']))

        plt.figure(figsize=(6.4 * 2, 4.8 * 2))

        # predictions at loc
        x_dim_test_loc = torch.linspace(interval[0, dim] - 0.2, interval[1, dim] + 0.2, 100)
        if loc is None:
            loc = torch.mean(interval, dim=0)
        X_test_loc = loc.repeat(x_dim_test_loc.shape[0], 1)
        X_test_loc[:, dim] = x_dim_test_loc

        y_test_dist_loc = net.predict_dist(guide, X_test_loc, num_samples=100)
        y_test_mean_loc = y_test_dist_loc.mean(0)
        y_test_mean_std = y_test_dist_loc.std(0)

        plt.plot(X_test_loc[:, dim], y_test_mean_loc, 'r-', label='Predictive mean - at loc')

        # illustrate uncertainty based on std by fill
        plt.fill_between(X_test_loc[:, dim].ravel(), (y_test_mean_loc + y_test_mean_std * 3).flatten(),
                         (y_test_mean_loc - y_test_mean_std * 3).flatten(), alpha=0.5, label='Uncertainty')

        # formal bounds
        if lin_bound_info is None or (torch.any(loc <= interval[0]) or torch.any(loc >= interval[1])):
            pass
        else:
            # formal bounds at loc
            x_dim_bound_loc = torch.linspace(interval[0, dim], interval[1, dim], 1000)
            X_bound_loc = loc.repeat(x_dim_bound_loc.shape[0], 1)
            X_bound_loc[:, dim] = x_dim_bound_loc

            upper_y = torch.einsum('ij,kj->ki', lin_bound_info['lbs']['after']['A_U'], X_bound_loc) + \
                      lin_bound_info['lbs']['after']['b_U']
            lower_y = torch.einsum('ij,kj->ki', lin_bound_info['lbs']['after']['A_L'], X_bound_loc) + \
                      lin_bound_info['lbs']['after']['b_L']

            plt.plot(X_bound_loc[:, dim], upper_y, '-g', linewidth=2, label='own')
            plt.plot(X_bound_loc[:, dim], lower_y, '-g', linewidth=2)

            # plt.plot(X_bound_loc[:, dim], torch.ones(X_bound_loc[:, dim].shape) * -0.698, '-k', label='deep mind')

        # settings
        plt.title('Powerplant')
        plt.legend()
        plt.xlabel('focus dim {} - other dims locket at {}'.format(dim, loc))
        plt.show()


def load_vogn_bnn(net, args, train_loader):
    params = support.pickle_load(f"{PATH_DATA}/VOGN/kin8nm1_{args.layers}_{args.width}")

    ## set params (improvised method. Alternative elagant way to initialize the guide?)
    args.epochs = 1
    guide = net.vi(train_loader, args)

    param_store = pyro.get_param_store()
    for key, value in param_store.items():
        new_value = torch.from_numpy(params[key.replace('AutoNormal.', '')])

        value.data = new_value
        param_store.replace_param(key, value.to(args.device), value)

    utils.save(net, guide, PATH_DATA, args)

    return guide


def get_bnn(args):
    bnn = get_net(args)

    dataset_size = 8192
    train_loader, test_loader, inp_shape, out_shape = \
        utils.data_loaders(dataset_name=args.example, batch_size_train=256, n_inputs=dataset_size,
                           batch_size_test=dataset_size, shuffle=False)

    if utils.available(PATH_DATA, args) and not args.train:
        bnn, guide = utils.load(bnn, PATH_DATA, args)
    elif args.trained_by == 'vogn':
        guide = load_vogn_bnn(bnn, args, train_loader)
        args.test = True
    else:
        guide = bnn.vi(train_loader, args)
        utils.save(bnn, guide, PATH_DATA, args)
        args.test = True

    if args.test:
        test(bnn, guide, test_loader=test_loader, train_loader=train_loader, view=2, dim=0)

    return bnn, guide, train_loader