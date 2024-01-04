import os
import torch
import matplotlib.pyplot as plt

from examples.vi import PyroVI, PyroBatchLinear, PyroReLU
import examples.utils as utils


###################################################################################################
# An example of Variational Inference BNN for regression on the Combined Cycle Power Plant Data Set
# https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
###################################################################################################

PATH = os.getcwd() + "/examples/powerplant"
PATH_DATA = PATH + "/data"


def get_net(args):
    if args.layers == 1:
        net = PyroVI(
            PyroBatchLinear(4, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    else:
        net = PyroVI(
            PyroBatchLinear(4, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    return net


@torch.no_grad()
def test(net, guide, test_loader, train_loader=None, lin_bound_info=None, view=2, dim=0, loc=0.):
    if view == 3:
        raise NotImplementedError
    elif view == 2:
        if lin_bound_info is None:
            X_test, y_test_true = next(iter(test_loader))
            y_test_dist = net.predict_dist(guide, X_test, num_samples=100)
            y_test_mean = y_test_dist.mean(0)
            y_test_std = y_test_dist.std(0)

            plt.figure(figsize=(6.4 * 2, 4.8 * 2))
            plt.scatter(X_test[:, dim], y_test_mean, s=y_test_std * 3)
            plt.title('std')

            plt.figure(figsize=(6.4 * 2, 4.8 * 2))
            plt.plot(X_test[:, dim], y_test_mean, 'r*', label='Predictive mean - Test Data')
            plt.plot(X_test[:, dim], y_test_true, 'y+', label='Measurements - Test Data')

            # if train_loader is not None:
            #     # plot mimic of training points (real points hard to extract for specific slice)
            #     X_train, y_train_true = next(iter(train_loader))
            #     plt.scatter(X_train[:, dim], y_train_true, marker='*', color='r', label='Measurements - Train data')
            #
            #     y_train_dist = net.predict_dist(guide, X_train, num_samples=100)
            #     y_train_mean = y_train_dist.mean(0)
            #     y_train_std = y_train_dist.std(0)
            #     plt.scatter(X_train[:, dim], y_train_mean, marker='+', color='g', label='Predictive mean - Train Data')

            # settings
            plt.title('Mean-field Gaussian VI BNN prediction')
            plt.legend()
            plt.xlabel('focus dim {} - other dims locket at {}'.format(dim, loc))
            plt.show()

        else:
            interval = torch.vstack((lin_bound_info['x_center'] - lin_bound_info['epsilon'],
                                     lin_bound_info['x_center'] + lin_bound_info['epsilon']))

            plt.figure(figsize=(6.4 * 2, 4.8 * 2))

            # predictions at loc
            x_dim_test_loc = torch.linspace(interval[0, dim] - 0.2, interval[1, dim] + 0.2, 100)
            X_test_loc = loc.repeat(x_dim_test_loc.shape[0], 1)
            X_test_loc[:, dim] = x_dim_test_loc

            y_test_dist_loc = net.predict_dist(guide, X_test_loc, num_samples=100)
            y_test_mean_loc = y_test_dist_loc.mean(0)
            y_test_mean_std = y_test_dist_loc.std(0)

            plt.plot(X_test_loc[:, dim], y_test_mean_loc, 'r-', label='Predictive mean - at loc')

            # illustrate uncertainty based on std by fill
            plt.fill_between(X_test_loc[:, dim].ravel(), (y_test_mean_loc + y_test_mean_std * 3).flatten(),
                             (y_test_mean_loc - y_test_mean_std * 3).flatten(), alpha=0.5, label='Uncertainty')

            # # predictions at interval (from test data)
            # X_test_data = torch.vstack([item[0] for item in test_loader.dataset])
            # y_test_true_data = torch.vstack([item[1] for item in test_loader.dataset])
            #
            # mask_cols = torch.ones(len(loc), dtype=bool)
            # mask_cols[dim] = False
            # mask_loc = torch.logical_and(torch.all(X_test_data[:, mask_cols] >= interval[0, mask_cols], dim=1),
            #                              torch.all(X_test_data[:, mask_cols] <= interval[1, mask_cols], dim=1))
            # X_test_data = X_test_data[mask_loc, :]
            #
            # y_test_dist_data = net.forward(X_test_data, n_samples=100, dist=True)
            # y_test_mean_data = y_test_dist_data.mean(0)
            #
            # plt.plot(X_test_data[:, dim], y_test_mean_data, 'r*', label='Predictive mean - Test Data (all dims)')

            # formal bounds
            if torch.any(loc <= interval[0]) or torch.any(loc >= interval[1]):
                pass
            else:
                # formal bounds at loc
                x_dim_bound_loc = torch.linspace(interval[0, dim], interval[1, dim], 1000)
                X_bound_loc = loc.repeat(x_dim_bound_loc.shape[0], 1)
                X_bound_loc[:, dim] = x_dim_bound_loc

                upper_y = torch.einsum('ij,kj->ki', lin_bound_info['A_U_bound'], X_bound_loc) + lin_bound_info[
                    'b_U_bound']
                lower_y = torch.einsum('ij,kj->ki', lin_bound_info['A_L_bound'], X_bound_loc) + lin_bound_info[
                    'b_L_bound']

                plt.plot(X_bound_loc[:, dim], upper_y, '-g', linewidth=2, label='own')
                plt.plot(X_bound_loc[:, dim], lower_y, '-g', linewidth=2)

                # plt.plot(X_bound_loc[:, dim], torch.ones(X_bound_loc[:, dim].shape) * -0.698, '-k', label='deep mind')

                # # formal bounds at interval (from test data)
                # mask_dim = torch.logical_and(X_test_data[:, dim] >= interval[0, dim],
                #                              X_test_data[:, dim] <= interval[1, dim])
                # X_bound_data = X_test_data[mask_dim, :]
                #
                # upper_y_data = torch.einsum('ij,kj->ki', lin_bound_info['A_U_bound'], X_bound_data) + lin_bound_info[
                #     'b_U_bound']
                # lower_y_data = torch.einsum('ij,kj->ki', lin_bound_info['A_L_bound'], X_bound_data) + lin_bound_info[
                #     'b_L_bound']
                #
                # plt.plot(X_bound_loc[:, dim], torch.ones(X_bound_loc[:, dim].shape) * torch.max(upper_y_data), '-k')
                # plt.plot(X_bound_loc[:, dim], torch.ones(X_bound_loc[:, dim].shape) * torch.min(lower_y_data), '-k')
                #
                # plt.xlim(interval[0, dim] - 0.2, interval[1, dim] + 0.2)
                # # plt.ylim(-2, 2)

            # settings
            plt.title('Powerplant')
            plt.legend()
            plt.xlabel('focus dim {} - other dims locket at {}'.format(dim, loc))
            plt.show()


def get_bnn(args):
    bnn = get_net(args)

    dataset_size = 9568
    train_loader, test_loader, inp_shape, out_shape = \
        utils.data_loaders(dataset_name=args.example, batch_size_train=256, n_inputs=dataset_size,
                           batch_size_test=dataset_size, shuffle=True)

    if utils.available(PATH_DATA, args) and not args.train:
        bnn, guide = utils.load(bnn, PATH_DATA, args)
    else:
        guide = bnn.vi(train_loader, args)
        utils.save(bnn, guide, PATH_DATA, args)
        args.test = True

        # params = pickle_load('{}/{}_{}_{}'.format(PATH_DATA, args.example, args.layers, args.width))
        #
        # param_store = pyro.get_param_store()
        # for key, value in param_store.items():
        #     new_value = torch.from_numpy(params[key])
        #     if 'weight' in key:
        #         new_value = new_value.moveaxis(0, 1)
        #     value.data = new_value.data
        #     param_store.replace_param(key, value.to(args.device), value)
        # bnn.save(path=PATH_DATA)

    if args.test:
        test(bnn, guide, test_loader=test_loader, train_loader=train_loader, view=2, dim=0)

    return bnn, guide,train_loader

