import os
import torch
import matplotlib.pyplot as plt

from examples.vi import PyroVI, PyroBatchLinear, PyroReLU
import examples.utils as utils
from examples.bnn import BNN

######################################################################################################
# An example of Variational Inference BNN for regression on the Concrete Compressive Strength Data Set
# https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
######################################################################################################

PATH = os.getcwd() + "/examples/concrete"
PATH_DATA = PATH + "/data"


@torch.no_grad()
def test(net, test_loader, train_loader=None , lin_bound_info=None, view=2, dim=0, loc=0.):
    if view == 3:
        raise NotImplementedError
    elif view == 2:
        # \TODO don't check at loc = 0, check at realistic loc

        X_test, y_test_true = next(iter(test_loader))
        y_test_dist = net.forward(X_test, n_samples=100, dist=True)
        y_test_mean = y_test_dist.mean(0)
        y__test_std = y_test_dist.std(0)

        plt.figure(figsize=(6.4 * 2, 4.8 * 2))

        # # plot samples to show uncertainty
        # for i in range(y_test_dist.size(0)):
        #     plt.plot(X_test[:, dim], y_test_dist[i], color='y', alpha=0.1, label='Uncertainty' if i == 0 else None)

        # # plot predicted mean
        # plt.plot(X_test[:, dim], y_test_mean, 'r*', label='Predictive mean - Test Data')
        # # plot true mean
        # plt.plot(X_test[:, dim], y_test_true, 'y+', label='Measurements - Test Data')

        # # illustrate uncertainty based on std by fill
        # plt.fill_between(X_test[:, dim].ravel(), (y_mean + y_std * 3).flatten(), (y_mean - y_std * 3).flatten(),
        #                  alpha=0.5, label='Uncertainty')

        if train_loader is not None:
            # plot mimic of training points (real points hard to extract for specific slice)
            X_train, y_train_true = next(iter(train_loader))
            plt.scatter(X_train[:, dim], y_train_true, marker='+', color='k', label='Measurements - Train data')

            y_train_dist = net.forward(X_train, n_samples=100, dist=True)
            y_train_mean = y_train_dist.mean(0)
            y_train_std = y_train_dist.std(0)
            plt.scatter(X_train[:, dim], y_train_mean, marker='+', color='b', label='Predictive mean - Train Data')

        if lin_bound_info is not None:
            interval = torch.vstack((lin_bound_info['x_center'] - lin_bound_info['epsilon'],
                                     lin_bound_info['x_center'] + lin_bound_info['epsilon']))
            if loc <= interval[0, 1-dim] or loc >= interval[1, 1-dim]:
                pass
            else:
                x_bound = torch.linspace(interval[0, dim], interval[1, dim], 1000)
                X_bound = torch.ones((x_bound.shape[0], 2)) * loc
                X_bound[:, dim] = x_bound

                upper_y = torch.einsum('ij,kj->ki', lin_bound_info['A_U_bound'], X_bound) + lin_bound_info['b_U_bound']
                lower_y = torch.einsum('ij,kj->ki', lin_bound_info['A_L_bound'], X_bound) + lin_bound_info['b_L_bound']

                plt.plot(X_bound[:, dim], upper_y, '-k', linewidth=3)
                plt.plot(X_bound[:, dim], lower_y, '-k', linewidth=3)

                plt.xlim(lin_bound_info['x_center'] - lin_bound_info['epsilon'] - 0.5,
                         lin_bound_info['x_center'] + lin_bound_info['epsilon'] + 0.5)

        # settings
        plt.title('Mean-field Gaussian VI BNN prediction')
        plt.legend()
        plt.xlabel('focus dim {} - other dims locket at {}'.format(dim, loc))
        plt.show()


def get_bnn(args):
    dataset_size = 1030
    train_loader, test_loader, inp_shape, out_shape = \
        utils.data_loaders(dataset_name=args.example, batch_size_train=int((dataset_size*4/5)/4),
                           batch_size_test=dataset_size, shuffle=True)

    bnn = BNN(dataset_name=args.example, hidden_size=args.width, activation=args.activation,
              architecture='fc{}'.format(args.layers), epochs=args.epochs, lr=args.lr, n_samples=None, warmup=None,
              input_shape=inp_shape, output_size=out_shape, trained_by=args.trained_by)

    if bnn.available(PATH_DATA) and not args.train:
        bnn.load(device=args.device, path=PATH_DATA)
    else:
        bnn.train(train_loader=train_loader, device=args.device, path=PATH_DATA)
        args.test = True

    if args.test:
        test(bnn, test_loader=test_loader, train_loader=train_loader, view=2, dim=0)

    return bnn, train_loader