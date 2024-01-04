import os
import torch
from torch import distributions
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from sys import platform

from examples.vi import PyroVI, PyroBatchLinear, PyroReLU
import examples.utils as utils

# mpl.use('TkAgg')

###################################################################################
# An example of Variational Inference BNN for regression on a 2d noisy sine dataset
###################################################################################

PATH = os.getcwd() + "/examples/noisy_sine_2"
PATH_DATA = PATH + "/data"
PATH_PLOTS = PATH + '/plots'

def get_net(args):
    if args.layers == 1:
        net = PyroVI(
            PyroBatchLinear(2, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    else:
        hidden_layers = [item for _ in range(args.layers - 1) for item in
                         [PyroReLU(), PyroBatchLinear(args.width, args.width)]]
        net = PyroVI(
            PyroBatchLinear(2, args.width),
            *hidden_layers,
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    return net


@torch.no_grad()
def test(net, guide, args, lin_bound_info=None, view=2, dim=0., loc=0.):
    if view == 3:
        # 3D
        x_test = torch.linspace(-1.0, 1.0, 100)
        X1_test, X2_test = torch.meshgrid(x_test, x_test)
        X_test = torch.cat((X1_test[:,:,None], X2_test[:,:,None]), dim=2).flatten(start_dim=0, end_dim=1)

        Y_test_dist = net.predict_dist(guide, X_test, num_samples=100)

        Y_mean = Y_test_dist.mean(0)
        Y_std = Y_test_dist.std(0)

        # plot intended mean
        # fig, ax = plt.subplots(subplot_kw={'projection': "3d"})
        # x_train = torch.linspace(-1.0, 1.0, 100)
        # X1_train, X2_train = torch.meshgrid(x_train, x_train)
        # X_train = torch.cat((X1_train[:,:,None], X2_train[:,:,None]), dim=2).flatten(start_dim=0, end_dim=1)
        # Y_train = utils.f_2d_func(X_train)
        # surf = ax.plot_surface(X1_train, X2_train, Y_train.reshape(X1_train.shape), cmap=cm.coolwarm, linewidth=0,
        #                        antialiased=False)
        # plt.title('2D Noisy Sine System')
        # plt.show()

        # plot predicted mean
        fig, ax = plt.subplots(subplot_kw={'projection': "3d"}, figsize=(4.8 * 2, 4.8 * 2))
        surf = ax.plot_surface(X1_test, X2_test, Y_mean.reshape(X1_test.shape), cmap=cm.coolwarm, linewidth=0,
                               antialiased=False)
        ax.set_xlabel('x', fontsize=30, labelpad=10)
        ax.set_ylabel('y', fontsize=30, labelpad=10)
        ax.set_zlabel('z', fontsize=30, labelpad=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.set_xticklabels([-1, 0, 1], fontsize=20)
        ax.set_yticklabels([-1, 0, 1], fontsize=20)
        ax.set_zticklabels([-1, 0, 1], fontsize=20)
        plt.title('2D Noisy Sine Model', fontsize=30)
        plt.savefig(f"{PATH_PLOTS}/{utils.file_name(args)}")
        plt.show()

        if lin_bound_info is not None:
            # plot bounds
            x1_int = torch.tensor([lin_bound_info['x_center'][0] - lin_bound_info['epsilon'],
                                   lin_bound_info['x_center'][0] + lin_bound_info['epsilon']])
            x2_int = torch.tensor([lin_bound_info['x_center'][1] - lin_bound_info['epsilon'],
                                   lin_bound_info['x_center'][1] + lin_bound_info['epsilon']])
            x1_sys = torch.linspace(1. * x1_int[0] - 0. * x1_int[1], 1. * x1_int[1] - 0. * x1_int[0], 100)
            x2_sys = torch.linspace(1. * x2_int[0] - 0. * x2_int[1], 1. * x2_int[1] - 0. * x2_int[0], 100)

            X1_lbs, X2_lbs = torch.meshgrid(x1_int, x2_int)
            X1_sys, X2_sys = torch.meshgrid(x1_sys, x2_sys)

            X_lbs = torch.cat((X1_lbs[:, :, None], X2_lbs[:, :, None]), dim=2).flatten(start_dim=0, end_dim=1)
            Y_u_bound = torch.einsum('ij,kj->ki', lin_bound_info['A_U_bound'], X_lbs) + lin_bound_info['b_U_bound']
            Y_l_bound = torch.einsum('ij,kj->ki', lin_bound_info['A_L_bound'], X_lbs) + lin_bound_info['b_L_bound']

            y_int = [1.1 * torch.min(Y_l_bound) - 0.1 * torch.max(Y_u_bound),
                          1.1 * torch.max(Y_u_bound) - 0.1 * torch.min(Y_l_bound)]

            X_sys = torch.cat((X1_sys[:, :, None], X2_sys[:, :, None]), dim=2).flatten(start_dim=0, end_dim=1)
            Y_sys_dist = net.predict_dist(guide, X_sys, num_samples=100)
            Y_sys_mean = Y_sys_dist.mean(0)

            fig, ax = plt.subplots(subplot_kw={'projection': "3d"}, figsize=(4.8* 2, 4.8 * 2))
            surf = ax.plot_surface(X1_lbs, X2_lbs, Y_l_bound.reshape(X1_lbs.shape), color='r', linewidth=0,
                                   antialiased=False, alpha=0.5)
            surf = ax.plot_surface(X1_sys, X2_sys, Y_sys_mean.reshape(X1_sys.shape), cmap=cm.coolwarm, linewidth=0,
                                   antialiased=False, alpha=0.4)
            surf = ax.plot_surface(X1_lbs, X2_lbs, Y_u_bound.reshape(X1_lbs.shape), color='g', linewidth=0,
                                   antialiased=False, alpha=0.5)

            ax.set_xlim(x1_sys[0]-0.005, x1_sys[-1]+0.005)
            ax.set_ylim(x2_sys[0]-0.005, x2_sys[-1]+0.005)
            ax.set_zlim(*y_int)

            x1_ticks = [x1_int[0].round(decimals=3).detach().numpy(), x1_int[1].round(decimals=3).detach().numpy()]
            x1_ticks.append(0.5*(x1_ticks[0]+x1_ticks[1]))
            x2_ticks = [x2_int[0].round(decimals=3).detach().numpy(), x2_int[1].round(decimals=3).detach().numpy()]
            x2_ticks.append(0.5*(x2_ticks[0]+x2_ticks[1]))
            # y_ticks = list(torch.arange(torch.floor(y_int[0] * 100) / 100, torch.ceil(y_int[1] * 100) / 100, 0.1).detach().numpy())
            y_ticks = [1., 1.05,1.1]
            # y_ticks = [0.2, 0.6, 1.]

            ax.set_xticks(x1_ticks)
            ax.set_yticks(x2_ticks)
            ax.set_zticks(y_ticks)
            ax.set_xticklabels(['{:.2f}'.format(item) for item in x1_ticks], fontsize=15)
            ax.set_yticklabels(['{:.2f}'.format(item) for item in x2_ticks], fontsize=15)
            ax.set_zticklabels(['{:.2f}'.format(item) for item in y_ticks], fontsize=15)

            ax.set_xlabel('x', fontsize=30, labelpad=10)
            ax.set_ylabel('y', fontsize=30, labelpad=10)
            ax.set_zlabel('z', fontsize=30, labelpad=10)

            plt.title('Linear Bounds', fontsize=30)
            ax.view_init(elev=20, azim=70, roll=0)
            # ax.view_init(elev=20, azim=135, roll=0)
            plt.savefig(f"{PATH_PLOTS}/{utils.file_name(args)}_bounds")
            plt.show()

            print('test')
    elif view == 2:
        x_test = torch.linspace(-2.0, 2.0, 1000)
        X_test = torch.ones((x_test.shape[0], 2)) * loc
        X_test[:, dim] = x_test

        y_test_dist = net.predict_dist(guide, X_test, num_samples=100)

        y_mean = y_test_dist.mean(0)
        y_std = y_test_dist.std(0)

        plt.figure(figsize=(6.4 * 2, 4.8 * 2))

        # plot samples to show uncertainty
        for i in range(y_test_dist.size(0)):
            plt.plot(X_test[:, dim], y_test_dist[i], color='y', alpha=0.1, label='Uncertainty' if i == 0 else None)

        # plot predicted mean
        plt.plot(X_test[:, dim], y_mean, 'r-', label='Predictive mean')

        # illustrate uncertainty based on std by fill
        plt.fill_between(X_test[:, dim].ravel(), (y_mean + y_std * 3).flatten(), (y_mean - y_std * 3).flatten(),
                         alpha=0.5, label='Uncertainty')

        # plot mimic of training points (real points hard to extract for specific slice)
        x_train = torch.linspace(-1.0, 1.0, 1000)
        X_train = torch.ones((x_train.shape[0], 2)) * loc
        X_train[:, dim] = x_train
        y_train = utils.f_2d_func_noise(X_train)
        plt.scatter(X_train[:, dim], y_train, marker='+', color='b', label='Training data')

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

                plt.plot(X_bound[:, dim], upper_y, '-g', linewidth=2, label='own')
                plt.plot(X_bound[:, dim], lower_y, '-g', linewidth=2)

                plt.xlim(interval[0, dim] - 0.2, interval[1, dim] + 0.2)
                plt.ylim(-2, 2)

                plt.plot(X_bound[:, dim], torch.ones(X_bound[:, dim].shape) * 1.6210, '-k', label='deep mind')


        # settings
        plt.title('Noisy Sine 2D')
        plt.legend()
        plt.xlabel('focus dim {} - other dims locket at {:.2f}'.format(dim, loc))
        plt.ylim(-2,2)
        plt.show()


def get_bnn(args):
    bnn = get_net(args)

    train_loader, test_loader, inp_shape, out_shape = \
        utils.data_loaders(dataset_name=args.example, batch_size_train=2*2**13, batch_size_test=2*2**13, shuffle=False)

    if utils.available(PATH_DATA, args) and not args.train:
        bnn, guide = utils.load(bnn, PATH_DATA, args)
    else:
        guide = bnn.vi(train_loader, args)
        utils.save(bnn, guide, PATH_DATA, args)
        args.test = True

    if args.test:
        test(bnn, guide, args=args, view=3)
        test(bnn, guide, args=args, view=2, dim=0, loc=0)
        test(bnn, guide, args=args, view=2, dim=1, loc=0)

    return bnn, guide, train_loader
