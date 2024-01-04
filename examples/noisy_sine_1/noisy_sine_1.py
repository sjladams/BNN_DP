import os
import torch
import matplotlib.pyplot as plt
import pickle

import support
from examples.vi import PyroVI, PyroBatchLinear, PyroReLU
import examples.utils as utils


###################################################################################
# An example of Variational Inference BNN for regression on a 1d noisy sine dataset
###################################################################################

PATH = os.getcwd() + "/examples/noisy_sine_1"
PATH_DATA = PATH + "/data"
PATH_PLOTS = PATH + '/plots'


def get_net(args):
    if args.layers == 1:
        net = PyroVI(
            PyroBatchLinear(1, args.width),
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    else:
        hidden_layers = [item for _ in range(args.layers - 1) for item in
                         [PyroReLU(), PyroBatchLinear(args.width, args.width)]]
        net = PyroVI(
            PyroBatchLinear(1, args.width),
            *hidden_layers,
            PyroReLU(),
            PyroBatchLinear(args.width, 1),
            sigma=args.sigma
        )
    return net


@torch.no_grad()
def test(net, guide, args, train_loader=None, lin_bound_info=None):
    x_test = torch.linspace(-1.5, 1.5, 1000).view(-1, 1)
    y_test_dist = net.predict_dist(guide, x_test, num_samples=100)
    y_mean = y_test_dist.mean(0)
    y_std = y_test_dist.std(0)

    fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))

    for i in range(100):
        plt.plot(x_test, y_test_dist[i], color='y', alpha=0.1, label='Samples' if i == 0 else None)

    plt.plot(x_test[:,0], y_mean, 'r-', label='Predictive mean')

    plt.fill_between(x_test[:, 0].ravel(), (y_mean + y_std * 3).flatten(), (y_mean - y_std * 3).flatten(),
                     alpha=0.5, label='Uncertainty')

    # to_store = dict()
    if train_loader is not None:
        x_train, y_train = next(iter(train_loader))
        plt.scatter(x_train[:, 0][::4], y_train[::4], marker='+', color='b', label='Training data')

    if lin_bound_info is not None:
        interval = torch.tensor([lin_bound_info['x_center']-lin_bound_info['epsilon'],
                                 lin_bound_info['x_center']+lin_bound_info['epsilon']])
        x = torch.linspace(interval[0], interval[1], 1000)[:,None]

        upper_y = torch.einsum('ij,kj->ki', lin_bound_info['lbs']['after']['A_U'], x) + lin_bound_info['lbs']['after'][
            'b_U']
        lower_y = torch.einsum('ij,kj->ki', lin_bound_info['lbs']['after']['A_L'], x) + lin_bound_info['lbs']['after'][
            'b_L']

        interval_y = [1.5*torch.min(lower_y) - 0.5*torch.max(upper_y),
                      1.5*torch.max(upper_y) - 0.5*torch.min(lower_y)]
        ax.fill_between(torch.tensor([lin_bound_info['x_center'] - lin_bound_info['epsilon'],
                                      lin_bound_info['x_center'] + lin_bound_info['epsilon']]), -10, 10, alpha=0.05, color='k')

        plt.plot(x, upper_y, '-b', linewidth=2, label='Ours')
        plt.plot(x, lower_y, '-b', linewidth=2)

        plt.xlim(lin_bound_info['x_center']-2*lin_bound_info['epsilon'],
                 lin_bound_info['x_center']+2*lin_bound_info['epsilon'])

        plt.xticks([interval[0].round(decimals=3).detach().numpy(), interval[1].round(decimals=3).detach().numpy()],
                   fontsize=30)
        file_name = f"{utils.file_name(args)}_bounds"
        # plt.title('Affine Relaxation $V_0$', fontsize=30)

        for line_type, dm_std in zip(['-', '--', '-.'], [1., 2., 3. ]):
            dm_file = "summary_{}_{}_{}_deepmind_anneal=4000_eps={:.3f}_x_center={:.1f}_std={:.1f}".format(
                args.example,  args.layers, args.width, lin_bound_info['epsilon'], lin_bound_info['x_center'][0], dm_std)
            if os.path.exists(f"{PATH_DATA}/deepmind/anneal=4000/{dm_file}.pickle"):
                dm_info = support.pickle_load(f"{PATH_DATA}/deepmind/anneal=4000/{dm_file}")
                plt.plot(x, torch.ones(x.shape) * dm_info['eval_bounds'][1], '{}k'.format(line_type), linewidth=1.5,
                         label='FL {:.1f} std'.format(dm_std))
                plt.plot(x, torch.ones(x.shape) * dm_info['eval_bounds'][0], '{}k'.format(line_type), linewidth=1.5)

                interval_y = [1.05 * torch.tensor(dm_info['eval_bounds'][0]) - .05 * torch.tensor(dm_info['eval_bounds'][1]),
                              1.05 * torch.tensor(dm_info['eval_bounds'][1]) - .05 * torch.tensor(dm_info['eval_bounds'][0])]

        plt.ylim(*interval_y)
        plt.yticks(torch.arange(torch.floor(interval_y[0] * 2) / 2, torch.ceil(interval_y[1] * 2) / 2,
                                0.5).detach().numpy(), fontsize=30)
        plt.yticks([-2, -1, 0., 1.], fontsize=30)

        # to_store = {**to_store, 'x': x, 'upper_y': upper_y, 'lower_y': lower_y, 'lin_bound_info': lin_bound_info}
    else:
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.xticks([-1, 0., 1], fontsize=30)
        plt.yticks([-1., 0., 1.], fontsize=30)
        file_name = utils.file_name(args)
        # plt.title(f'Noisy Sine Model ($K={args.layers}$' + ',$n_{hid}$' + f'$={args.width}$)', fontsize=30)

    handles, labels = ax.get_legend_handles_labels()
    handles[0]._alpha = 1.
    # plt.legend(fontsize=30, handles=handles, labels=labels, loc='lower right')
    plt.xlabel('x', fontsize=35)
    plt.ylabel('y', fontsize=35)
    # plt.savefig(f"{PATH_PLOTS}/{file_name}", bbox_inches='tight')
    # pickle_out = open(f"{PATH_PLOTS}/{file_name}" "ww")
    # pickle.dump(ax, pickle_out)
    # pickle_out.close()

    # to_store = {**to_store, 'x_test': x_test, 'y_test_dist': y_test_dist, 'y_mean': y_mean, 'y_std': y_std}
    # support.pickle_dump(to_store, f"{PATH_PLOTS}/{file_name}")
    plt.show()


def get_bnn(args):
    bnn = get_net(args)

    train_loader, test_loader, inp_shape, out_shape = \
        utils.data_loaders(dataset_name=args.example, batch_size_train=2*2**12,
                     batch_size_test=2*2**12, shuffle=True)

    if utils.available(PATH_DATA, args) and not args.train:
        bnn, guide = utils.load(bnn, PATH_DATA, args)
    else:
        guide = bnn.vi(train_loader, args)
        utils.save(bnn, guide, PATH_DATA, args)
        args.test = True

    if args.test:
        test(bnn, guide, args=args)
        # test(bnn, guide, args=args, train_loader=train_loader)

    return bnn, guide, train_loader
