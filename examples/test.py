import torch
from matplotlib import pyplot as plt

from examples.noisy_sine_1.noisy_sine_1 import NoisySineDataset
from examples.mnist_old import make_mnist_dataloaders

def plot_bnn(model, guide, args, x_center, epsilon, bound_info=None, lin_bound_info=None):
    dataset = NoisySineDataset(dim=args.dim)
    X_train, y_train = dataset[:]

    X_test = torch.linspace(-2.0, 2.0, 1000).view(-1, 1).to(args.device)
    y_dist = model.predict_dist(guide, X_test, num_samples=800)
    # X_test, y_dist = X_test[..., 0].cpu(), y_dist[..., 0].cpu()

    y_mean = y_dist.mean(0)
    y_std = y_dist.std(0)

    plt.figure(figsize=(6.4 * 2, 4.8 * 2))


    plt.xlim(x_center-epsilon-0.05, x_center+epsilon+0.05)
    # plt.xlim(0.25, 0.55)
    # plt.xlim(-1., 1.)
    plt.ylim(-1., 2.)

    for i in range(y_dist.size(0)):
        plt.plot(X_test, y_dist[i], color='y', alpha=0.1, label='Uncertainty' if i == 0 else None)

    plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X_train[::4], y_train[::4], marker='+', color='b', label='Training data')
    plt.fill_between(X_test.ravel(), y_mean + y_std * 3, y_mean - y_std * 3, alpha=0.5, label='Uncertainty')

    if bound_info is not None:
        interval = torch.tensor([bound_info['x_center']-bound_info['epsilon'],
                                 bound_info['x_center']+bound_info['epsilon']])
        plt.plot(interval, [bound_info['lower_bound'], bound_info['lower_bound']],
                 '-k')
        plt.plot(interval, [bound_info['upper_bound'], bound_info['upper_bound']],
                 '-k')

    if lin_bound_info is not None:
        interval = torch.tensor([lin_bound_info['x_center']-lin_bound_info['epsilon'],
                                 lin_bound_info['x_center']+lin_bound_info['epsilon']])
        x = torch.linspace(interval[0], interval[1], 1000)[:,None]
        # upper_y = lin_bound_info['A_U_bound']*x + lin_bound_info['b_U_bound']
        # lower_y = lin_bound_info['A_L_bound']*x + lin_bound_info['b_L_bound']

        upper_y = torch.einsum('ij,kj->ki', lin_bound_info['A_U_bound'], x) + lin_bound_info['b_U_bound']
        lower_y = torch.einsum('ij,kj->ki', lin_bound_info['A_L_bound'], x) + lin_bound_info['b_L_bound']

        plt.plot(x, upper_y, '-g')
        plt.plot(x, lower_y, '-r')

    plt.title('Mean-field Gaussian VI BNN prediction')
    plt.legend()
    # plt.ylim(torch.min(lower_y) - 1., torch.max(upper_y)+ 1.)
    # plt.ylim(-0.5, 1.5)

    plt.show()

def test_mnist(model, guide):
    train_loaders, test_loaders = make_mnist_dataloaders()
    test_loader = test_loaders[0]

    print('Prediction when network is forced to predict')
    correct = 0
    total = 0
    for j, data in enumerate(test_loader):
        images, labels = data
        predicted = torch.mean(model.predict_dist(guide, images.view(-1,28*28), num_samples=80), dim=0)
        predicted_labels = torch.argmax(predicted, dim=1)

        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()
    print("accuracy: %d %%" % (100 * correct / total))


    print('hier')


@torch.no_grad()
def test(model, guide, args):
    plot_bnn(model, guide, args, x_center=0., epsilon=0.1)
    # test_mnist(model, guide)


