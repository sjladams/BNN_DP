import os
from examples.vi import PyroVI, PyroBatchLinear, PyroReLU, PyroSoftMax
from examples.utils import data_loaders
import matplotlib.pyplot as plt
import torch
import examples.utils as utils
from examples.bnn import BNN

######################################################################################
# An example of Variational Inference BNN for classification on the half-moons dataset
######################################################################################

PATH = os.getcwd() + "/examples/half_moons"
PATH_DATA = PATH + "/data"


# def get_net(args):
#     if args.layers == 1:
#         net = PyroVI(
#             PyroBatchLinear(2, args.width),
#             PyroReLU(),
#             PyroBatchLinear(args.width, 2),
#             PyroSoftMax(),
#             sigma=0.2
#         )
#     else:
#         net = PyroVI(
#             PyroBatchLinear(2, args.width),
#             PyroReLU(),
#             PyroBatchLinear(args.width, args.width),
#             PyroReLU(),
#             PyroBatchLinear(args.width, 2),
#             PyroSoftMax(),
#             sigma=0.2
#         )
#     return net


def plot(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(*x.reshape(x.shape[0], 2).T, c=torch.argmax(y, axis=1), cmap=plt.cm.Accent)
    plt.title("Generated half moons data")
    plt.show()


def test(net, test_loader, args):
    net.evaluate(test_loader, device=args.device, n_samples=10)

    x, y_true = next(iter(test_loader))
    x = x.to(args.device)
    # y_pred_dist = net.predict_dist(guide, x.flatten(start_dim=1, end_dim=3), num_samples=800)
    y_pred = net.forward(x, n_samples=100)
    # y_pred = y_pred_dist.mean(0)

    label_true = y_true.to(args.device).argmax(-1)
    label_pred = y_pred.argmax(-1)
    correct = (label_true == label_pred).sum().item()

    accuracy = 100 * correct / x.shape[0]
    print("Accuracy: %.2f%%" % (accuracy))

    plot(x, torch.round(y_pred))
    return accuracy


def get_bnn(args):
    # net = get_net(args)

    train_loader, test_loader, inp_shape, out_shape = \
        data_loaders(dataset_name=args.example, batch_size_train=128, n_inputs=args.n_inputs,
                     batch_size_test=128, shuffle=True)

    plot(*next(iter(train_loader)))

    # if file_check and not args.train:
    #     net, guide = utils.load(net, PATH_DATA, args)
    # else:
    #     guide = utils.train(net, train_loader, args)
    #     utils.save(net, guide, PATH_DATA, args)
    #     args.test = True

    bnn = BNN(dataset_name=args.example, hidden_size=args.width, activation=args.activation,
              architecture='fc{}'.format(args.layers), epochs=args.epochs, lr=args.lr, n_samples=None, warmup=None,
              input_shape=inp_shape, output_size=out_shape, trained_by=args.trained_by)

    if bnn.available(PATH_DATA) and not args.train:
        bnn.load(device=args.device, path=PATH_DATA)
    else:
        bnn.train(train_loader=train_loader, device=args.device, path=PATH_DATA)
        args.test = True

    if args.test:
        test(bnn, test_loader=train_loader, args=args)

    return bnn, test_loader