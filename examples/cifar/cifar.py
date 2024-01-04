import os
from examples.vi import PyroVI, PyroBatchLinear, PyroReLU, PyroSoftMax
from examples.utils import data_loaders, plot_grid_images
from examples.bnn import BNN

#################################################################################
# An example of Variational Inference BNN for classification on the cifar dataset
#################################################################################

PATH = os.getcwd() + "/examples/cifar"
PATH_DATA = PATH + "/data"


# def get_net(args):
#     if args.layers == 1:
#         net = PyroVI(
#             PyroBatchLinear(32*32, args.width),
#             PyroReLU(),
#             PyroBatchLinear(args.width, 10),
#             PyroSoftMax(),
#             sigma=0.2
#         )
#     else:
#         net = PyroVI(
#             PyroBatchLinear(32*32, args.width),
#             PyroReLU(),
#             PyroBatchLinear(args.width, args.width),
#             PyroReLU(),
#             PyroBatchLinear(args.width, 10),
#             PyroSoftMax(),
#             sigma=0.2
#         )
#     return net

def test(net, test_loader, args):
    net.evaluate(test_loader, device=args.device, n_samples=10)

def get_bnn(args):
    # net = get_net(args)
    train_loader, test_loader, inp_shape, out_shape = \
        data_loaders(dataset_name=args.example, batch_size_train=128, n_inputs=args.n_inputs, batch_size_test=128,
                           shuffle=False)

    plot_grid_images(next(iter(train_loader))[0][0:3], next(iter(train_loader))[1][0:3])

    bnn = BNN(dataset_name=args.example, hidden_size=args.width, activation=args.activation,
              architecture='fc{}'.format(args.layers), epochs=args.epochs, lr=args.lr, n_samples=None, warmup=None,
              input_shape=inp_shape, output_size=out_shape, trained_by=args.trained_by)

    if bnn.available(PATH_DATA) and not args.train:
        bnn.load(device=args.device, path=PATH_DATA)
    else:
        bnn.train(train_loader=train_loader, device=args.device, path=PATH_DATA)
        args.test = True

    if args.test:
        test(bnn, test_loader=test_loader, args=args)

    return bnn, test_loader