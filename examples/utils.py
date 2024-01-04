import torch
import math
import pyro
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

import random
from torchvision import datasets, transforms
from support import pickle_load


def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))


################
# data loaders #
################

def data_loaders(dataset_name, batch_size_train, batch_size_test, n_inputs=None, channels="first", shuffle=False):
    x_train, y_train, x_test, y_test, input_shape, num_classes = \
        load_dataset(dataset_name=dataset_name, n_inputs=n_inputs, channels=channels, shuffle=shuffle)

    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size_train,
                              shuffle=shuffle, worker_init_fn=np.random.seed(0),
                              num_workers=0)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size_test,
                             shuffle=shuffle, worker_init_fn=np.random.seed(0),
                             num_workers=0)

    return train_loader, test_loader, input_shape, num_classes

def plot_loss_accuracy(dict, path):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
    ax1.plot(dict['loss'])
    ax1.set_title("loss")
    ax2.plot(dict['accuracy'])
    ax2.set_title("accuracy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)


def load_half_moons(channels="first", n_samples=30000):
    x, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.1, random_state=0)
    x, y = (x.astype('float32'), y.astype('float32'))
    x = (x-np.min(x))/(np.max(x)-np.min(x))

    split_size = int(0.8 * len(x))
    x_train, y_train = x[:split_size], y[:split_size]
    x_test, y_test = x[split_size:], y[split_size:]

    # image-like representation for compatibility with old code
    n_channels = 1
    n_coords = 2
    if channels == "first":
        x_train = x_train.reshape(x_train.shape[0], n_channels, n_coords, 1)
        x_test = x_test.reshape(x_test.shape[0], n_channels, n_coords, 1)

    elif channels == "last":
        x_train = x_train.reshape(x_train.shape[0], 1, n_coords, n_channels)
        x_test = x_test.reshape(x_test.shape[0], 1, n_coords, n_channels)
    input_shape = x_train.shape[1:]

    # binary one hot encoding
    num_classes = 2

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return torch.from_numpy(x_train), y_train, torch.from_numpy(x_test), y_test, input_shape, num_classes


def load_fashion_mnist(channels, img_rows=28, img_cols=28):
    path = "{}/examples/fashion_mnist/data".format(os.getcwd())
    trainset = datasets.FashionMNIST(root=path, train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),]))
    x_train = trainset.data / 255
    x_train = x_train.type(torch.float32)
    y_train = to_categorical(trainset.targets, 10)

    # testset = datasets.FashionMNIST(root=path, train=False, download=True,
    #                            transform=transforms.Compose([transforms.ToTensor(),]))
    # x_test = testset.data / 255
    # x_test = x_test.type(torch.float32)
    # y_test = to_categorical(testset.targets, 10)

    x_test = torch.from_numpy(pickle_load(f"{path}/x_test_deepmind.npy"))
    y_test = to_categorical(pickle_load(f"{path}/y_test_deepmind.npy"), 10)

    if channels == "first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    elif channels == "last":
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = x_train.shape[1:]
    num_classes = 10
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes, dtype=torch.uint8)[y]


def load_mnist(channels, img_rows=28, img_cols=28):
    path = "{}/examples/mnist/data".format(os.getcwd())

    mnist = datasets.MNIST(path, train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),]))
    x_train = mnist.train_data / 255
    y_train = to_categorical(mnist.train_labels, 10)

    x_test = torch.from_numpy(pickle_load(f"{path}/x_test_deepmind.npy"))
    y_test = to_categorical(pickle_load(f"{path}/y_test_deepmind.npy"), 10)
    # x_test = mnist.test_data / 255
    # y_test = to_categorical(mnist.test_labels, 10)

    if channels == "first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    elif channels == "last":
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = x_train.shape[1:]
    num_classes = 10
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def noise(train_size, sigma):
    dist1 = torch.distributions.Normal(0., sigma)
    dist2 = torch.distributions.Normal(0., sigma)
    # if (train_size % 2) == 0:
    #     return torch.cat([dist1.sample((int(train_size / 2),)), dist2.sample((int(train_size / 2),))])
    # else:
    #     return torch.cat([dist1.sample((int(train_size / 2),)), dist2.sample((int(train_size / 2 + 1),))])
    train_size = (train_size,)
    return torch.cat([dist1.sample(train_size), dist2.sample(train_size)])


def f_1d(train_size):
    sigma = 0.2
    X = torch.linspace(-1.0, 1.0, train_size).repeat(2).view(-1, 1)
    # return X, torch.sin(2 * np.pi * X) + noise(train_size, sigma).view(-1, 1) * (1-torch.sin(2 * np.pi * X))
    return X, torch.sin(2 * np.pi * X) + noise(train_size, sigma).view(-1, 1) * \
              (1.5-torch.abs(torch.sin(2 * np.pi * X)))

# test = torch.sin(2 * np.pi * X) + noise(train_size, sigma).view(-1, 1)
# # test = torch.sin(2 * np.pi * X) + noise(train_size, sigma).view(-1, 1) *(1+ (1-torch.abs(torch.sin(2 * np.pi * X))))
# plt.plot(test)
# plt.show()

def f_2d_func(X):
    return 0.5 * torch.sin(2 * np.pi * X[:, 0]) + 0.5 * torch.sin(2 * np.pi * X[:, 1])


def f_2d_func_noise(X):
    sigma = 0.4 # 0.2
    return f_2d_func(X) + noise(int(X.shape[0] / 2), sigma)


def f_2d(train_size):
    x_space = torch.linspace(-1.0, 1.0, int(math.sqrt(train_size)))
    X = torch.cartesian_prod(x_space, x_space).repeat(2, 1)
    y = f_2d_func_noise(X)
    return X, y.view(-1, 1)

def load_noisy_sine_1():
    dataset_size = 2**12

    x_train, y_train = f_1d(dataset_size)
    x_test, y_test = f_1d(dataset_size)

    input_shape = x_train.shape[1]
    output_shape = y_train.shape[1]
    return x_train, y_train, x_test, y_test, input_shape, output_shape

def load_noisy_sine_2():
    dataset_size = 2**12

    x_train, y_train = f_2d(dataset_size)
    x_test, y_test = f_2d(dataset_size)

    input_shape = x_train.shape[1]
    output_shape = y_train.shape[1]
    return x_train, y_train, x_test, y_test, input_shape, output_shape


def labels_to_onehot(integer_labels, n_classes=None):
    n_rows = len(integer_labels)
    n_cols = n_classes if n_classes else integer_labels.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot


def onehot_to_labels(y):
    if type(y) is np.ndarray:
        return np.argmax(y, axis=1)
    elif type(y) is torch.Tensor:
        return torch.max(y, 1)[1]

def load_cifar(channels, img_rows=32, img_cols=32):
    path = "{}/examples/cifar/data".format(os.getcwd())
    trainset = datasets.CIFAR10(root=path, train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),]))
    x_train = torch.from_numpy(trainset.data) / 255
    y_train = to_categorical(torch.tensor(trainset.targets), 10)

    testset = datasets.CIFAR10(root=path, train=False, download=True,
                               transform=transforms.Compose([transforms.ToTensor(),]))
    x_test = torch.from_numpy(testset.data) / 255
    y_test = to_categorical(torch.tensor(testset.targets), 10)

    if channels == "first":
        x_train = torch.moveaxis(x_train, 3, 1)
        x_test = torch.moveaxis(x_test, 3, 1)

    input_shape = x_train.shape[1:]
    num_classes = 10
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def normalize(x: torch.tensor):
    x -= torch.min(x)
    x /= torch.max(x)
    return x


def load_concrete():
    # \TODO implement StandardScaler() method
    raise NotImplementedError

    file_path = '{}/{}'.format(os.getcwd(), 'examples/concrete/data/concrete_database.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        resp = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
        df = pd.read_excel(BytesIO(resp.read()))
        df.to_csv(file_path, index=False)

    train_points = int(4 * df.shape[0] / 5)
    test_points = int(df.shape[0] / 1)

    var0 = normalize(torch.tensor(df[df.columns[0]],  dtype=torch.float32))
    var1 = normalize(torch.tensor(df[df.columns[1]],  dtype=torch.float32))
    var2 = normalize(torch.tensor(df[df.columns[2]],  dtype=torch.float32))
    var3 = normalize(torch.tensor(df[df.columns[3]],  dtype=torch.float32))
    var4 = normalize(torch.tensor(df[df.columns[4]],  dtype=torch.float32))
    var5 = normalize(torch.tensor(df[df.columns[5]],  dtype=torch.float32))
    var6 = normalize(torch.tensor(df[df.columns[6]],  dtype=torch.float32))

    var7 = normalize(torch.tensor(df[df.columns[7]],  dtype=torch.float32))
    out = normalize(torch.tensor(df[df.columns[8]],  dtype=torch.float32))

    x = torch.vstack((var0, var1, var2, var3, var4, var5, var6, var7)).moveaxis(0,1)
    x_train = x[:train_points, :]
    y_train = out[:train_points]
    x_test = x[-test_points:, :]
    y_test = out[-test_points:]

    input_shape = x_train.shape[1]
    num_classes = 1
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def load_powerplant():
    file_path = '{}/{}'.format(os.getcwd(), 'examples/powerplant/data/powerplant_database.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        resp = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip')
        myzip = ZipFile(BytesIO(resp.read()))
        myzip.namelist()
        df = pd.read_excel(BytesIO(myzip.read('CCPP/Folds5x2_pp.xlsx')))
        df.to_csv(file_path, index=False)

    train_points = 8611
    test_points = 957

    AT = torch.tensor(df['AT'], dtype=torch.float32)
    V = torch.tensor(df['V'], dtype=torch.float32)
    AP = torch.tensor(df['AP'], dtype=torch.float32)
    RH = torch.tensor(df['RH'], dtype=torch.float32)
    PE = torch.tensor(df['PE'], dtype=torch.float32)

    x = torch.vstack((AT, V, AP, RH)).moveaxis(0,1)
    x_train = x[:train_points, :]
    y_train = PE[:train_points][:,None]
    x_test = x[-test_points:, :]
    y_test = PE[-test_points:][:, None]

    x_scalar = StandardScaler()
    x_scalar.fit(x_train)

    y_scalar = StandardScaler()
    y_scalar.fit(y_train)

    x_train, x_test = torch.from_numpy(x_scalar.transform(x_train)), torch.from_numpy(x_scalar.transform(x_test))
    x_train, x_test = x_train.to(torch.float32), x_test.to(torch.float32)
    y_train, y_test = torch.from_numpy(y_scalar.transform(y_train)), torch.from_numpy(y_scalar.transform(y_test))
    y_train, y_test = y_train.to(torch.float32), y_test.to(torch.float32)

    input_shape = x_train.shape[1]
    num_classes = 1
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def load_kin8nm():
    file_path = '{}/{}'.format(os.getcwd(), 'examples/kin8nm/data/data.txt')
    if os.path.exists(file_path):
        data = np.loadtxt(file_path)
    else:
        raise NotImplementedError('data file not available')

    train_points = int(data.shape[0] * (7/8))
    test_points = int(data.shape[0] * (1/8))

    x = torch.tensor(data[:, :-1], dtype=torch.float32)
    y = torch.tensor(data[:, -1], dtype=torch.float32)

    x_train = x[:train_points, :]
    y_train = y[:train_points][:,None]
    x_test = x[-test_points:, :]
    y_test = y[-test_points:][:, None]

    x_scalar = StandardScaler()
    x_scalar.fit(x_train)

    y_scalar = StandardScaler()
    y_scalar.fit(y_train)

    x_train, x_test = torch.from_numpy(x_scalar.transform(x_train)), torch.from_numpy(x_scalar.transform(x_test))
    x_train, x_test = x_train.to(torch.float32), x_test.to(torch.float32)
    y_train, y_test = torch.from_numpy(y_scalar.transform(y_train)), torch.from_numpy(y_scalar.transform(y_test))
    y_train, y_test = y_train.to(torch.float32), y_test.to(torch.float32)

    input_shape = x_train.shape[1]
    num_classes = 1
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def load_dataset(dataset_name, n_inputs=None, channels="first", shuffle=False):
    if dataset_name == "mnist":
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_mnist(channels)
    elif dataset_name == "cifar":
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_cifar(channels)
    elif dataset_name == "fashion_mnist":
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_fashion_mnist(channels)
    elif dataset_name == "half_moons":
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_half_moons()
    elif dataset_name == 'noisy_sine_1':
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_noisy_sine_1()
    elif dataset_name == 'noisy_sine_2':
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_noisy_sine_2()
    elif dataset_name == 'concrete':
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_concrete()
    elif dataset_name == 'powerplant':
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_powerplant()
    elif dataset_name == 'kin8nm':
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_kin8nm()
    else:
        raise AssertionError("\nDataset not available.")

    if n_inputs:
        x_train, y_train, x_test, y_test = (x_train[:n_inputs], y_train[:n_inputs],
                                            x_test[:n_inputs], y_test[:n_inputs])

    print('x_train shape =', x_train.shape, '\nx_test shape =', x_test.shape)
    print('y_train shape =', y_train.shape, '\ny_test shape =', y_test.shape)

    if shuffle is True:
        random.seed(0)
        idxs = torch.randperm(len(x_train))
        x_train, y_train = (x_train[idxs], y_train[idxs])
        idxs = torch.randperm(len(x_test))
        x_test, y_test = (x_test[idxs], y_test[idxs])

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def plot_grid_images(images, labels):
    plt.figure(figsize=(8, 8))
    rows = cols = min(int(np.ceil(np.sqrt(len(images)))), 10)
    for i in range(0, cols * rows):
        if i >= len(images):
            break
        plt.subplot(rows, cols, i+1)

        image = torch.moveaxis(images[i].detach().cpu(), 0, 2)
        plt.imshow(image)
        plt.title('tag: {}'.format(torch.argmax(labels[i])))

    plt.show()


def file_name(args):
    return f"{args.example}_bnn_hid={args.width}_act={args.activation}_arch=fc{args.layers}_{args.trained_by}"


def save(model, guide, path_data, args):
    torch.save({"model": model.state_dict(), "guide": guide}, f"{path_data}/{file_name(args)}.pt")
    pyro.get_param_store().save(f"{path_data}/params_{file_name(args)}.pt")


def load(net, path_data, args):
    saved_model_dict = torch.load(f"{path_data}/{file_name(args)}.pt")
    net.load_state_dict(saved_model_dict['model'])
    guide = saved_model_dict['guide']
    pyro.get_param_store().load(f"{path_data}/params_{file_name(args)}.pt")
    return net, guide


def available(path_data, args):
    return os.path.exists(f'{path_data}/{file_name(args)}.pt')


def train(model, train_loader, args):
    guide = model.vi(train_loader, args)
    return guide