import torch
from tqdm import tqdm
import numpy as np
from sympy import *
import pickle
import pyro
from argparse import ArgumentParser
import cProfile, pstats, io


def gradient_descent(interval, func, grad_func, n_iter=1000, learn_rate=0.02, tol=1e-4, starting_point=None):
    if starting_point is None:
        vector = torch.mean(interval, dim=0)
    else:
        vector = starting_point

    conv_meas = torch.zeros(n_iter)

    for i in tqdm(range(n_iter)):
        grad = grad_func(vector)
        new_vector = torch.clip(vector-learn_rate * grad, interval[0], interval[1])

        conv_meas[i] = torch.linalg.norm(new_vector - vector)

        if conv_meas[i] < tol:
            break

        if torch.any(torch.isnan(new_vector)):
            raise ValueError('encountered nan value in {} procedure of {}'.format('gradient_descent', func.__name__))

        vector = new_vector

    # add extra check:
    val2check = func(vector)
    interval_vals = torch.tensor([func(interval[0]), func(interval[1])])
    val_ref = torch.min(interval_vals)
    if val_ref < val2check:
        return interval[torch.argmin(interval_vals)], val_ref
    else:
        return vector, val2check


def get_vertices(l,u):
    dim = l.shape[0]
    start = 0
    needed = dim+1
    interval = torch.stack((l,u), dim=1)
    vertices = cartesian_product(interval, needed=needed, start=start)
    data = torch.cat((vertices, torch.ones((vertices.shape[0],1))), dim=1)
    matrix = Matrix(data)
    row_space = torch.squeeze(torch.from_numpy(np.array(matrix.rowspace()).astype(np.float32)))

    start = 0
    while row_space.shape[0] < dim+1:
        start += dim+1
        needed = 1
        vertices = torch.cat((vertices, cartesian_product(interval, needed=needed, start=start)))
        data = torch.cat((vertices, torch.ones((vertices.shape[0],1))), dim=1)
        matrix = Matrix(data)
        row_space = torch.squeeze(torch.from_numpy(np.array(matrix.rowspace()).astype(np.float32)))

    dummy_col = 1/row_space[:,-1]
    dummy = dummy_col[:,None].repeat(1,dim+1)
    to_return = torch.einsum('ij,ij->ij', row_space, dummy)
    return to_return[:,:-1]


def cartesian_product(arrays, needed=None, start=0):
    if needed is not None:
        dim = arrays.shape[0]
        if needed == dim+1:
            # to_return = arrays[:,0][None].repeat(dim+1, 1)
            # to_return [1:,:] += torch.diag(arrays[:, 1] - arrays[:, 0])
            to_return = arrays[:,1][None].repeat(dim+1, 1)
            to_return [1:,:] += torch.diag(arrays[:, 0] - arrays[:, 1])
            return to_return
        else:
            # old method
            n = int(torch.ceil(torch.log2(torch.tensor(needed))))
            arrays_stacked = arrays.repeat(4,1)
            arrays_use = arrays_stacked[start:n+start]
            vertices = torch.cartesian_prod(*arrays_use)[0:needed]
            if vertices.dim() == 1:
                vertices = vertices[:, None]
            constants = arrays_stacked[n+start:dim+start,0][None,:].repeat(needed,1)
            return torch.cat((vertices, constants), dim=1)
    else:
        return torch.cartesian_prod(*arrays)


def cartesian_product_inner(arrays, out=None):
    arrays = [torch.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = torch.prod(torch.tensor([x.size() for x in arrays]))
    if out is None:
        out = torch.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / torch.tensor(arrays[0].size()))
    out[:, 0] = arrays[0].repeat(m)
    if arrays[1:]:
        cartesian_product_inner(arrays[1:], out=out[0:m, 1:])
        for j in range(1, torch.tensor(arrays[0].size())):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def cartesian_product_inner_np(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian_product_inner_np(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def get_eigvals(H):
    return torch.linalg.eigvals(H)


def is_neg_sem_def(H):
    return torch.all(torch.round(torch.linalg.eigvals(H).real, decimals=5) <=0.)


def is_pos_sem_def(H):
    return torch.all(torch.round(torch.linalg.eigvals(H).real, decimals=5) >=0.)


def is_pos_def(H):
    torch.linalg.cholesky(H)
    return True


def check4inf(toCheck):
    return any([torch.any(torch.isinf(elem)) for elem in toCheck])


def check4nan(toCheck):
    return any([torch.any(torch.isnan(elem)) for elem in toCheck])


def prop_lb(A_L_in, b_L_in, A_U_in, b_U_in, A_L_out, b_L_out, A_U_out=None, b_U_out=None):
    if A_U_out is None and b_U_out is None:
        A_L, b_L, A_U, b_U = onesided_prop_lb(A_L_in, b_L_in, A_U_in, b_U_in, A_L_out, b_L_out)
        return {'A_L': A_L, 'b_L': b_L, 'A_U': A_U, 'b_U': b_U}
    else:
        A_L, b_L, _, _ = onesided_prop_lb(A_L_in, b_L_in, A_U_in, b_U_in, A_L_out, b_L_out)
        _, _, A_U, b_U = onesided_prop_lb(A_L_in, b_L_in, A_U_in, b_U_in, A_U_out, b_U_out)
        return {'A_L': A_L, 'b_L': b_L, 'A_U': A_U, 'b_U': b_U}


def onesided_prop_lb(A_L_in, b_L_in, A_U_in, b_U_in, A_out, b_out):
    mask = A_out >= 0.

    A_L_bar = torch.einsum('on, ni ->oni', mask, A_L_in) + torch.einsum('on, ni ->oni', ~mask, A_U_in)
    A_L = torch.einsum('on, oni -> oi', A_out, A_L_bar)

    A_U_bar = torch.einsum('on, ni ->oni', mask, A_U_in) + torch.einsum('on, ni ->oni', ~mask, A_L_in)
    A_U = torch.einsum('on, oni -> oi', A_out, A_U_bar)

    b_L_bar = torch.einsum('on, n ->on', mask, b_L_in) + torch.einsum('on, n ->on', ~mask, b_U_in)
    b_L = torch.einsum('on, on -> o', A_out, b_L_bar) + b_out

    b_U_bar = torch.einsum('on, n ->on', mask, b_U_in) + torch.einsum('on, n ->on', ~mask, b_L_in)
    b_U = torch.einsum('on, on -> o', A_out, b_U_bar) + b_out

    return A_L, b_L, A_U, b_U


def pickle_dump(obj, tag):
    pickle_out = open("{}.pickle".format(tag), "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def pickle_load(tag):
    if not ".npy" in tag or ".pickle" in tag:
        tag = f"{tag}.pickle"
    pickle_in = open(tag, "rb")
    if "npy" in tag:
        to_return = np.load(pickle_in)
    else:
        to_return = pickle.load(pickle_in)
    pickle_in.close()
    return to_return


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--debug', type=eval, default=False, help='debug')
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cpu',
                        help='Select device for tensor operations')

    ## Network Settings
    parser.add_argument('--example', choices=['mnist', 'noisy_sine_1', 'half_moons', 'cifar', 'fashion_mnist',
                                              'noisy_sine_2', 'concrete', 'powerplant', 'kin8nm'], type=str,
                        default="noisy_sine_1", help='Example')
    parser.add_argument('--layers', choices=[1, 2, 3, 4], type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--activation', choices=["relu", "leaky", "sigm", "tanh"], type=str, default="relu")
    parser.add_argument('--width', choices=[48, 64, 100, 128, 256, 512, 1024, 2048], type=int, default=256,
                        help='Width hidden layer')

    # Training
    parser.add_argument("--sigma", default=0.01, type=float, help="observation noise")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--prior_weight_scale", default=1., type=float, help="Scale prior")
    parser.add_argument("--prior_bias_scale", default=1., type=float, help="Scale prior")
    parser.add_argument("--epochs", default=500, type=int, help="number of epochs")
    parser.add_argument("--n_inputs", default=2 ** 12, type=int, help="number of input points")
    parser.add_argument("--train", default=False, type=eval, help="train or load saved model")
    parser.add_argument("--test", default=False, type=eval, help="evaluate on test data")
    parser.add_argument("--trained_by", default='own', choices=['own', 'oxford', 'vogn'], type=str,
                        help="choose dir for loading the BNN: DATA, TESTS")

    ## Verification Settings
    parser.add_argument('--fixed_epsilon', default=True, type=eval,
                        help="find robustness accuracy for fixed epsilon, or find maximal robust epsilon")
    parser.add_argument('--epsilon', choices=[0.025, 0.01, 0.001], type=float, default=0.01,
                        help='fixed epsilon if enabled, else initialization')
    parser.add_argument('--epsilon_step_size', default=0.1, type=float,
                        help='step size epsilon tuning if not fixed epsilon')

    # Classification
    parser.add_argument('--nr_images', default=100, type=int, help="if classification, number of images to verify")

    return parser.parse_args()


def get_param(weight_bias, loc_scale, layer):
    options = list(pyro.get_param_store()._params.keys())
    mask = torch.tensor([all(tag in option for tag in [weight_bias, loc_scale, str(layer)]) for option in options])
    if sum(mask) != 1.:
        raise ValueError('no or more than one parameter matches keys')
    param_name = options[torch.where(mask)[0]]
    return pyro.param(param_name).data


def profile(fnc):
    """A decorator fhat uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner