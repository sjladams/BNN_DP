from matplotlib import pyplot as plt
from support import *
from functools import partial
import sympy
from copy import copy

def func_param_erf_term(loc, scale2, loc_cor):
    scale = scale2**0.5
    return 1 - torch.erf(-(loc-loc_cor) / (scale * np.sqrt(2)))


def func_param_exp_term(loc, scale2, loc_cor):
    return torch.exp(-(loc - loc_cor)**2 / (2 * scale2))


def func_param(loc, scale, loc_cor): # \TODO scale2 -> scale
    return 0.5 * (loc - loc_cor) * func_param_erf_term(loc_cor=loc_cor, loc=loc, scale2=scale**2) + \
           scale / np.sqrt(2 * np.pi) * func_param_exp_term(loc_cor=loc_cor, loc=loc, scale2=scale**2)


def grad_param(loc, scale, loc_cor): # \TODO scale2 -> scale
    return torch.tensor([0.5 * (1 - torch.erf(-(loc-loc_cor) / (scale * np.sqrt(2)))),
                        1 / (np.sqrt(2*np.pi)) * torch.exp(-(loc - loc_cor)**2 / (2 * scale**2))])


def func_prob_param(loc, scale2, loc_cor):
    scale = scale2**0.5
    return loc_cor * 0.5 * (1 - torch.erf(-(loc - loc_cor) / scale * (1 / np.sqrt(2))))


## LINEAR BOUNDS ---------------------------------------------------------------------------------------------------- ##
class LinearBoundsMult: # \TODO change name to LinearBounds
    def __init__(self, L, U, **kwargs):
        if not all([L.dim() == 1, U.dim() == 1]):
            raise Warning('L and U should be tensors of shape 1')

        self.L_init = L
        self.U_init = U
        self.L, self.U = self.L_init, self.U_init
        if all(['A_L' in kwargs, 'b_L' in kwargs, 'A_U' in kwargs, 'b_U' in kwargs]):
            self.init(**kwargs)
            self.lbs = {i: LinearBounds(L,U, a_l=kwargs['A_L'][i, :], b_l=kwargs['b_L'][i][None],
                                        a_u=kwargs['A_U'][i, :], b_u=kwargs['b_U'][i][None])
                        for i in range(kwargs['A_L'].shape[0])}
        else:
            self.a_l, self.b_l, self.a_u, self.b_u = None, None, None, None
            self.lbs = {i: LinearBounds(l,u) for i, (l,u) in enumerate(zip(L,U))}

    def init(self, A_L: torch.tensor, b_L: torch.tensor, A_U: torch.tensor, b_U: torch.tensor):
        if not all([A_L.dim() == 2, b_L.dim() == 1, A_U.dim() == 2, b_U.dim() == 1]):
            raise Warning('A should be a tensor of shape 2, and b should be a tensor of shape 1')
        self.A_L, self.b_L, self.A_U, self.b_U = A_L, b_L, A_U, b_U

    def grab_dim(self, dim):
        return self.lbs[dim]

    def backprop(self, A_L_in, b_L_in, A_U_in=None, b_U_in=None):
        if A_U_in is None or b_U_in is None:
            lbs_dict = prop_lb(A_L_in=A_L_in, b_L_in=b_L_in, A_U_in=A_L_in, b_U_in=b_L_in,
                    A_L_out=self.A_L, b_L_out=self.b_L, A_U_out=self.A_U, b_U_out=self.b_U)
        else:
            lbs_dict = prop_lb(A_L_in=A_L_in, b_L_in=b_L_in, A_U_in=A_U_in, b_U_in=b_U_in,
                    A_L_out=self.A_L, b_L_out=self.b_L, A_U_out=self.A_U, b_U_out=self.b_U)
        return lbs_dict['A_L'], lbs_dict['b_L'], lbs_dict['A_U'], lbs_dict['b_U']




class LinearBounds: # \TODO change name to LinearBound, let l,u be tensors of shape 0, and b_l and b_u
    def __init__(self, l, u, **kwargs):
        if not all([l.dim() == 1, u.dim() == 1]):
            raise Warning('lower and upper bound should be tensors of shape 1')

        self.l_init = l
        self.u_init = u
        self.l, self.u = self.l_init, self.u_init
        if all(['a_l' in kwargs, 'b_l' in kwargs, 'a_u' in kwargs, 'b_u' in kwargs]):
            self.init(**kwargs)
        else:
            self.a_l, self.b_l, self.a_u, self.b_u = None, None, None, None


    def init(self, a_l, b_l, a_u, b_u):
        if not all([a_l.dim() == 1, b_l.dim() == 1, a_u.dim() == 1, b_u.dim() == 1]):
            raise Warning('both a and b should be tensors of shape 1')
        self.a_l, self.b_l, self.a_u, self.b_u = a_l, b_l, a_u, b_u
        self.set_l_u()

    def lin_u(self, x):
        if x.dim() == 1:
            return torch.einsum('i,i', self.a_u, x) + self.b_u
        elif x.dim() == 2:
            return torch.einsum('i,ki->k', self.a_u, x) + self.b_u

    def lin_l(self, x):
        if x.dim() == 1:
            return torch.einsum('i,i', self.a_l, x) + self.b_l
        elif x.dim() == 2:
            return torch.einsum('i,ki->k', self.a_l, x) + self.b_l

    def set_l_u(self):
        inp_int = torch.cat((self.l_init[None], self.u_init[None]))
        _, self.l, _, self.u = lb2ib(inp_int, self.a_l, self.b_l, self.a_u, self.b_u, sat=False)

        if self.u.dim() == 0 or self.l.dim() == 0:
            raise ValueError('{} cdf.l and cdf.u should be tensors of dim 1')

    def forward(self, a_l, b_l, a_u, b_u):
        if len(self.a_l.shape) > 1 or len(a_l.shape) > 1:
            raise Warning('only handle 1D tensors')

        new_lb = prop_lb(self.a_l[None], self.b_l, self.a_u[None], self.b_u, a_l[None], b_l, a_u[None], b_u)
        self.a_l, self.b_l, self.a_u, self.b_u = new_lb['A_L'].flatten(), new_lb['b_L'], new_lb['A_U'].flatten(), new_lb['b_U']

        # _ = lb_zero_check(torch.stack((self.l_init[None], self.u_init[None])), self.a_l[None], self.b_l, self.a_u[None], self.b_u)
        _ = lb_zero_check(torch.stack((self.l_init, self.u_init)), self.a_l, self.b_l, self.a_u, self.b_u)

        self.set_l_u()


def fit_1dlin_points(l, u, val_l, val_u):
    a = torch.divide(val_u-val_l, u-l)
    if torch.isnan(a) or torch.isinf(a):
        a = torch.tensor([0.])

    b = val_u - a*u
    return a, b


def fit_1Dlin_func(l, u, func):
    if l.shape[0] > 1:
        raise ValueError('{} only accepts 1D inputs'.format('fit_1Dlin_func'))
    return fit_1dlin_points(l, u, func(l), func(u))


def tang(*args, func, grad_func):
    a = grad_func(*args)
    b = func(*args)-torch.einsum('i,i->', a, torch.cat(args))
    return a, b


## QUAD ------------------------------------------------------------------------------------------------------------- ##
def grad_quad(x, Sigma2):
    """
    :param x: point
    :return:
    """
    return 2*torch.einsum('ij,j->i', Sigma2, x)


def quad(x, Sigma2):
    """
    :param x: tensor of points or point
    :return:
    """
    if x.dim() == 1:
        return torch.einsum('i,ij,j->', x, Sigma2, x)
    elif x.dim() == 2:
        return torch.einsum('ki,ij,kj->k', x, Sigma2, x)
    else:
        raise ValueError('quad does not support >2 dim tensors')


def bound_quad_l(inp_int, Sigma2, use_ibp=True):
    """
    Lower bounding of (pos-def) multi-variate quadratic term. Strictly increasing, so just check if glob min, origin,
    is in inp_int, else check bound interval.
    :param inp_int:
    :param Sigma2: POS DEF matrix
    :param use_loc_min:
    :return:
    """
    if torch.any(Sigma2 - torch.diag(torch.diagonal(Sigma2)) != 0):
        U, S, V = torch.linalg.svd(Sigma2)
        Sigma2_z = torch.diag(S)

        inp_int_z = bound_rotated_box(inp_int, V)

        a_l_z, b_l_z = bound_quad_l(inp_int_z, Sigma2_z, use_ibp=use_ibp)

        lbs_x = prop_lb(A_L_in=V, b_L_in=torch.zeros(V.shape[0]), A_U_in=V, b_U_in=torch.zeros(V.shape[0]),
                        A_L_out=a_l_z[None], b_L_out=b_l_z)
        a_l, b_l = lbs_x['A_L'].flatten(), lbs_x['b_L'].flatten()
        return a_l, b_l
    else:
        dim = inp_int.shape[1]
        quad_dum = partial(quad, Sigma2=Sigma2)
        quad_grad_dum = partial(grad_quad, Sigma2=Sigma2)

        if torch.all(inp_int[0] <= 0.) and torch.all(inp_int[1] >= 0.):
            x = torch.zeros(inp_int[0].shape)
        else:
            x = inp_int[torch.argmin(torch.tensor([quad_dum(inp_int[0]), quad_dum(inp_int[1])]))]

        # x, val = gradient_descent(inp_int, quad_dum, quad_grad_dum)
        if use_ibp:
            return torch.zeros(dim), quad_dum(x)[None]
        else:
            return tang(x, func=quad_dum, grad_func=quad_grad_dum)


def get_hyperplane(interval, func=None, add_vertice=torch.tensor([]), vertices=None, vals=None):
    if all([torch.all(elem == vals) for elem in vals]):
        return torch.zeros(2), max(vals)[None]

    dim = interval.shape[1]

    if vertices is None or vals is None:
        vertices = cartesian_product(torch.moveaxis(interval, 0, 1), needed=dim+1, start=0)

        if add_vertice not in vertices:
            vertices = torch.cat((vertices[0:-1], add_vertice[None]), dim=0)

        vals = func(vertices)
    else:
        vertices = vertices[:dim+1, :]
        vals = vals[:dim+1]

    matrix = torch.cat((vertices, torch.ones((dim+1, 1)), vals[:, None]), dim=1)

    params = sympy.Matrix(matrix.detach().numpy()).nullspace()

    params = torch.squeeze(torch.from_numpy(np.array(params).astype(np.float32)))
    if len(params.shape) > 1 or params[-1] < 1e-8:
        # print('Failed to fit hyperplane due to non-singularity') [torch.all(elem == vals_tensor) for elem in vals_tensor]
        return torch.zeros(2), max(vals)[None]
    else:
        params = params[:-1]/-params[-1]
        return params[:-1], params[-1][None]

def bound_quad_u(inp_int, Sigma2, use_ibp_u=True):
    """
    Upper bounding of (pos-def) multi-variate quadratic term
    :param inp_int: possibly origin covering intervals
    :param Sigma2: POS DEF matrix
    :param use_ibp:
    :return:
    """
    if not is_pos_def(Sigma2):
        raise Warning('bound_quad_u only able to find max of pos-def quadratic term')
    elif torch.any(Sigma2 - torch.diag(torch.diagonal(Sigma2)) != 0):
        U, S, V = torch.linalg.svd(Sigma2)
        Sigma2_z = torch.diag(S)

        inp_int_z = bound_rotated_box(inp_int, V)

        a_u_z, b_u_z = bound_quad_u(inp_int_z, Sigma2_z, use_ibp_u=use_ibp_u)

        # U.moveaxis(0, 1) @ Sigma2 @ U
        # z = Vx
        # x = Uz
        # S = V @ Sigma2 @ U = inverse(U) @ Sigma2 @ U

        lbs_x = prop_lb(A_L_in=V, b_L_in=torch.zeros(V.shape[0]), A_U_in=V, b_U_in=torch.zeros(V.shape[0]),
                A_L_out=a_u_z[None], b_L_out=b_u_z)
        a_u, b_u = lbs_x['A_U'].flatten(), lbs_x['b_U'].flatten()
        return a_u, b_u
    elif use_ibp_u:
        dim = inp_int.shape[1]
        quad_dum = partial(quad, Sigma2=Sigma2)
        return torch.zeros(dim), torch.max(quad_dum(inp_int[0]), quad_dum(inp_int[1]))[None]
    else:
        # https://sharmaeklavya2.github.io/theoremdep/nodes/linear-algebra/matrices/bounding-quadratic-form-using-eigenvalues.html
        a_u = torch.nan_to_num(torch.divide(torch.einsum('i,i->i', torch.diagonal(Sigma2), inp_int[1]**2) -
                                            torch.einsum('i,i->i', torch.diagonal(Sigma2), inp_int[0]**2),
                                            inp_int[1]-inp_int[0]))
        b_u = torch.nan_to_num(torch.einsum('i,ij,j->', inp_int[1], Sigma2, inp_int[1]) - \
                               torch.einsum('i,i->', a_u, inp_int[1]))[None]
        return a_u, b_u

        # U, S, _ = torch.linalg.svd(Sigma2)      # Sigma2_tilde = torch.einsum('ij, jk, nk->in', U, torch.diag(S), U)
        # T = torch.einsum('ij,jk->ik', U, torch.diag(S**-0.5)) # S_tilde = torch.einsum('ji,jk,kn->in', U, Sigma2, U)
        # T_inv = torch.einsum('ij,kj->ik', torch.diag(S**0.5), U) # I = torch.einsum('ij, jk, nk ->in ', T_inv, Sigma2, T)
        # # torch.einsum('ij, jk, kn, nm ->im', torch.diag(S**-0.5), torch.transpose(U,0,1), U, torch.diag(S**-0.5))
        # # I = torch.einsum('ji,jk,kn->in', T, Sigma2, T)
        # # torch.einsum('ij,jk->ik', torch.transpose(U,0,1), U)
        # # inp_int_trans = torch.einsum('ij,kj->ki', T_inv, inp_int) # \Changed from T_inv -> T
        # inp_int_trans = torch.einsum('ij,kj->ki', T, inp_int) # \Changed from T_inv -> T
        # # center = torch.mean(inp_int_trans, dim=0)
        # # width = 0.5*torch.linalg.norm(inp_int_trans[1] - inp_int_trans[0], ord=1)
        # # inp_int_trans_overapprox = torch.stack((center-width, center+width), dim=0)
        # # a_u_trans = torch.nan_to_num(torch.divide(inp_int_trans_overapprox[1]**2 - inp_int_trans_overapprox[0]**2, \
        # #                          inp_int_trans_overapprox[1] - inp_int_trans_overapprox[0]))
        # # b_u_trans = torch.nan_to_num(torch.einsum('i->', inp_int_trans_overapprox[0]**2) - \
        # #             torch.einsum('i,i->', a_u_trans, inp_int_trans_overapprox[0]))
        # a_u_trans = torch.nan_to_num(torch.divide(inp_int_trans[1]**2 - inp_int_trans[0]**2, \
        #                                           inp_int_trans[1] - inp_int_trans[0]))
        # b_u_trans = torch.nan_to_num(torch.einsum('i->', inp_int_trans[0]**2) - \
        #                              torch.einsum('i,i->', a_u_trans, inp_int_trans[0]))
        # # return torch.einsum('ij,j->i', T_inv, a_u_trans), b_u_trans[None]
        # return torch.einsum('i, ij->j', a_u_trans, T), b_u_trans[None]


def bound_quad(Sigma2_w, sigma2_b, lb: LinearBounds, use_ibp=False, use_ibp_u=True):
    """
    Bound f(x) = x^T Sigma2_w x + sigma2_b
    :param Sigma2_w: n x n
    :param sigma2_b: n
    :param lb: LinearBounds object
    :param use_ibp: initialized LinearBounds object
    :return:
    """
    lb_new = copy(lb)
    if not (lb_new.a_l is None or lb_new.a_u is None or lb_new.b_l is None or lb_new.b_u is None):
        raise Warning("bound_quad does not propagate bounds")
    inp_int = torch.stack((lb_new.l, lb_new.u), dim=0)

    lb_new.a_u, lb_new.b_u = bound_quad_u(inp_int, Sigma2=Sigma2_w, use_ibp_u=use_ibp_u)
    lb_new.b_u += sigma2_b

    lb_new.a_l, lb_new.b_l = bound_quad_l(inp_int, Sigma2=Sigma2_w, use_ibp=use_ibp)
    lb_new.b_l = lb_new.b_l + sigma2_b

    if use_ibp:
        lb_new.a_l, lb_new.b_l, lb_new.a_u, lb_new.b_u = lb2ib(inp_int, lb_new.a_l, lb_new.b_l, lb_new.a_u, lb_new.b_u)

    lb_new.set_l_u()
    return lb_new


## SQRT ------------------------------------------------------------------------------------------------------------- ##
def sqrt(x):
    return torch.sqrt(x.round(decimals=8))


def grad_sqrt(x):
    return 1/(2*torch.sqrt(x))


def bound_sqrt(lb):
    lb_new = copy(lb)
    if lb_new.l <= 0.:
        raise Warning('{} lower bound should be larger than zero'.format('bound_sqrt'))
    l = lb_new.l
    u = lb_new.u
    a_l, b_l = fit_1Dlin_func(l, u, sqrt)
    a_u, b_u = tang(u, func=sqrt, grad_func=grad_sqrt)
    lb_new.forward(a_l, b_l, a_u, b_u)
    return lb_new


## RELU ----------------------------------------------------------------------------------------------------------------
# def bound_ReLU(lb:LinearBounds, use_ibp=False):
#     lb_new = copy(lb)
#     inp_int = torch.stack((lb_new.l, lb_new.u), dim=0)
#
#     if lb_new.l >= 0.:
#         lb_new.a_l, lb_new.b_l, lb_new.a_u, lb_new.b_u = 1., 0., 1., 0.
#     elif lb_new.u <= 0.:
#         lb_new.a_l, lb_new.b_l, lb_new.a_u, lb_new.b_u = 0., 0., 0., 0.
#     else:
#         lb_new.a_l = 0.
#         lb_new.b_l = 0.
#         lb_new.a_u = lb_new.u / (lb_new.u - lb_new.l)
#         lb_new.b_u = -lb_new.l * lb_new.a_u
#
#     if use_ibp:
#         lb_new.a_l, lb_new.b_l, lb_new.a_u, lb_new.b_u = lb2ib(inp_int, lb_new.a_l, lb_new.b_l, lb_new.a_u, lb_new.b_u)
#
#     lb_new.set_l_u()
#     return lb_new


## DIVIDE ----------------------------------------------------------------------------------------------------------- ##
def grad_divide(x):
    return -1/x**2


def divide(x):
    return 1/x


def bound_divide(lb_in):
    lb = copy(lb_in)
    l = lb.l
    u = lb.u
    if l < 0. or u <= 0.:
        raise ImportError("{} takes l >= 0., u > 0.".format('bound_divide'))
    if l == 0.:
        a_u = torch.zeros(l.shape)
        b_u = torch.inf
    else:
        a_u, b_u = fit_1Dlin_func(l, u, divide, upper_prio=True)

    a_l, b_l = tang(u, func=divide, grad_func=grad_divide)
    lb.forward(a_l, b_l, a_u, b_u)
    return lb


## EXP -------------------------------------------------------------------------------------------------------------- ##
def exp(x):
    return torch.exp(x)


def grad_exp(x):
    return torch.exp(x)


def bound_exp(lb_in):
    lb = copy(lb_in)
    a_u, b_u = fit_1Dlin_func(lb.l, lb.u, exp)
    a_l, b_l = tang(lb.l, func=exp, grad_func=grad_exp)
    lb.forward(a_l, b_l, a_u, b_u)
    return lb


## LB Checks / Operations ------------------------------------------------------------------------------------------- ##
def flip_lb(a_l, b_l, a_u, b_u):
    return -a_u, -b_u, -a_l, -b_l


def lbcheck(inp_int, a_l, b_l, a_u, b_u, warn=True):
    if a_l.dim() > 1:
        raise NotImplementedError
    elif check4inf([a_l, b_l, a_u, b_u]) or check4nan([a_l, b_l, a_u, b_u]):
        raise Warning('detected inf or nan in bounds to check')

    mask_L = a_l >= 0.
    mask_U = a_u >= 0.

    l_min = torch.einsum('j,j->', a_l[mask_L], inp_int[0][mask_L]) + \
            torch.einsum('j,j->', a_l[~mask_L], inp_int[1][~mask_L]) + b_l
    l_max = torch.einsum('j,j->', a_l[mask_L], inp_int[1][mask_L]) + \
            torch.einsum('j,j->', a_l[~mask_L], inp_int[0][~mask_L]) + b_l
    point = inp_int[0].clone() # point[~mask_L] = inp_int[1][~mask_L]
    u_min = torch.einsum('j,j->', a_u[mask_U], inp_int[0][mask_U]) + \
            torch.einsum('j,j->', a_u[~mask_U], inp_int[1][~mask_U]) + b_u
    u_max = torch.einsum('j,j->', a_u[mask_U], inp_int[1][mask_U]) + \
            torch.einsum('j,j->', a_u[~mask_U], inp_int[0][~mask_U]) + b_u

    if warn:
        if l_min - u_min  > 1e-2:
            raise Warning('upper and lower bound overlap (u_min < l_min)')
        elif l_max - u_max > 1e-2:
            raise Warning('upper and lower bound overlap (l_max > u_max)')


    return l_min, l_max, u_min, u_max


def lb2ib(inp_int, a_l_in, b_l_in, a_u_in, b_u_in, sat=None, warn=True):
    a_l, b_l, a_u, b_u = a_l_in.clone(), b_l_in.clone(), a_u_in.clone(), b_u_in.clone()
    if len(a_l.shape) > 1:
        for i in range(a_l.shape[0]):
            a_l[i], b_l[i], a_u[i], b_u[i] = lb2ib(inp_int, a_l[i], b_l[i], a_u[i], b_u[i], warn=warn)
        return a_l, b_l, a_u, b_u
    else:
        b_l, l_max, u_min, b_u = lbcheck(inp_int, a_l, b_l, a_u, b_u, warn=warn)

        a_l = torch.zeros(a_l.shape)
        a_u = torch.zeros(a_u.shape)

        if torch.any(b_l.round(decimals=4) > b_u.round(decimals=4)):
            Warning('lower bounds exceed upper bounds in lib2ib')

        return a_l, b_l, a_u, b_u


def lb_zero_check(inp_int, a_l, b_l, a_u, b_u, disabled=True, warn=True):
    if a_l.dim() > 1:
        raise NotImplementedError

    l_min, l_max, u_min, u_max = lbcheck(inp_int, a_l, b_l, a_u, b_u, warn=warn)

    if not disabled:
        if l_min <= 0.:
            a_l, b_l = torch.zeros(a_l.shape), torch.tensor([0.])

        if u_max <= 0.:
            a_u, b_u = torch.zeros(a_u.shape), torch.tensor([0.])

    return a_l, b_l, a_u, b_u


def lin_func(x, a, b):
    if x.dim() == 1:
        return torch.einsum('i,i', a, x) + b
    elif x.dim() == 2:
        return torch.einsum('i,ki->k', a, x) + b

## -- SOFTMAX
def bound_softmax(inp_int):
    classes = inp_int.shape[1]
    logit_L = inp_int[0]
    logit_U = inp_int[1]

    softmax = torch.nn.Softmax(dim=-1)

    class_L = torch.zeros(logit_L.shape)
    class_U = torch.zeros(logit_U.shape)

    for i in range(classes):
        lower_in = logit_U.clone()
        lower_in[i] = logit_L[i]
        class_L[i] = softmax(lower_in)[i]

        upper_in = logit_L.clone()
        upper_in[i] = logit_U[i]
        class_U[i] = softmax(upper_in)[i]

    # true class
    true_label = torch.argmax(logit_U)

    cor = torch.max(torch.zeros(1), torch.ones(1) - (torch.sum(class_U) - class_U[true_label]))
    class_L[true_label] = torch.min(torch.max(class_L[true_label], cor), torch.max(class_U[true_label]))

    return {'A_L': torch.zeros((classes, classes)),
            'b_L': class_L,
            'A_U': torch.zeros((classes, classes)),
            'b_U': class_U}


## -- ACTIVATION FUNCTIONS
def bound_ReLU(inp_int: torch.tensor):
    if not all([inp_int.dim() == 1, inp_int[0] <= inp_int[1]]):
        raise Warning('inp_int should be tensor shape 1, with inp_int[0] <= inp_int[1]')

    if inp_int[0] >= 0.:
        return 1., 0., 1., 0.
    elif inp_int[1] <= 0.:
        return 0., 0., 0., 0.
    else:
        a_l = 0.
        b_l = 0.
        a_u = inp_int[1] / (inp_int[1] - inp_int[0])
        b_u = -inp_int[0] * a_u
        return a_l, b_l, a_u, b_u


## BOXES
def bound_rotated_box(orig_int: torch.tensor, T: torch.tensor):
    """
    http://www.realtimerendering.com/resources/GraphicsGems/gems/TransBox.c
    """
    tf_int = torch.zeros(orig_int.shape)
    dims = orig_int.shape[1]

    for i in range(dims):
        for j in range(dims):
            a = T[i,j] * orig_int[0, i]
            b = T[i,j] * orig_int[1, j]
            if a < b:
                tf_int[0, i] += a
                tf_int[1, i] += b
            else:
                tf_int[0, i] += b
                tf_int[1, i] += a
    return tf_int



## PLOTTING --------------------------------------------------------------------------------------------------------- ##
def plot_func_only(a_l, b_l, a_u, b_u, inp_int, orig_func, margin=0.5, x_fixt = 0.1, title=' ', z_int=None):
    dim = inp_int.shape[1]
    N = 100
    x = torch.linspace(max(inp_int[0,0]-margin,0.), inp_int[1,0]+margin, N)
    x_approx = torch.linspace(inp_int[0,0], inp_int[1,0], N)
    x = x[:, None]
    x_approx = x_approx[:, None]

    # x = torch.cat((x, torch.ones(x.shape)*x_fixt), dim=1)
    # x_approx = torch.cat((x_approx, torch.ones(x_approx.shape)*x_fixt), dim=1)
    y = orig_func(x)
    y_approx_l = a_l*x_approx + b_l
    y_approx_u = a_u*x_approx + b_u

    plt.title(title)
    # plt.plot(0., orig_func(x[0]*0.), 'r*')
    plt.plot(x[:,0], y, 'k-')
    plt.plot(x_approx[:,0], y_approx_l, 'r-')
    plt.plot(x_approx[:,0], y_approx_u, 'g-')
    # plt.plot(bounds.l_init[0], orig_func(torch.cat((bounds.l_init[0][None], torch.ones(dim-1)*x_fixt))), 'r*')
    # plt.plot(bounds.u_init[0], orig_func(torch.cat((bounds.u_init[0][None], torch.ones(dim-1)*x_fixt))), 'r*')
    if z_int is not None:
        plt.plot(z_int, [0.,0.], 'b-')
    plt.xlim(x[0],x[-1])
    plt.show()


def plot(bounds, orig_func, margin=0.5, x_fixt = 0.1, title=' ', z_int=None):
    """
    can be used to plot >1D by fixing other dimension than dim 0
    :param bounds:
    :param orig_func:
    :param margin:
    :param x_fixt:
    :param title:
    :return:
    """
    dim = bounds.l_init.shape[0]
    N = 100
    x = torch.linspace(max((bounds.l_init-margin)[0],0.), (bounds.u_init+margin)[0], N)
    x_approx = torch.linspace(bounds.l_init[0], bounds.u_init[0], N)
    x = x[:, None]
    x_approx = x_approx[:, None]

    if bounds.l_init.shape[0]> 1:
        x = torch.cat((x, torch.ones(x.shape)*x_fixt), dim=1)
        x_approx = torch.cat((x_approx, torch.ones(x_approx.shape)*x_fixt), dim=1)
    y = orig_func(x)
    y_approx_l = bounds.lin_l(x_approx)
    y_approx_u = bounds.lin_u(x_approx)

    plt.title(title)
    # plt.plot(0., orig_func(x[0]*0.), 'r*')
    plt.plot(x[:,0], y, 'k-')
    plt.plot(x_approx[:,0], y_approx_l, 'r-')
    plt.plot(x_approx[:,0], y_approx_u, 'g-')
    plt.plot(bounds.l_init[0], orig_func(torch.cat((bounds.l_init[0][None], torch.ones(dim-1)*x_fixt))), 'r*')
    plt.plot(bounds.u_init[0], orig_func(torch.cat((bounds.u_init[0][None], torch.ones(dim-1)*x_fixt))), 'r*')
    if z_int is not None:
        plt.plot(z_int, [0.,0.], 'b-')
    plt.xlim(x[0],x[-1])
    plt.show()

