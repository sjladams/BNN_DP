import torch
from functools import partial
import numpy as np

import bound

## ------------------------------------------------------------------------------------------------------------------ ##
class Node:
    def __init__(self, weight_loc, weight_std, bias_loc, bias_std, act: str):
        self.act = act
        self.weight_loc = weight_loc
        self.weight_var = weight_std ** 2
        self.bias_loc = bias_loc
        self.bias_var = bias_std ** 2

        self.dim = weight_loc.shape[0]
        self.plot_steps = False

    def loc_scale_extremes(self, inp_int, use_ibp=True, use_ibp_u=True):
        if not all([torch.is_tensor(inp_int), inp_int.dim() == 2, inp_int.numel() == 2 * self.dim,
                    torch.inf not in inp_int]):
            raise ImportError('{} inp_int should be 2D tensor with non inf values'.format('loc_scale_extremes'))
        if self.act == 'ReLU' or self.act == 'Exp':
            if not torch.all(inp_int >= 0.):
                raise ImportError('{} inp_int should contain >= 0. values'.format('log_scale_extremes'))

        lb_mu = bound.LinearBounds(l=inp_int[0], u=inp_int[1])
        lb_mu.init(self.weight_loc, self.bias_loc, self.weight_loc, self.bias_loc)
        mu_min, mu_max = lb_mu.l, lb_mu.u

        lb_sigma2 = bound.LinearBounds(l=inp_int[0], u=inp_int[1])
        lb_sigma2 = bound.bound_quad(Sigma2_w=self.weight_var, sigma2_b=self.bias_var, lb=lb_sigma2, use_ibp=use_ibp,
                               use_ibp_u=use_ibp_u)
        sigma2_min, sigma2_max = lb_sigma2.l, lb_sigma2.u

        lb_sigma = bound.bound_sqrt(lb=lb_sigma2)

        return lb_mu, lb_sigma2, lb_sigma

    def cond_expect(self, inp_int=None, out_int=None, use_ibp=True, use_ibp_u=True):
        """
        :param inp_int:
        :param out_int:
        :param use_ibp:
        :param use_extrema: optimize using max/min of sigma and mu w.r.t. x
        :return:
        """

        if use_ibp:
            use_ibp_u = True

        if self.act == 'Linear' and (out_int[0] == -torch.inf and out_int[1] == torch.inf):
            return self.weight_loc, self.bias_loc, self.weight_loc, self.bias_loc
        elif self.act == 'Exp':
            if not (out_int[0] == 0. and out_int[1] == torch.inf):
                raise NotImplementedError
            else:
                a_l, b_l = self.l_bound_expect_exp_gauss(inp_int, use_ibp=use_ibp)
                a_u, b_u = self.u_bound_expect_exp_gauss(inp_int, use_ibp=use_ibp, use_ibp_u=use_ibp_u)
                _ = bound.lbcheck(inp_int, a_l, b_l, a_u, b_u, warn=True)
                return a_l, b_l, a_u, b_u
        elif self.act == 'PreReLU':
            ## linear relaxation z_k w.r.t. zeta_k
            A_L_x, A_U_x = torch.eye(self.dim), torch.eye(self.dim)
            b_L_x, b_U_x = torch.zeros(self.dim), torch.zeros(self.dim)

            for i in range(self.dim):
                A_L_x[i,i], b_L_x[i], A_U_x[i,i], b_U_x[i] = bound.bound_ReLU(inp_int=inp_int[:, i])

            ## linear relaxation zeta_{k+1} w.r.t. z_k
            inp_int_sat = torch.clip(inp_int, 0, torch.inf)

            lb_mu, lb_sigma2, lb_sigma = self.loc_scale_extremes(inp_int_sat, use_ibp=use_ibp, use_ibp_u=use_ibp_u)
            options = torch.cartesian_prod(torch.cat((lb_mu.l, lb_mu.u)), torch.cat((lb_sigma2.l, lb_sigma2.u)))

            if out_int[0] != -torch.inf:
                prob_0_max = torch.max(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[0]),
                                                             options[:, 0], options[:, 1]))))[None]
                prob_0_min = torch.min(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[0]),
                                                             options[:, 0], options[:, 1]))))[None]

                a_l_0, b_l_0 = self.l_bound_expect_rect_gauss(inp_int_sat,  out_int_l=out_int[0], use_ibp=use_ibp,
                                                              lb_mu=lb_mu, lb_sigma=lb_sigma, lb_sigma2=lb_sigma2)
                a_u_0, b_u_0 = self.u_bound_expect_rect_gauss(inp_int_sat, out_int_l=out_int[0], use_ibp=use_ibp,
                                                              use_ibp_u=use_ibp_u, lb_mu=lb_mu, lb_sigma=lb_sigma,
                                                              lb_sigma2=lb_sigma2)

                _ = bound.lbcheck(inp_int_sat, a_l_0, b_l_0, a_u_0, b_u_0, warn=True) # CHECK if bounds don't violate

            if out_int[1] != torch.inf:
                prob_1_max = -torch.min(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[1]),
                                                              options[:, 0], options[:, 1]))))[None]
                prob_1_min = -torch.max(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[1]),
                                                              options[:, 0], options[:, 1]))))[None]

                a_u_1, b_u_1 = self.u_bound_expect_rect_gauss(inp_int_sat, out_int_l=out_int[1], use_ibp=use_ibp,
                                                              use_ibp_u=use_ibp_u, lb_mu=lb_mu, lb_sigma=lb_sigma,
                                                              lb_sigma2=lb_sigma2)
                a_l_1, b_l_1 = self.l_bound_expect_rect_gauss(inp_int_sat, out_int_l=out_int[1], use_ibp=use_ibp,
                                                              lb_mu=lb_mu, lb_sigma=lb_sigma, lb_sigma2=lb_sigma2)
                a_l_1, b_l_1, a_u_1, b_u_1 = bound.flip_lb(a_l_1, b_l_1, a_u_1, b_u_1)

                _ = bound.lbcheck(inp_int_sat, a_l_1, b_l_1, a_u_1, b_u_1, warn=True) # CHECK if bound don't violate

            # -- possible combinations of out_int --------------------------------------------------------------
            if out_int[0] == -torch.inf and out_int[1] == torch.inf:
                a_l, b_l, a_u, b_u = self.weight_loc, self.bias_loc, self.weight_loc, self.bias_loc
            elif out_int[0] == -torch.inf:
                a_l, b_l = a_l_1, b_l_1 + prob_1_min
                a_u, b_u = a_u_1, b_u_1 + prob_1_max
            elif out_int[1] == torch.inf:
                a_l, b_l = a_l_0, b_l_0 + prob_0_min
                a_u, b_u = a_u_0, b_u_0 + prob_0_max
            else:
                a_l, b_l = a_l_0 + a_l_1, b_l_0 + b_l_1 + prob_0_min + prob_1_min
                a_u, b_u = a_u_0 + a_u_1, b_u_0 + b_u_1 + prob_0_max + prob_1_max

            lb = bound.prop_lb(A_L_in=A_L_x, b_L_in=b_L_x, A_U_in=A_U_x, b_U_in=b_U_x,
                               A_L_out=a_l[None], b_L_out=b_l, A_U_out=a_u[None], b_U_out=b_u)
            return lb['A_L'].squeeze(), lb['b_L'], lb['A_U'].flatten(), lb['b_U']

        elif self.act == 'ReLU' or self.act == 'Linear':
            if self.act == 'ReLU':
                out_int = torch.clip(out_int, 0, torch.inf)

            lb_mu, lb_sigma2, lb_sigma = self.loc_scale_extremes(inp_int, use_ibp=use_ibp, use_ibp_u=use_ibp_u)
            options = torch.cartesian_prod(torch.cat((lb_mu.l, lb_mu.u)), torch.cat((lb_sigma2.l, lb_sigma2.u)))

            if out_int[0] != -torch.inf:
                prob_0_max = torch.max(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[0]),
                                                             options[:, 0], options[:, 1]))))[None]
                prob_0_min = torch.min(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[0]),
                                                             options[:, 0], options[:, 1]))))[None]

                a_l_0, b_l_0 = self.l_bound_expect_rect_gauss(inp_int,  out_int_l=out_int[0], use_ibp=use_ibp)
                a_u_0, b_u_0 = self.u_bound_expect_rect_gauss(inp_int, out_int_l=out_int[0], use_ibp=use_ibp,
                                                              use_ibp_u=use_ibp_u, lb_mu=lb_mu, lb_sigma=lb_sigma,
                                                              lb_sigma2=lb_sigma2)
                # # DEBUG
                # try:
                #     _ = bound.lbcheck(inp_int, a_l_0, b_l_0, a_u_0, b_u_0, warn=True) # CHECK if bounds don't violate
                # except:
                #     _ = self.cond_expect(inp_int=inp_int, out_int=out_int, use_ibp=use_ibp, use_ibp_u=use_ibp_u)

                _ = bound.lbcheck(inp_int, a_l_0, b_l_0, a_u_0, b_u_0, warn=True)  # CHECK if bounds don't violate

            if out_int[1] != torch.inf:
                prob_1_max = -torch.min(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[1]),
                                                              options[:, 0], options[:, 1]))))[None]
                prob_1_min = -torch.max(torch.tensor(list(map(partial(bound.func_prob_param, loc_cor=out_int[1]),
                                                              options[:, 0], options[:, 1]))))[None]

                a_u_1, b_u_1 = self.u_bound_expect_rect_gauss(inp_int, out_int_l=out_int[1], use_ibp=use_ibp,
                                                              use_ibp_u=use_ibp_u, lb_mu=lb_mu, lb_sigma=lb_sigma,
                                                              lb_sigma2=lb_sigma2)
                a_l_1, b_l_1 = self.l_bound_expect_rect_gauss(inp_int, out_int_l=out_int[1], use_ibp=use_ibp)
                a_l_1, b_l_1, a_u_1, b_u_1 = bound.flip_lb(a_l_1, b_l_1, a_u_1, b_u_1)

                # # DEBUG
                # try:
                #     _ = bound.lbcheck(inp_int, a_l_1, b_l_1, a_u_1, b_u_1, warn=True) # CHECK if bound don't violate
                # except:
                #     _ = self.cond_expect(inp_int=inp_int, out_int=out_int, use_ibp=use_ibp, use_ibp_u=use_ibp_u)

                _ = bound.lbcheck(inp_int, a_l_1, b_l_1, a_u_1, b_u_1, warn=True)  # CHECK if bound don't violate

            # -- possible combinations of out_int --------------------------------------------------------------
            if out_int[0] == -torch.inf:
                a_l, b_l = a_l_1, b_l_1 + prob_1_min
                a_u, b_u = a_u_1, b_u_1 + prob_1_max
            elif out_int[1] == torch.inf:
                a_l, b_l = a_l_0, b_l_0 + prob_0_min
                a_u, b_u = a_u_0, b_u_0 + prob_0_max
            else:
                a_l, b_l = a_l_0 + a_l_1, b_l_0 + b_l_1 + prob_0_min + prob_1_min
                a_u, b_u = a_u_0 + a_u_1, b_u_0 + b_u_1 + prob_0_max + prob_1_max

            if use_ibp:
                a_l, b_l, a_u, b_u = bound.lb2ib(inp_int, a_l, b_l, a_u, b_u)
            else:
                a_l, b_l, a_u, b_u = bound.lb_zero_check(inp_int, a_l, b_l, a_u, b_u, disabled=False, warn=True)

            return a_l, b_l, a_u, b_u

    def loc_scale(self, x, int_l=0.):
        loc = torch.einsum('i,i->', x, self.weight_loc) + self.bias_loc
        loc -= int_l
        scale = torch.sqrt(torch.einsum('i,ij,j->', x, self.weight_var, x) + self.bias_var)
        return loc, scale

    def func(self, x, int_l=0.): # \TODO int_l -> cor_loc
        if x.dim() == 1:
            loc, scale = self.loc_scale(x, int_l=int_l)
            if torch.all(loc == 0.) and torch.all(scale == 0.):
                return torch.tensor(0.)
            else:
                f = 0.5 * loc * (1 - torch.erf(-loc / (scale * np.sqrt(2)))) + \
                    (scale / np.sqrt(2 * np.pi)) * torch.exp(-(loc ** 2) / (2 * (scale ** 2)))
                return f
        elif x.dim() == 2:
            return torch.tensor(list(map(partial(self.func, int_l=int_l), x)))

    def grad(self, x, int_l=0.): # \TODO int_l -> cor_loc
        loc, scale = self.loc_scale(x, int_l=int_l)
        g = 0.5 * (1 - torch.erf(-loc / (scale * np.sqrt(2)))) * self.weight_loc + \
            (1 / (np.sqrt(2 * np.pi))) * (torch.einsum('ij,j->i', self.weight_var, x) / scale) * \
            torch.exp(-0.5 * (loc ** 2) / (scale ** 2))
        return g

    def create_disc(self, inp_int, precision, use_ibp=True, use_ibp_u=True):
        if not self.act in ['ReLU', 'Linear', 'PreReLU']:
            raise NotImplementedError

        if self.act == 'PreReLU':
            inp_int = torch.clip(inp_int, 0, torch.inf)

        lb_mu, lb_sigma2, lb_sigma = self.loc_scale_extremes(inp_int, use_ibp=use_ibp, use_ibp_u=use_ibp_u)

        a_u = precision * np.sqrt(2) * lb_sigma.a_u + lb_mu.a_u
        b_u = precision * np.sqrt(2) * lb_sigma.b_u + lb_mu.b_u
        _, _, _, z_u = bound.lb2ib(inp_int, a_u, b_u, a_u, b_u)

        a_l = - precision * np.sqrt(2) * lb_sigma.a_l + lb_mu.a_l
        b_l = - precision * np.sqrt(2) * lb_sigma.b_l + lb_mu.b_l
        _, z_l , _, _ = bound.lb2ib(inp_int, a_l, b_l, a_l, b_l)

        if self.act == 'ReLU':
            z_l = torch.clip(z_l, 0, torch.inf)
            z_u = torch.clip(z_u, 0, torch.inf)
        if z_u - z_l <= 1e-7:
            z_u += 1e-7
        return z_l, z_u

    def interval_cdf(self, inp_int, point_int, use_ibp=True, use_ibp_u=True):
        """
        \Phi(point_int[1] | mu, sigma) - \Phi(point_int[0] | mu, sigma) * [\delta(point_int[0] - 1]
        """

        if use_ibp:
            use_ibp_u = True

        if self.act == 'Exp':
            if point_int[0] == 0. and point_int[1] == torch.inf:
                return torch.zeros(self.dim), torch.ones(1), torch.zeros(self.dim), torch.ones(1)
            else:
                raise NotImplementedError
        elif self.act == 'PreReLU':
            if point_int[0] == -torch.inf and point_int[1] == torch.inf:
                return torch.zeros(self.dim), torch.ones(1), torch.zeros(self.dim), torch.ones(1)
            else:
                raise NotImplementedError
        elif self.act == 'SoftMax':
            if point_int[0] == 0. and point_int[1] == 1.:
                return torch.zeros(self.dim), torch.ones(1), torch.zeros(self.dim), torch.ones(1)
            else:
                raise NotImplementedError
        elif self.act == 'ReLU' or self.act == 'Linear' or self.act == 'PreReLU':
            # Since we're applying ibp can solve the PreReLU case by only considering the inp_int space, ReLU is linear:
            if self.act == 'PreReLU':
                inp_int = torch.clip(inp_int, 0, torch.inf)

            lb_mu, lb_sigma2, lb_sigma = self.loc_scale_extremes(inp_int, use_ibp=use_ibp, use_ibp_u=use_ibp_u)
            options = torch.cartesian_prod(torch.cat((lb_mu.l, lb_mu.u)), torch.cat((lb_sigma2.l, lb_sigma2.u)))

            # In case of relu function (-> rectified gaussian) all mass of the negative support is cumulated at zero
            if point_int[0] == 0. and self.act == 'ReLU':
                cdf_0_min, cdf_0_max = torch.zeros(1), torch.zeros(1)
            else:
                cdf_0_options = 1 - 0.5 * torch.tensor(list(map(partial(bound.func_param_erf_term, loc_cor=point_int[0]),
                                                            options[:, 0], options[:, 1])))
                cdf_0_min = torch.min(cdf_0_options)[None]
                cdf_0_max = torch.max(cdf_0_options)[None]

            cdf_1_options = 1 - 0.5 * torch.tensor(list(map(partial(bound.func_param_erf_term, loc_cor=point_int[1]),
                                                            options[:, 0], options[:, 1])))
            cdf_1_min = torch.min(cdf_1_options)[None]
            cdf_1_max = torch.max(cdf_1_options)[None]

            # Combine
            b_u = cdf_1_max - cdf_0_min
            b_l = torch.max(cdf_1_min - cdf_0_max, torch.zeros(1))

            a_u = torch.zeros(self.dim)
            a_l = torch.zeros(self.dim)

            _ = bound.lbcheck(inp_int, a_l, b_l, a_u, b_u, warn=True)

            return a_l, b_l, a_u, b_u
        else:
            raise NotImplementedError

    def expon_exp_grad(self, x):
        loc, scale = self.loc_scale(x, torch.tensor(0.))
        return torch.exp(loc + scale ** 2 / 2) * (self.weight_loc + torch.einsum('ij,j->i', self.weight_var, x))

    def expon_exp(self, x):
        loc, scale = self.loc_scale(x, torch.tensor(0.))
        return torch.exp(loc + scale ** 2 / 2)

    def l_bound_expect_exp_gauss(self, inp_int, use_ibp: bool):
        mask = self.weight_loc >= 0.
        x_min = inp_int[0].clone()
        x_min[~mask] = inp_int[1][~mask]
        val = self.expon_exp(x_min)

        # x_min, val = gradient_descent(inp_int, self.expon_exp, self.expon_exp_grad, n_iter=500, learn_rate=0.05, tol=1e-3)

        if use_ibp:
            return torch.zeros(self.dim), val
        else:
            # return tang(torch.mean(inp_int, axis=0), self.expon_exp, self.expon_exp_grad)
            return bound.tang(x_min, func=self.expon_exp, grad_func=self.expon_exp_grad)

    def l_bound_expect_rect_gauss(self, inp_int, out_int_l, use_ibp, lb_mu: bound.LinearBounds=None,
                                  lb_sigma2: bound.LinearBounds=None, lb_sigma: bound.LinearBounds=None):
        """
        Proposition 11: Lower bound:
        g(x) = \mu/2 * (1 - erf(-mu/(\sigma\sqrt{2}))) + \sigma/(\sqrt(2\pi) exp(-\mu^2 / (2\sigma^2)
        """

        if lb_mu is None and lb_sigma2 is None and lb_sigma is None:
            func_dum = partial(self.func, int_l=out_int_l)
            grad_dum = partial(self.grad, int_l=out_int_l)

            mask = self.weight_loc >= 0.
            x_min = inp_int[0].clone()
            x_min[~mask] = inp_int[1][~mask]
            val_min = func_dum(x_min)

            mask = self.weight_loc >= 0.
            x_max = inp_int[0].clone()
            x_max[mask] = inp_int[1][mask]
            val_max = func_dum(x_max)

            if val_max < val_min:
                val = val_max
                x = x_max
            else:
                val = val_min
                x = x_min

            # x, val = gradient_descent(inp_int, func_dum, grad_dum, n_iter=10000, learn_rate=0.01, tol=1e-7, starting_point=x)
            a_l, b_l = bound.tang(x, func=func_dum, grad_func=grad_dum)
            _, ref, _, _ = bound.lb2ib(inp_int, a_l, b_l, a_l, b_l)

            if use_ibp:
                return torch.zeros(self.dim), val[None]
            else:
                return bound.tang(x, func=func_dum, grad_func=grad_dum)
        else:
            a_l_mu_sigma, b_l_mu_sigma = bound.tang(lb_mu.l, lb_sigma.l,
                                                    func=partial(bound.func_param, loc_cor=out_int_l),
                                                    grad_func=partial(bound.grad_param, loc_cor=out_int_l))

            ## propagate bounds
            lb_mu_part = bound.prop_lb(A_L_in=lb_mu.a_l[None], b_L_in=lb_mu.b_l, A_U_in=lb_mu.a_u[None],
                                       b_U_in=lb_mu.b_u, A_L_out=a_l_mu_sigma[0][None, None],
                                       b_L_out=torch.zeros(1))

            lb_sigma_part = bound.prop_lb(A_L_in=lb_sigma.a_l[None], b_L_in=lb_sigma.b_l, A_U_in=lb_sigma.a_u[None],
                                          b_U_in=lb_sigma.b_u, A_L_out=a_l_mu_sigma[1][None, None],
                                          b_L_out=torch.zeros(1))

            a_l = lb_mu_part['A_L'][0] + lb_sigma_part['A_L'][0]
            b_l = lb_mu_part['b_L'] + lb_sigma_part['b_L'] + b_l_mu_sigma

            if use_ibp:
                _, b_l_ib, _, _ = bound.lb2ib(inp_int, a_l, b_l, a_l, b_l)
                return torch.zeros(self.dim), b_l_ib
            else:
                return a_l, b_l

    def u_bound_expect_exp_gauss(self, inp_int, use_ibp: bool, use_ibp_u: bool):
        mu, sigma2, sigma = self.loc_scale_extremes(inp_int, use_ibp=use_ibp, use_ibp_u=use_ibp_u)

        lb_exp = bound.LinearBounds(l=inp_int[0], u=inp_int[1])
        lb_exp.init(mu.a_l + sigma2.a_l, mu.b_l + sigma2.b_l, mu.a_u + sigma2.a_u, mu.b_u + sigma2.b_u)

        lb_exp = bound.bound_exp(lb_exp)

        if use_ibp_u:
            return torch.zeros(self.dim), lb_exp.u
        else:
            return lb_exp.a_u, lb_exp.b_u

    def u_bound_expect_rect_gauss(self, inp_int: torch.tensor, out_int_l: torch.tensor, use_ibp: bool, use_ibp_u: bool,
                                  lb_mu: bound.LinearBounds, lb_sigma2: bound.LinearBounds, lb_sigma: bound.LinearBounds):
        """
        Proposition 11: Upper bound:
        g(x) = \mu/2 * (1 - erf(-mu/(\sigma\sqrt{2}))) + \sigma/(\sqrt(2\pi) exp(-\mu^2 / (2\sigma^2)
        """

        if inp_int.shape[1] == 1:  # enable for perfect 1D handling
            a_u, b_u = bound.fit_1Dlin_func(inp_int[0], inp_int[1], partial(self.func, int_l=out_int_l))
        else:
            ## find \alpha_{g/\mu}, \alpha_{g/\sigma}, \beta_{g}
            options = torch.cartesian_prod(torch.cat((lb_mu.l, lb_mu.u)), torch.cat((lb_sigma.l, lb_sigma.u)))
            vals = list(map(partial(bound.func_param, loc_cor=out_int_l), options[:, 0], options[:, 1]))
            vertices_idx = list(set([np.argmax(vals), np.argmin(vals)]))
            vertices_idx.append([idx for idx in range(0, len(vals)) if idx not in vertices_idx][0])
            if len(vertices_idx) < 3:
                vertices_idx.append([idx for idx in range(0, len(vals)) if idx not in vertices_idx][1])
            vertices = options[vertices_idx]
            # vertices[:, 1] = vertices[:, 1]**0.5
            vals_vertices = torch.tensor(vals)[vertices_idx]

            int_mu_sigma = torch.stack((torch.cat((lb_mu.l, lb_sigma.l)), torch.cat((lb_mu.u, lb_sigma.u))))
            a_u_mu_sigma, b_u_mu_sigma = bound.get_hyperplane(interval=int_mu_sigma, vertices=vertices,
                                                              vals=vals_vertices)

            ## propagate bounds
            lb_mu_part = bound.prop_lb(A_L_in=lb_mu.a_l[None], b_L_in=lb_mu.b_l, A_U_in=lb_mu.a_u[None],
                                       b_U_in=lb_mu.b_u, A_L_out=a_u_mu_sigma[0][None, None],
                                       b_L_out=torch.zeros(1))

            lb_sigma_part = bound.prop_lb(A_L_in=lb_sigma.a_l[None], b_L_in=lb_sigma.b_l, A_U_in=lb_sigma.a_u[None],
                                              b_U_in=lb_sigma.b_u, A_L_out=a_u_mu_sigma[1][None, None],
                                              b_L_out=torch.zeros(1))

            a_u = lb_mu_part['A_U'][0] + lb_sigma_part['A_U'][0]
            b_u = lb_mu_part['b_U'] + lb_sigma_part['b_U'] + b_u_mu_sigma

        if use_ibp_u:
            _, _, _, b_u_ib = bound.lb2ib(inp_int, a_u, b_u, a_u, b_u)
            return torch.zeros(self.dim), b_u_ib
        else:
            return a_u, b_u