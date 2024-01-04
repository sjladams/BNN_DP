import torch
from bound import Node

import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy import stats

if __name__=='__main__':
    # problamatic values for bound propagation cdf part: z_l = 0.5, z_u = 0.75, z_l=0.5, x_u=1, Sigma_w=eye,mu_w=1
    # Example
    z_l = torch.tensor(0.)
    z_u = torch.tensor(torch.inf)
    x_l = torch.tensor([0.35]) # test mult dim x
    x_u = torch.tensor([0.45])
    interval = torch.stack((x_l, x_u), dim=0)
    interval_z = torch.stack((z_l, z_u), dim=0)

    # Sigma_w = torch.tensor([[0.014**0.5]])
    # sigma_b = torch.tensor([0.0074**0.5])
    # mu_w = torch.tensor([-0.1277])
    # mu_b = torch.tensor([0.014])
    Sigma_w = torch.eye(x_l.shape[0])
    sigma_b = torch.tensor([0.001])
    mu_w = torch.ones(x_l.shape[0])
    mu_b = torch.tensor([0.])

    # z\in [z_l, z_u]
    # pro_l_int(z_l, interval, Sigma_w, sigma_b, mu_w, mu_b)
    node = Node(weight_loc=mu_w, weight_std=Sigma_w, bias_loc=mu_b, bias_std=sigma_b)
    # node.use_ibp = False
    node.plot_steps = True
    a_l, b_l, a_u, b_u = node.cond_expect(interval, interval_z, use_ibp=False, use_extrema=True)
    # a_l, b_l, a_u, b_u = node.cond_expect(interval, interval_z, use_ibp=True)
    # _, prob_l, _, prob_u = node.interval_cdf(inp_int=interval, point_int=interval_z, use_ibp=True, use_extrema=False)
    # _, prob_l, _, prob_u = node.interval_cdf(inp_int=interval, point_int=interval_z, use_ibp=True)

    # print(prob_u-prob_l)
    x_approx = torch.linspace(x_l[0], x_u[0], 100)
    y_l_approx = a_l*x_approx + b_l
    y_u_approx = a_u*x_approx + b_u

    ## numerical verification
    def expectation(sigma, mu, lbound=0., ubound=20.):
        def func(x):
            return x*stats.norm.pdf(x, loc = mu, scale = sigma)
        return integrate.quad(func, lbound, ubound)[0]

    N = 100
    y0 = np.zeros(N)
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    y3 = np.zeros(N)
    yref = np.zeros(N)
    x = np.linspace(x_l[0]-0.1,x_u[0]+0.1 ,N)

    for i in range(N):
        mu = mu_w*x[i]
        sigma = np.sqrt(Sigma_w**2*x[i]**2)

        y0[i] = expectation(sigma, mu - z_l, lbound=0.)
        y1[i] = -expectation(sigma, mu - z_u, lbound=0.)
        y2[i] = z_l*0.5*(1 - erf((-mu+z_l)/(sigma*np.sqrt(2))))
        y3[i] = -z_u*0.5*(1 - erf((-mu+z_u)/(sigma*np.sqrt(2))))
        # yref[i] = expectation(sigma, mu, lbound=z_l, ubound=20.)
        # yref[i] = expectation(sigma, mu, lbound=z_u, ubound=20.)
        yref[i] = expectation(sigma, mu, lbound=z_l, ubound=z_u)

    y = y0+y1+y2+y3

    plt.plot(x,y,'-k')
    plt.plot(x,yref,'-g')

    plt.plot(x_approx, y_l_approx, '-r', linewidth=1)
    plt.plot(x_approx, y_u_approx, '-g', linewidth=1)
    # plt.xlim([x_l[0],x_u[0]])
    # plt.ylim([0.4,1])
    plt.show()
    print('nu dan')

