from support import *
from nodes import Node
import bound
import math
# from bound import bound_softmax, lb2ib

softplus = torch.nn.Softplus()
gamma = 1 + 1 / (8 * math.pi * math.e)

## -- LAYER ------------------------------------------------------------------------------------------------------------
class Layer:
    def __init__(self, act: str, layer_tag=0, imp_tag=0, apply_softplus=False, true_preds=None):
        self.layer_tag = layer_tag
        if act not in ['Linear', 'Exp', 'ReLU', 'SoftMax', 'PreReLU']:
            raise ValueError('act not implemented')

        if act == 'SoftMax':
            if true_preds is None:
                raise ValueError('if last_mnist_layer, provide true_preds')
            elif len(true_preds.shape) > 1:
                raise ValueError('true_preds should be an unit vector, indicating the true class')

            nr_classes = true_preds.shape[0]
            M_last = torch.eye(nr_classes) - torch.einsum('i,j->ij', torch.ones(true_preds.shape), true_preds)
            self.W_loc = M_last
            self.b_loc = torch.zeros(nr_classes)
            self.W_std = torch.zeros((nr_classes, nr_classes))
            self.b_std = torch.zeros(nr_classes)

        else:
            self.W_loc = get_param('weight', 'loc', imp_tag)
            self.b_loc = get_param('bias', 'loc', imp_tag)
            self.W_std = get_param('weight', 'scale', imp_tag)
            self.b_std = get_param('bias', 'scale', imp_tag)

            if apply_softplus:
                self.W_std = softplus(self.W_std)
                self.b_std = softplus(self.b_std)

        if self.b_loc.dim() == 1:
            self.b_loc = self.b_loc[:, None]
        if self.b_std.dim() == 1:
            self.b_std = self.b_std[:, None]

        self.act = act
        self.neurons = self.W_loc.shape[0]

        self.nodes = [Node(weight_loc=self.W_loc[i,:],
                           weight_std=torch.diag(self.W_std[i,:]),
                           bias_loc=self.b_loc[i],
                           bias_std=self.b_std[i],
                           act=self.act)
                      for i in range(0, self.neurons)]
        self.inp_dim = self.W_loc.shape[1]
        self.out_dim = self.neurons

    def empty_lin_bounds(self):
        return {'A_L': torch.zeros((self.neurons, self.inp_dim)),
                'b_L': torch.zeros((self.neurons)),
                'A_U': torch.zeros((self.neurons, self.inp_dim)),
                'b_U': torch.zeros((self.neurons))}

    def empty_onesided_lin_bounds(self):
        return {'A': torch.zeros((self.neurons, self.inp_dim)),
                'b': torch.zeros((self.neurons))}

    def emtpy_prob_bounds(self):
        return {'l_probs': torch.zeros((self.neurons)),
                'u_probs': torch.zeros((self.neurons)),
                'l_prob': torch.zeros(1),
                'u_prob': torch.zeros(1)}

    def cond_expect(self, inp_int=None, out_int=None, use_ibp=True, use_ibp_u=True):
        if self.act == 'SoftMax':
            lin_bounds = bound.bound_softmax(inp_int)
        elif self.act in ['ReLU', 'Linear', 'Exp', 'PreReLU']:
            lin_bounds = self.empty_lin_bounds()
            for i, node in enumerate(self.nodes):
                lin_bounds['A_L'][i], lin_bounds['b_L'][i], lin_bounds['A_U'][i], lin_bounds['b_U'][i] = \
                    node.cond_expect(inp_int=inp_int, out_int=out_int[:, i], use_ibp=use_ibp, use_ibp_u=use_ibp_u)
        else:
            raise NotImplementedError

        # checks contain nan or inf?
        toCheck = [lin_bounds['A_L'], lin_bounds['A_U'], lin_bounds['b_L'], lin_bounds['b_U']]
        if check4inf(toCheck) or check4nan(toCheck):
            raise Warning('inf or nan values detected in linear bounds on conditional expectation')

        return lin_bounds

    def create_disc(self, disc_in, args, use_ibp=True, use_ibp_u=True, precision: float = 0.98, **kwargs):
        if not self.act in ['ReLU', 'Linear', 'PreReLU']:
            raise NotImplementedError

        precision_per_node = torch.erfinv(torch.tensor(precision ** (1 / self.out_dim)))

        z_l = torch.zeros(self.out_dim)
        z_u = torch.zeros(self.out_dim)
        for i in range(self.out_dim):
            z_l[i], z_u[i] = self.nodes[i].create_disc(inp_int=disc_in.int, precision=precision_per_node,
                                                       use_ibp=use_ibp, use_ibp_u=use_ibp_u)
        if not torch.all(z_l<=z_u):
            raise ValueError

        l_prob_node = torch.tensor(precision ** (1 / self.out_dim))

        if self.act == 'ReLU':
            parents = [Disc(z_l, z_u, id=0, sat=True)]
        else:
            parents = [Disc(z_l, z_u, id=0, sat=False)]

        prob_info = {0: {'l_probs': torch.ones(self.out_dim) * l_prob_node,
                         'u_probs': torch.ones(self.out_dim), 'spec_nodes': [],
                         'precision': precision}
                     }

        discs = Discs(discs=parents, layer=self, disc_inp=disc_in, layer_tag=self.layer_tag + 1, prob_info=prob_info)
        return discs


    def interval_cdf(self, inp_int, out_int, use_ibp=True, use_ibp_u=True, spec_nodes=None):
        if spec_nodes is None:
            spec_nodes = range(len(self.nodes))

        prob_bounds = self.emtpy_prob_bounds()
        for i in spec_nodes:
            node = self.nodes[i]
            _, pot_l_prob, _, pot_u_prob = node.interval_cdf(inp_int=inp_int,  point_int=out_int[:, i],
                                                             use_ibp=use_ibp, use_ibp_u=use_ibp_u)
            prob_bounds['l_probs'][i] = torch.max((pot_l_prob), torch.tensor(0.))
            prob_bounds['u_probs'][i] = torch.max((pot_u_prob), torch.tensor(0.))

        prob_bounds['l_prob'] = torch.prod(prob_bounds['l_probs'][spec_nodes])
        prob_bounds['u_prob'] = torch.prod(prob_bounds['u_probs'][spec_nodes])
        return prob_bounds


## -- Disc -------------------------------------------------------------------------------------------------------------
class Disc:
    def __init__(self, l, u, id=0, parent_id=None, sat=True):
        self.sat = sat
        self.int = torch.stack((l, u)) # \TODO change to interval..
        if self.sat:
            self.int = torch.clip(self.int, 0, torch.inf)

        self.id = id
        self.parent_id = parent_id
        self.children = None
        self.cond_expect = dict()
        self.free_expect = dict()

        self.l_probs = torch.ones(self.int.shape[1])
        self.u_probs = torch.ones(self.int.shape[1])

    def init_children_disc(self, layer: Layer, eps_create=1., full_space=False, ibp_bounds=None, direct=True, **kwargs):
        if full_space:
            interval = get_full_space(layer.act, layer.neurons)
            self.children = create_discs(layer, self, interval, 0.)
        else:
            if ibp_bounds is not None:
                tag = layer.layer_tag * 2 + 1
                self.children = create_discs(layer, self, ibp_bounds[tag], eps_create)
                self.children.refine_discs(layer=layer, disc_in=self, **kwargs)
            elif direct:
                self.children = layer.create_disc(disc_in=self, use_ibp=False, use_ibp_u=False, **kwargs)
            else:
                forward_step_disc(layer, self, use_ibp=False, use_ibp_u=False)
                A_L, b_L, A_U, b_U = bound.lb2ib(self.int, self.free_expect['A_L'].clone(), self.free_expect['b_L'].clone(),
                                           self.free_expect['A_U'].clone(), self.free_expect['b_U'].clone())
                self.children = create_discs(layer, self, torch.cat((b_L[None], b_U[None]), dim=0), eps_create)
                self.children.refine_discs(layer=layer, disc_in=self, **kwargs)

    def get_prob(self, layer, disc_in, spec_nodes=None):
        if spec_nodes is None:
            spec_nodes = range(len(layer.nodes))

        prob = layer.interval_cdf(inp_int=disc_in.int, out_int=self.int, use_ibp=True, use_ibp_u=True,
                                  spec_nodes=spec_nodes)

        self.l_probs[spec_nodes] = prob['l_probs'][spec_nodes]
        self.u_probs[spec_nodes] = prob['u_probs'][spec_nodes]

        self.l_prob = torch.prod(self.l_probs)
        self.u_prob = torch.prod(self.u_probs)

    def grow_node_int(self, idx2replace: int, layer: Layer, disc_in, eps_rel=0.2, eps_abs=None, threshold=0.999,
                      msg=False, max_iter=10): # \TODO change disc_in -> just inp_int
        if eps_abs is None:
            eps = eps_rel * (self.int[1, idx2replace] - self.int[0, idx2replace])
        else:
            eps = eps_abs

        eps_vec = torch.tensor([-eps, eps])

        k = 0
        while True:
            if self.l_probs[idx2replace] > threshold:
                reason = 'ITER {}'.format(k)
                break
            if k > max_iter:
                reason = 'ITER MAX'
                break

            if self.sat:
                self.int[:, idx2replace] = torch.clip(self.int[:, idx2replace] + eps_vec, 0., torch.inf)
            else:
                self.int[:, idx2replace] += eps_vec

            self.get_prob(layer, disc_in, spec_nodes=[idx2replace])
            k += 1

        if msg:
            print('Node [{}] - after {}: {:.6f}-{:.6f} - {}'.format(idx2replace, 'replace', self.l_probs[idx2replace],
                                                                    self.u_probs[idx2replace], reason))

    def shrink_node_int(self, idx2replace: int, layer: Layer, disc_in, left_side=True, right_side=True,
                        eps_rel=0.2, threshold=0.999, msg=False, max_iter=10): # \TODO change disc_in -> just inp_int
        k = 0
        while True:
            eps = eps_rel * (self.int[1, idx2replace] - self.int[0, idx2replace])
            eps_vec = torch.zeros(2)
            if left_side:
                eps_vec[0] = eps
            if right_side:
                eps_vec[1] = -eps

            self.int[:, idx2replace] += eps_vec
            self.get_prob(layer, disc_in, spec_nodes=[idx2replace])

            if self.l_probs[idx2replace] < threshold:
                self.int[:, idx2replace] += -eps_vec
                self.get_prob(layer, disc_in, spec_nodes=[idx2replace])
                reason = 'ITER {}'.format(k)
                break
            if k > max_iter:
                reason = 'ITER MAX'
                break

            k += 1

        if msg:
            print('Node [{}] - after {}: {:.6f}-{:.6f} - {}'.format(idx2replace, 'shrink', self.l_probs[idx2replace],
                                                                    self.u_probs[idx2replace], reason))

    def split(self, id_start, to_split=None):
        if to_split is None:
            to_split = torch.argmax(self.int[1] - self.int[0])

        eps_split = (self.int[1, to_split] - self.int[0, to_split]) / 3. # ratio never smaller than 2
        old_int = self.int[:, to_split].clone()

        int_ref_0 = self.int.clone()
        int_ref_0[0, to_split] = old_int[0]
        int_ref_0[1, to_split] = old_int[0] + eps_split
        if self.sat:
            int_ref_0[1, to_split] = torch.clip(int_ref_0[1, to_split], 0., torch.inf)

        ref_0 = Disc(int_ref_0[0], int_ref_0[1], id=id_start, parent_id=self.id, sat=self.sat)

        int_ref_1 = self.int.clone()
        int_ref_1[0, to_split] = old_int[1] - eps_split
        if self.sat:
            int_ref_1[0, to_split] = torch.clip(int_ref_1[0, to_split], 0, torch.inf)

        int_ref_1[1, to_split] = old_int[1]
        ref_1 = Disc(int_ref_1[0], int_ref_1[1], id=id_start+1, parent_id=self.id, sat=self.sat)

        int_ref_2 = self.int.clone()
        int_ref_2[0, to_split] = old_int[0] + eps_split
        if self.sat:
            int_ref_2[0, to_split] = torch.clip(int_ref_2[0, to_split], 0, torch.inf)

        int_ref_2[1, to_split] = old_int[1] - eps_split
        if self.sat:
            int_ref_2[1, to_split] = torch.clip(int_ref_2[1, to_split], 0, torch.inf)
        ref_2 = Disc(int_ref_2[0], int_ref_2[1], id=id_start+2, parent_id=self.id, sat=self.sat)

        return [ref_0, ref_1, ref_2]


class Discs:
    def __init__(self, *args, **kwargs):
        self.discs = dict()
        self.layer_tag = kwargs['layer_tag']
        self.__call__(*args, **kwargs)

    def __call__(self, discs: list, layer=None, disc_inp=None, prob_info=None, **kwargs):
        """
        prob_info: {id_disc: {l_probs: Tensor(out_dim disc), u_probs: Tensor(out dim disc}, spec_nodes: list()}
        """
        self.discs.update({disc.id: disc for disc in discs})

        if layer is not None and disc_inp is not None:
            if prob_info:
                for i in self.discs:
                    self.discs[i].l_probs = prob_info[i]['l_probs']
                    self.discs[i].u_probs = prob_info[i]['u_probs']
                    self.get_prob(layer, disc_inp, spec_nodes=prob_info[i]['spec_nodes'])
                    self.precision = prob_info[i]['precision']
            else:
                self.get_prob(layer, disc_inp)

    def get_kth_layer_children(self, k):
        if self.layer_tag == k:
            return self
        else:
            if len(self.discs) > 1:
                raise NotImplementedError('only implemented for discs consisting of a single region')
            else:
                if self.discs[0].children is None:
                    return None
                else:
                    return self.discs[0].children.get_kth_layer_children(k)

    def init_children(self, layer, **kwargs):
        if layer.layer_tag == self.layer_tag:
            for i in self.discs:
                self.discs[i].init_children_disc(layer=layer, **kwargs)
        else:
            for i in self.discs:
                self.discs[i].children.init_children(layer=layer, **kwargs)

    def print(self):
        for i in self.discs:
            print('region {}, prob lb: {:.4f}, prob ub: {:.4f}'.format(
                i, self.discs[i].l_prob, self.discs[i].u_prob))
            if self.discs[i].refinements is not None:
                self.discs[i].refinements.print()

    def get_prob(self, layer, disc_inp, spec_nodes=None):
        for i in self.discs:
            self.discs[i].get_prob(layer, disc_inp, spec_nodes=spec_nodes)
            # print('disc {}, probs: {:.4f}-{:.4f}'.format(self.discs[i].id, self.discs[i].l_prob,
            #                                              self.discs[i].u_prob))

    def refine_discs(self, split=False, grow=False, shrink=False, msg=True, eps_rel_grow=0.1, eps_rel_shrink=0.1,
                     threshold=0.999, **kwargs):
        # \TODO implement overlapping check
        # \TODO allow for multiple parents
        if len(self.discs) > 1:
            raise ImportError('{} TODO allow for more than 1 parent to replace and split'.format('split_discs'))
        else:
            j = 0

        if msg:
            print('Old probs {}: {:.4f}-{:.4f}'.format(j, self.discs[j].l_prob, self.discs[j].u_prob))

        grow_idxs = []
        shrink_idxs = []
        if grow and not shrink:
            grow_idxs = torch.where(self.discs[j].l_probs <= threshold)[0]
            shrink_idxs = grow_idxs
        elif grow:
            grow_idxs = torch.where(self.discs[j].l_probs <= threshold)[0]
        elif shrink:
            shrink_idxs = range(self.discs[j].int[0].shape[0])

        for i in grow_idxs:
            self.discs[j].grow_node_int(i, eps_rel=eps_rel_grow, msg=msg, threshold=threshold, **kwargs)

        for i in shrink_idxs:
            self.discs[j].shrink_node_int(i, right_side=False, left_side=True, eps_rel=eps_rel_shrink, msg=msg,
                                          threshold=threshold, **kwargs)
            self.discs[j].shrink_node_int(i, right_side=True, left_side=False, eps_rel=eps_rel_shrink, msg=msg,
                                          threshold=threshold, **kwargs)

        print('After Replace & Shrink: {:.6f}-{:.6f}'.format(self.discs[j].l_prob, self.discs[j].u_prob))

        if split:
            raise NotImplementedError


def get_full_space(act: str, neurons: int):
    if act == 'Linear' or act == 'PreReLU':
        l, u = -torch.ones(neurons) * torch.inf, torch.ones(neurons) * torch.inf
    elif act == 'Exp' or act == 'ReLU':
        l, u = torch.zeros(neurons), torch.ones(neurons) * torch.inf
    elif act == 'SoftMax':
        l, u = torch.zeros(neurons), torch.ones(neurons)
    else:
        raise NotImplementedError
    return torch.stack((l,u))


def create_discs(layer, disc_in, int, eps):
    # Options:
    # (1) relative eps
    # (2) if cov diagonal: eps = cov

    if layer.act == 'ReLU' or layer.act == 'Exp':
        sat = True
    elif layer.act == 'Linear' or layer.act == 'PreReLU':
        sat = False
    else:
        raise NotImplementedError

    if eps == 0.:
        discs = [Disc(int[0], int[1], id=0, sat=sat)]
    else:
        # # "diff of expected value" method:
        # offset = (lb_exp['b_U'] - lb_exp['b_L'])*eps

        ## Old cov based approach, only works for single input:
        # offset = eps*layer.W_std.flatten()

        # ## Old weighted cov method
        # ## weight covariance
        # offset = torch.einsum('ij,ij->i', layer.W_loc, layer.W_std) # new cov based method
        # ## normalize weighted covariance
        # offset = torch.einsum('i,i->i', torch.einsum('ij->i', layer.W_loc)**(-1), offset) * eps

        ## Variance method
        scale_bounds = torch.max(
            torch.sqrt(torch.einsum('ij,j->i', layer.W_std ** 2, disc_in.int[0] ** 2) + layer.b_std.flatten()**2),
            torch.sqrt(torch.einsum('ij,j->i', layer.W_std ** 2, disc_in.int[1] ** 2) + layer.b_std.flatten()**2))
        offset = scale_bounds * 3 * eps

        discs = [Disc(int[0] - offset, int[1] + offset, id=0, sat=sat)]
    return Discs(discs=discs, layer=layer, disc_inp=disc_in, layer_tag=layer.layer_tag+1)


# -- GENERALIZED APPROACH ----------------------------------------------------------------------------------------------
def forward_step_disc(layer: Layer, disc_inp: Disc, use_ibp: bool, use_ibp_u: bool):
    if disc_inp.children is None:
        out_int = get_full_space(layer.act, layer.neurons)
        disc_inp.free_expect = layer.cond_expect(inp_int=disc_inp.int, out_int=out_int, use_ibp=use_ibp,
                                                 use_ibp_u=use_ibp_u)
    else:
        disc_inp.cond_expect[layer.layer_tag+1] = dict()
        for i in disc_inp.children.discs:
            disc_inp.cond_expect[layer.layer_tag + 1][i] = layer.cond_expect(inp_int=disc_inp.int,
                                                                             out_int=disc_inp.children.discs[i].int,
                                                                             use_ibp=use_ibp, use_ibp_u=use_ibp_u)
            disc_inp.free_expect = {
                                    'A_L': torch.zeros(layer.W_loc.shape),
                                    'b_L': torch.zeros(layer.neurons),
                                    'A_U': disc_inp.cond_expect[layer.layer_tag + 1][i]['A_U'] * gamma,
                                    'b_U': disc_inp.cond_expect[layer.layer_tag + 1][i]['b_U'] * gamma
                                    }


def forward_step(layer: Layer, discs_inp: Discs, use_ibp: bool, use_ibp_u: bool):
    if layer.layer_tag == discs_inp.layer_tag:
        for i in discs_inp.discs:
            forward_step_disc(layer, discs_inp.discs[i], use_ibp=use_ibp, use_ibp_u=use_ibp_u)
    else:
        for i in discs_inp.discs:
            forward_step(layer, discs_inp.discs[i].children, use_ibp=use_ibp, use_ibp_u=use_ibp_u)


def backward_step_disc(low_layer_tag, upper_layer_tag, disc_inp, use_ibp=False):
    for i in disc_inp.children.discs:
        if low_layer_tag == 0:
            lbs_in = disc_inp.cond_expect[low_layer_tag + 1][i]
        else:
            lbs_in = {'A_L': disc_inp.cond_expect[low_layer_tag + 1][i]['A_L'],
                      'b_L': disc_inp.cond_expect[low_layer_tag + 1][i]['b_L'],
                      'A_U': disc_inp.free_expect['A_U'],
                      'b_U': disc_inp.free_expect['b_U']}

        if disc_inp.children.discs[i].children is None:
            lbs_out = disc_inp.children.discs[i].free_expect
        else:
            lbs_out = disc_inp.children.discs[i].cond_expect[upper_layer_tag]

        ## Linear Term
        lbs_lin_term = prop_lb(*lbs_in.values(), lbs_out['A_L'], lbs_out['b_L']*0., lbs_out['A_U'], lbs_out['b_U']*0.)

        ## Bias Term
        ibs_bias_part = {'b_L': lbs_out['b_L'],
                         'b_U': lbs_out['b_U']}

        lbs = {'A_L': lbs_lin_term['A_L'],
               'b_L': lbs_lin_term['b_L'] + ibs_bias_part['b_L'],
               'A_U': lbs_lin_term['A_U'],
               'b_U': lbs_lin_term['b_U'] + ibs_bias_part['b_U']}

        if upper_layer_tag not in disc_inp.cond_expect:
            disc_inp.cond_expect[upper_layer_tag] = lbs
        else:
            disc_inp.cond_expect[upper_layer_tag]['A_L'] += lbs['A_L']
            disc_inp.cond_expect[upper_layer_tag]['b_L'] += lbs['b_L']
            disc_inp.cond_expect[upper_layer_tag]['A_U'] += lbs['A_U']
            disc_inp.cond_expect[upper_layer_tag]['b_U'] += lbs['b_U']


def backward_step(low_layer_tag, upper_layer_tag, discs_inp, use_ibp=False):
    if low_layer_tag == discs_inp.layer_tag:
        for i in discs_inp.discs:
            backward_step_disc(low_layer_tag, upper_layer_tag, discs_inp.discs[i], use_ibp=use_ibp)
    else:
        for i in discs_inp.discs:
            backward_step(low_layer_tag, upper_layer_tag, discs_inp.discs[i].children, use_ibp=use_ibp)


def filter_children(discs_inp, last_layer_tag):
    if len(discs_inp.discs) > 1:
        # check if ref is avaiable:
        test_tag = list(discs_inp.discs.keys())[0]
        if last_layer_tag in discs_inp.discs[test_tag].cond_expect:
            vals = {i: discs_inp.discs[i].cond_expect[last_layer_tag]['b_U'] for i in discs_inp.discs}
            order = sorted(vals, key=vals.get)
            order.reverse()
            # l_probs = [discs_inp.discs[i].l_prob for i in discs_inp.discs]
            # u_probs = [discs_inp.discs[i].u_prob for i in discs_inp.discs]
            # l_left = torch.tensor(1.)
            u_left = torch.tensor(1.)
            for i in order:
                prob2fill = torch.max(discs_inp.discs[i].l_prob, torch.min(u_left, discs_inp.discs[i].u_prob))
                u_left -= prob2fill
                discs_inp.discs[i].u_prob = prob2fill
                discs_inp.discs[i].l_prob = torch.min(discs_inp.discs[i].l_prob, prob2fill)
                order.remove(i)
                if u_left <= 0:
                    break

            for i in order:
                discs_inp.discs.pop(i, None)
    else: # \TODO in future remove this one for filtereing of refinements over multiple layers
        for i in discs_inp.discs:
            if discs_inp.discs[i].children is not None:
                filter_children(discs_inp.discs[i].children, last_layer_tag)
