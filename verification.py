import torch
import numpy as np

import bound
import modules


def process_class(args, lbs, label_pred, logits_int, inp_int, lbs_x2logits, prob_main_disc):
    # Naive IBP
    diff = lbs['b_U'] - lbs['b_L'][label_pred]
    diff[label_pred] = 0.
    robust = torch.all(diff <= 0.)

    # NEW PROP
    fl = False
    if not robust:
        non_robust_label_idxs = np.where(diff > 0.)[0]
        for label_idx in non_robust_label_idxs:
            left_term = torch.exp(logits_int[1, label_idx]) - torch.exp(logits_int[0, label_pred])
            right_term = sum(torch.exp(logits_int[1]) * (1 - 1 / prob_main_disc))

            if left_term <= right_term:
                robust = True
            else:
                robust = False
                break

    return diff, robust, fl


def verf(x_center: torch.tensor, epsilon, args, ibp=False, true_preds=None, return_lbs_x2logits=False):
    if torch.all(x_center - epsilon) <= 0:
        raise ValueError('only support regions in positive half plain')

    layers = []
    if args.example in ['noisy_sine_1', 'noisy_sine_2', 'powerplant', 'kin8nm']:
        ## create layers
        for k in range(args.layers):
            layers.append(modules.Layer(layer_tag=k, imp_tag=k*2, act='ReLU'))
        layers.append(modules.Layer(layer_tag=args.layers, imp_tag=args.layers*2, act='Linear'))

        ## create discretization
        # discretize x
        disc_x = modules.Disc((x_center-epsilon), (x_center+epsilon))
        discs_x = modules.Discs([disc_x], layer_tag=0)

        # discretize z_{K-1}
        for k in range(args.layers-1):
            discs_x.init_children(layers[k], args=args)
        discs_x.init_children(layers[-2], full_space=True, args=args)

        ## forward props
        for k in range(args.layers, -1, -1):
            modules.forward_step(layers[k], discs_x, use_ibp=False, use_ibp_u=False)

        ## backward props
        for k in range(args.layers-1, -1, -1):
            modules.backward_step(k, args.layers+1, discs_x)

        if not ibp:
            return discs_x.discs[0].cond_expect[args.layers+1]
        else:
            ibs = bound.lb2ib(discs_x.discs[0].int, *discs_x.discs[0].cond_expect[args.layers + 1].values())
            return {'b_L': ibs[1], 'b_U': ibs[3]}
    elif args.example in ['mnist', 'fashion_mnist']:
        ## create layers
        for k in range(args.layers):
            layers.append(modules.Layer(layer_tag=k, imp_tag=k * 2, act='ReLU'))
        layers.append(modules.Layer(layer_tag=args.layers, imp_tag=args.layers * 2, act='Linear'))
        layers.append(modules.Layer(layer_tag=args.layers+1, imp_tag=None, act='SoftMax', true_preds=true_preds))

        ## create discretization
        # discretize x
        disc_x = modules.Disc((x_center-epsilon), (x_center+epsilon))
        discs_x = modules.Discs([disc_x], layer_tag=0)

        # discretize z_{K-1}
        for k in range(args.layers):
            discs_x.init_children(layers[k], args=args)

        if return_lbs_x2logits:
            ## forward props
            for k in range(args.layers, -1, -1):
                modules.forward_step(layers[k], discs_x, use_ibp=False, use_ibp_u=False)

            ## backward props
            for k in range(args.layers - 1, -1, -1):
                modules.backward_step(k, args.layers + 1, discs_x)

        discs_x.init_children(layers[args.layers], args=args)

        if ibp:
            ## forward props
            modules.forward_step(layers[-1], discs_x, use_ibp=False, use_ibp_u=False)
            lbs = discs_x.get_kth_layer_children(args.layers + 1).discs[0].free_expect
            logits_int = discs_x.get_kth_layer_children(args.layers + 1).discs[0].int
            prob_main_disc_ll = disc_x.children.precision ** args.layers
            if return_lbs_x2logits:
                return lbs, logits_int, discs_x.discs[0].cond_expect[args.layers + 1], prob_main_disc_ll
            else:
                return lbs, logits_int, None, prob_main_disc_ll
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError