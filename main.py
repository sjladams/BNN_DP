import torch
import os

from verification import verf, process_class
import support
import bound
import configuration as config
from get_bnn import get_bnn, test, get_data_folder
import numpy as np
import time

LINUX = False

if __name__ == '__main__':
    args = support.parse_arguments()

    net, guide, test_loader = get_bnn(args)

    ##-- Classification stuff ---------------------------------------------------------------------------------------------------
    if args.example in ['mnist', 'fashion_mnist']:
        N = args.nr_images
        summary = {'true': torch.zeros(N),
                   'pred': {'probs': torch.zeros(N, 10), 'label': torch.zeros(N)},
                   'ibp': {'diff': torch.zeros(N, 10), 'robust': torch.zeros(N)},
                   'fgsm': {'epsilon': torch.zeros(N, 1), 'robust': torch.zeros(N)},
                   'formal': {'diff': torch.zeros(N, 10), 'robust': torch.zeros(N),
                              'bounds_logits': torch.zeros(N, 2, 10), 'time': torch.zeros(N),
                              'epsilon': torch.zeros(N), 'fl': torch.zeros(N)},
                   }

        ## Import former data if available
        folder_init_data = f"{get_data_folder(args)}/backup summaries"
        file_init_data = 'summary_{}_bnn_hid={}_arch={}_{}_N={}_find_eps'.format(
            args.example, args.width, args.layers, args.trained_by, N)
        if not args.fixed_epsilon and os.path.exists(f"{folder_init_data}/{file_init_data}.pickle"):
            print('INCREASING EPS')
            summary['use_fgsm'] = False
            init_epss = support.pickle_load(f"{folder_init_data}/{file_init_data}")['formal']['epsilon'].clone()
            init_epss[init_epss == 0.] = 1e-4
        else:
            summary['use_fgsm'] = True
            init_epss = torch.ones(N) * 1e-6

        offset = 0
        for image_idx, (x_data, label) in enumerate(test_loader.dataset[offset:N+offset]):
            summary['true'][image_idx] = label.argmax()

            prob_pred = net.predict_mean(guide, x_data, num_samples=20)
            label_pred = torch.argmax(prob_pred)
            summary['pred']['probs'][image_idx] = prob_pred
            summary['pred']['label'][image_idx] = label_pred

            if label_pred != label.argmax():
                continue

            if args.fixed_epsilon:
                summary['formal']['epsilon'][image_idx] = args.epsilon
                ## Statistical verification
                # IBP
                ibp_bounds = net.ibp(guide, x_data, args.epsilon, n_samples=50)

                summary['ibp']['diff'][image_idx] = ibp_bounds['probs'][1] - ibp_bounds['probs'][0][label_pred]
                summary['ibp']['diff'][image_idx, label_pred] = 0.
                summary['ibp']['robust'][image_idx] = torch.all(summary['ibp']['diff'][image_idx] <= 0.)

                # FGSM attack
                robust, probs_min = net.fgsm_attack(guide, x_data, label, args.epsilon, n_samples=50)
                summary['fgsm']['robust'][image_idx] = robust

                ## Formal Verification
                # Naive IBP softmax
                start_time = time.time()
                lbs, logits_int, lbs_x2logits, prob_main_disc = verf(x_center=x_data.flatten(), epsilon=args.epsilon,
                                                                     args=args, ibp=True, true_preds=label,
                                                                     return_lbs_x2logits=False)

                inp_int = torch.stack((x_data.flatten() - args.epsilon,
                                       x_data.flatten() + args.epsilon)).clip(0, torch.inf)
                diff, robust, fl = process_class(args, lbs, label_pred, logits_int, inp_int, lbs_x2logits,
                                                 prob_main_disc)

                summary['formal']['diff'][image_idx] = diff
                summary['formal']['robust'][image_idx] = robust
                summary['formal']['fl'][image_idx] = fl
                summary['formal']['bounds_logits'][image_idx] = logits_int

                summary['formal']['time'][image_idx] = time.time() - start_time
                print("Image {} \n IBP {} \n FGSM {} \n Formal {}".format(image_idx,
                                                                          summary['ibp']['robust'][image_idx],
                                                                          summary['fgsm']['robust'][image_idx],
                                                                          summary['formal']['robust'][image_idx]))
            else:
                epsilon = init_epss[image_idx]

                if summary['use_fgsm']:
                    ## Use FGSM to initialize epsilon at the statistical maximum possible epsilon
                    increase_options = [0.1, 0.05, 0.01, 0.005, 0.001]

                    fgsm_eps_step = 1
                    while fgsm_eps_step < 30:
                        robust, probs_min = net.fgsm_attack(guide, x_data, label, epsilon, n_samples=50)

                        if robust:
                            epsilon += increase_options[0]
                        else:
                            epsilon -= increase_options[0]
                            if len(increase_options) > 1:
                                increase_options.remove(increase_options[0])
                                epsilon += increase_options[0]
                            else:
                                break

                        if epsilon <= 0:
                            break

                        fgsm_eps_step += 1

                    summary['fgsm']['robust'][image_idx] = robust
                    summary['fgsm']['epsilon'][image_idx] = epsilon

                ## Formal Verification
                formal_eps_step = 0
                while formal_eps_step < 20:
                    start_time = time.time()

                    try:
                        lbs, logits_int, lbs_x2logits, prob_main_disc = verf(x_center=x_data.flatten(), epsilon=epsilon,
                                                                             args=args, ibp=True, true_preds=label,
                                                                             return_lbs_x2logits=True)
                    except:
                        break

                    inp_int = torch.stack((x_data.flatten() - epsilon,
                                           x_data.flatten() + epsilon)).clip(0, torch.inf)
                    diff, robust, fl = process_class(args, lbs, label_pred, logits_int, inp_int, lbs_x2logits,
                                                     prob_main_disc)

                    if robust:
                        summary['formal']['diff'][image_idx] = diff
                        summary['formal']['robust'][image_idx] = robust
                        summary['formal']['fl'][image_idx] = fl
                        summary['formal']['bounds_logits'][image_idx] = logits_int
                        summary['formal']['epsilon'][image_idx] = epsilon

                    if summary['use_fgsm']: # Decreasing
                        if robust:
                            break
                        else:
                            epsilon *= 0.5
                    else: # Increasing
                        if robust:
                            epsilon *= 1 + args.epsilon_step_size
                        else:
                            break

                    formal_eps_step += 1

                summary['formal']['time'][image_idx] = time.time() - start_time

                print("Image {} \n Epsilon robust {:.4f}, fgsm {:.4f}".format(image_idx,
                                                                              float(summary['formal']['epsilon'][
                                                                                        image_idx]),
                                                                              float(summary['fgsm']['epsilon'][
                                                                                        image_idx])))

        if args.fixed_epsilon:
            support.pickle_dump(summary, '{}/summary_{}_bnn_hid={}_arch={}_{}_eps={}_N={}'.format(
                get_data_folder(args), args.example, args.width, args.layers, args.trained_by, args.epsilon, N))
        else:
            file_data = 'summary_{}_bnn_hid={}_arch={}_{}_N={}_after_find_eps'.format(
                args.example, args.width, args.layers, args.trained_by, N)
            if summary['use_fgsm']:
                support.pickle_dump(summary, f"{get_data_folder(args)}/{file_data}")
            else:
                support.pickle_dump(summary, f"{get_data_folder(args)}/{file_data}")

    # -- Regression stuff ----------------------------------------------------------------------------------------------
    elif args.example in ['noisy_sine_1', 'noisy_sine_2', 'powerplant', 'kin8nm']:
        # ibp_bounds = net.ibp(guide, x_center, epsilon, n_samples=100)
        if args.example == 'noisy_sine_1':
            x_center = torch.tensor(config.CONFIG_T[args.example][args.layers]['x_center'])
            epsilon = torch.tensor(config.CONFIG_T[args.example][args.layers]['epsilon'])

            lbs = verf(x_center, epsilon, args)

            test(args=args, guide=guide, net=net, lin_bound_info={'x_center': x_center, 'epsilon': epsilon,
                                                                  'lbs': {'after': lbs}})
        elif args.example == 'noisy_sine_2':
            ## Average Robustness
            # Generate points if not available
            start_time = time.time()
            if os.path.exists(f"{get_data_folder(args)}/test_info_{args.layers}_{args.example}.pickle"):
                test_info = support.pickle_load(f"{get_data_folder(args)}/test_info_{args.layers}_{args.example}")
            else:
                if args.layers == 1:
                    test_info = {'x_centers': np.random.uniform(-1, 1, (100, 2)).astype(np.float32), 'epsilon': 0.01}
                    support.pickle_dump(test_info, f"{get_data_folder(args)}/test_info_{args.layers}_{args.example}")
                else:
                    test_info = {'x_centers': np.random.uniform(-1, 1, (100, 2)).astype(np.float32), 'epsilon': 0.001}
                    support.pickle_dump(test_info, f"{get_data_folder(args)}/test_info_{args.layers}_{args.example}")

            ## get stds
            x_centers_stds = net.predict_dist(guide, torch.from_numpy(test_info['x_centers']), num_samples=100).std(0).squeeze()
            support.pickle_dump(x_centers_stds, f'{get_data_folder(args)}/std_hid={args.width}_arch={args.layers}')

            deltas = []
            deltas_pre = []
            for i, x_center in enumerate(test_info['x_centers']):
                try:
                    ibs = verf(torch.from_numpy(x_center), test_info['epsilon'], args, ibp=True)
                    deltas.append(ibs['b_U'] - ibs['b_L'])
                except:
                    deltas.append(torch.tensor(torch.inf)[None])

                print(f"finnished {i} / {len(test_info['x_centers'])} data point")

            support.pickle_dump({'delta': torch.cat(deltas), 'avg_delta': torch.mean(torch.cat(deltas)),
                                 'total_time': time.time() - start_time},
                                f"{get_data_folder(args)}/test_results_hid={args.width}_arch={args.layers}")

            # x_center = torch.tensor(config.CONFIG_T[args.example][args.layers]['x_center'])
            # epsilon = torch.tensor(config.CONFIG_T[args.example][args.layers]['epsilon'])
            #
            # lbs = verf(x_center, epsilon, args)
            #
            # dim2plot = 0
            # test(args=args, guide=guide, net=net, view=3, dim=dim2plot, loc=x_center[dim2plot],
            #      lin_bound_info={'x_center': x_center,
            #                      'epsilon': epsilon,
            #                      'A_U_bound': A_U,
            #                      'b_U_bound': b_U[0],
            #                      'A_L_bound': A_L,
            #                      'b_L_bound': b_L[0]})
        elif args.example == 'powerplant' or args.example == 'kin8nm':
            ## Average Robustness
            # Generate points if not available
            start_time = time.time()
            if os.path.exists(f"{get_data_folder(args)}/test_info_{args.layers}.pickle"):
                test_info = support.pickle_load(f"{get_data_folder(args)}/test_info_{args.layers}")
            else:
                if args.layers == 1:
                    test_info = {'x_centers': (torch.cat([data_point[0] for data_point in test_loader])[:100] + \
                                              torch.randn(100, 8) * 0.001).detach().numpy(), 'epsilon': 0.01}
                    support.pickle_dump(test_info, f"{get_data_folder(args)}/test_info_{args.layers}")
                else:
                    test_info = {'x_centers': (torch.cat([data_point[0] for data_point in test_loader])[:100] + \
                                              torch.randn(100, 8) * 0.001).detach().numpy(), 'epsilon': 0.001}
                    support.pickle_dump(test_info, f"{get_data_folder(args)}/test_info_{args.layers}")


            ## get stds
            x_centers_stds = net.predict_dist(guide, torch.from_numpy(test_info['x_centers']), num_samples=100).std(0).squeeze()
            support.pickle_dump(x_centers_stds, f'{get_data_folder(args)}/std_hid={args.width}_arch={args.layers}')


            deltas = []
            deltas_pre = []
            for i, x_center in enumerate(test_info['x_centers']):
                try:
                    ibs = verf(torch.from_numpy(x_center), test_info['epsilon'], args, ibp=True)
                    deltas.append(ibs['b_U'] - ibs['b_L'])
                except:
                    deltas.append(torch.tensor(torch.inf)[None])

                print(f"finnished {i} / {len(test_info['x_centers'])} data point")



            support.pickle_dump({'delta': torch.cat(deltas), 'avg_delta': torch.mean(torch.cat(deltas)),
                                 'total_time': time.time() - start_time},
                                f"{get_data_folder(args)}/test_results_hid={args.width}_arch={args.layers}")

    print('stop')