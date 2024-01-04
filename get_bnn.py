import examples.noisy_sine_1.noisy_sine_1 as ns1
import examples.noisy_sine_2.noisy_sine_2 as ns2
import examples.mnist.mnist as mn
import examples.fashion_mnist.fashion_mnist as fmn
import examples.cifar.cifar as cf
import examples.half_moons.half_moons as hm
import examples.powerplant.powerplant as pw
import examples.concrete.concrete as cn
import examples.kin8nm.kin8nm as kn


# if 'win' in platform:
#     mpl.use('TkAgg')  # or can use 'TkAgg'

def get_data_folder(args):
    if args.example == 'kin8nm':
        return kn.PATH_DATA
    elif args.example == 'mnist':
        return mn.PATH_DATA
    elif args.example == 'fashion_mnist':
        return fmn.PATH_DATA
    elif args.example == 'noisy_sine_2':
        return ns2.PATH_DATA
    else:
        raise NotImplementedError

def get_bnn(args):
    if args.example == 'noisy_sine_1':
        return ns1.get_bnn(args)
    elif args.example == 'noisy_sine_2':
        return ns2.get_bnn(args)
    elif args.example == 'mnist':
        return mn.get_bnn(args)
    elif args.example == 'fashion_mnist':
        return fmn.get_bnn(args)
    elif args.example == 'cifar':
        return cf.get_bnn(args)
    elif args.example == 'half_moons':
        return hm.get_bnn(args)
    elif args.example == 'concrete':
        return cn.get_bnn(args)
    elif args.example == 'powerplant':
        return pw.get_bnn(args)
    elif args.example == 'kin8nm':
        return kn.get_bnn(args)
    else:
        raise NotImplementedError


def test(args, **kwargs):
    if args.example == 'noisy_sine_1':
        ns1.test(args=args, **kwargs)
    elif args.example == 'noisy_sine_2':
        ns2.test(args=args, **kwargs)
    elif args.example == 'mnist':
        mn.test(**kwargs)
    elif args.example == 'fashion_mnist':
        fmn.test(**kwargs)
    elif args.example == 'cifar':
        cf.test(**kwargs)
    elif args.example == 'half_moons':
        hm.test(**kwargs)
    elif args.example == 'concrete':
        cn.test(**kwargs)
    elif args.example == 'powerplant':
        pw.test(**kwargs)
    elif args.example == 'kin8nm':
        kn.test(**kwargs)
    else:
        raise NotImplementedError
