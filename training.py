import examples.noisy_sine_1.noisy_sine_1 as ns
from support import parse_arguments
from get_bnn import get_bnn


if __name__ == '__main__':
    args = parse_arguments()
    _ = get_bnn(args)

