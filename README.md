In this repository you will find all of the code and data to run and replicate the results of our paper:


## BNN-DP: Robustness Certification of Bayesian Neural Networks via Stochastic Dynamic Programming

Prior to running the code in this repository we recommend you run the following command in order to ensure that all of the correct dependencies are used

```
virtualenv -p /usr/bin/python3.8 virt
source virt/bin/activate
pip install -r requirements.txt
```

This will activate the virtual environment for the project. From there, you can run the main.py file providing the arguments to the parser. Use the --help to command to view the options, and description of the arguments


Examples:

main.py --example noisy_sine_1 --layers 1 --width 256
main.py --example noisy_sine_1 --layers 2 --width 64

main.py --example mnist --layers 1 --width 128 --trained_by vogn
main.py --example mnist --layers 2 --width 64 --trained_by vogn

