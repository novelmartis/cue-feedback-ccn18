# cue-feedback-ccn18
The codes associated with our CCN'18 paper - https://ccneuro.org/2018/proceedings/1044.pdf

Requirements:
1. Tensorflow 1.9 with Python 2 (with numpy and scipy)
2. Fashion-MNIST data: Download the files from https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion and place them under the folder 'fMNIST_data' at the same level as the models_new folder.

Procedure:
Choose a representational capacity (RC+/RC-) and run files which mention the appropriate tag. Manipulate the neural capacity within the files with 'n_hl'
1. Run the 'train1_' file first. This will train the OPS with the required neural and representational capacity.
2. Run the 'train2_' file second. There are two steps therein. First set 'cond_h' to 1 to train the probe network, and then to 7 to train the cue network.
3. Run the associated ipynb notebook sequentially. It will output the performance of the network on the various cases (probe-only, uninformative cue, informative cue)

Contact: Mail me at sushrut.thorat94@gmail.com for any clarifications.

