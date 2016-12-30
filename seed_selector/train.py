import matplotlib 
matplotlib.use('Agg') 
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")
import theano
import theano.tensor as T
import lasagne
from sklearn.model_selection import train_test_split
import time
import skimage
import skimage.io
from skimage.color import gray2rgb


def get_cmd_options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mem_file", default = '/unreliable/aliaksandr/memorability/texture_nets/abstract_art_sw10.txt', 
                        help = "File with memorability mesurments")
    parser.add_argument("--content_img_dir", default = '/unreliable/aliaksandr/memorability/lamem/lamem/images/',
                        help = "directory with content images")
    parser.add_argument("--observed_part_of_samples", default = 1, type = float,
                        help = "Fraction of seed that is used for training")
    parser.add_argument("--random_state", default = 0, type = int,
                        help = "Seed for train/test/val split and generating and selecting observed_part_of_samples")
    parser.add_argument("--train_size", default = 0.8,  type = float, 
                        help = "Part of data that is used for training")
    parser.add_argument("--val_size", default = 0.1,  type = float, 
                        help = "Part of data that is used for early stoping")
    parser.add_argument("--test_size", default = 0.1,  type = float, 
                        help = "Part of data that is used for testing")
    parser.add_argument("--trainable_layers", default = 'fc7,fc6,conv5_3,conv5_2,conv5_1',
                        help = "Which layers of default network finetune")
    parser.add_argument("--learning_rate", default = 1e-3, type = float)
    parser.add_argument("--learning_method", default = 'nesterov_momentum')
    parser.add_argument("--output_model", default = 'model_script.npy', help = "Trained network")
    parser.add_argument("--network", default = 'vgg16', 
                        help = "File with network definition, should define build_model transform_train, transform_test, transform_val")
    parser.add_argument("--num_epochs", default = 1000, type = int,
                        help = "Number of iterations throught train set")
    parser.add_argument("--batch_size", default = 50, type = int,
                        help = "Size of the batch")
  
    options = parser.parse_args()

    return options.__dict__

def load_data(options):
    def read_images(image_names, directory):
        images = []
        for fname in image_names:
            image = skimage.io.imread(os.path.join(directory, fname))
            if len(image.shape) == 2:
                image = gray2rgb(image)
            images.append(image)
        return images
    
    df = pd.read_csv(options['mem_file'])
    pivot = pd.pivot(df['content_img_name'], df['seed_img_name'], df['diff_mem_score'])
    image_names = np.array(pivot.index)
    X = read_images(image_names, options['content_img_dir'])
    Y = np.array(pivot)
    return X, Y

def split_data(options, X, Y):
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = options['val_size'],
                                                  random_state = options['random_state'])
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, 
                                                        test_size = options['test_size'] / (1 - options['val_size']),
                                                        random_state = options['random_state'])
    
    np.random.seed(options['random_state'])
    mask_train = np.random.binomial(1, p = options['observed_part_of_samples'], size = y_train.shape)
    mask_test = np.ones_like(y_test)
    mask_val = np.ones_like(y_val)

    return x_train, x_val, x_test, y_train, y_val, y_test, mask_train, mask_val, mask_test

def pass_dataset(options, x, y, mask, fun, dataset):
    net_file = __import__(options['network'])
    transform_func_dict = {"train" : net_file.transform_train, 'test' : net_file.transform_test, 'val' : net_file.transform_val}
    def iterate_minibatches(x, y, mask, batchsize, dataset = 'train', shuffle = True):
        perm = list(range(len(x)))
        if shuffle:        
            np.random.shuffle(perm)

        for begin in range(0, len(perm), batchsize):
            end = min(begin + batchsize, len(perm))
            x_batch = transform_func_dict[dataset](np.take(x, perm[begin:end]))
            y_batch = y[perm[begin:end]]
            mask_batch = mask[perm[begin:end]]
            yield (x_batch, y_batch, mask_batch)
    err = 0
    err_acc = 0
    batches = 0
    for inputs, targets, batch_mask in iterate_minibatches(x, y, mask, options['batch_size'], dataset = dataset):
        loss, loss_acc = fun(inputs, targets, batch_mask)
        err += loss
        err_acc += loss_acc
        batches += 1
    return err / batches, err_acc / batches

def create_net(options, num_classes):
    net_file = __import__(options['network'])
    net = net_file.build_model()
    net['out'] = lasagne.layers.DenseLayer(net['fc7_dropout'], num_units = num_classes, nonlinearity = None, name = 'out')
    return net


def train(options, x_train, x_val, y_train, y_val, mask_train, mask_val):
    net = create_net(options, y_train.shape[1])
    input_image = T.tensor4('input', dtype = 'float32')
    out_score = T.matrix('out', dtype = 'float32')
    mask = T.matrix('weight', dtype = 'float32')

    def acc_loss(ground, predicted, mask):
        return (mask * T.gt(ground * predicted,  0)).sum() / mask.sum()

    def square_loss(ground, predicted, mask):
        return (mask * ((ground - predicted) ** 2)).sum() / mask.sum()

    y_rand = lasagne.layers.get_output(net['out'], input_image, deterministic = False)
    y_det = lasagne.layers.get_output(net['out'], input_image, deterministic = True)

    loss_train = square_loss(out_score, y_rand, mask).mean()
    loss_test = square_loss(out_score, y_det, mask).mean()

    loss_acc_train = acc_loss(out_score, y_rand, mask)
    loss_acc_test = acc_loss(out_score, y_det, mask)
    
    all_weights = net['out'].get_params()
    for layer_name in options['trainable_layers'].split(','):
        if layer_name in net:
            all_weights += net[layer_name].get_params()
        else:
            print ("No layer %s in network" % (layer_name, ))

    print ("Params to optimize:")
    print (all_weights)

    updates_sgd = lasagne.updates.__dict__[options['learning_method']](loss_train, all_weights, options['learning_rate'])

    print ('Compiling...')
    train_fun = theano.function([input_image, out_score, mask], [loss_train, loss_acc_train], updates = updates_sgd,
                                allow_input_downcast=True)
    loss_fun = theano.function([input_image, out_score, mask], [loss_test, loss_acc_test], allow_input_downcast=True)
    
    
    print ("Training...")
    best_params = None

    num_epochs = options['num_epochs']

    best_val_score = (1e10, 1e10)
    begin_of_learning = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        err_train = pass_dataset(options, x_train, y_train, mask_train, train_fun, 'train')
        err_val = pass_dataset(options, x_val, y_val, mask_val, loss_fun, 'val')
	

        if best_val_score > err_val:
            best_val_score = err_val
            best_params = lasagne.layers.get_all_param_values(net['out'])       
            print ("Saving params...")
            np.save(options['output_model'], best_params)


        log_string = (
            "Epoch {} of {} took {:.3f}s\n".format(epoch + 1, num_epochs, time.time() - start_time) + 
            "  training loss (in-iteration):\t\t{}\n".format(err_train) +
            "  validation loss (in-iteration):\t\t{}\n".format(err_val) +
            "  best val loss (in-iteration):\t\t{}\n".format(best_val_score)
        )
        print (log_string)

    log_string = "Elapsed Time: {:.3f}s".format(time.time() - begin_of_learning)
    print (log_string)
    
    lasagne.layers.set_all_param_values(net['out'], best_params)
    return net, loss_fun 
    
def test(options, net, loss_fun, x_test, y_test, mask_test):
    test_err = pass_dataset(options, x_test, y_test, mask_test, loss_fun, 'test')
    log_string = "Test error: squre_error %f , accuracy_error %f" % test_err
    print (log_string)   
    
    
def main():
    options = get_cmd_options()
    X, Y = load_data(options)
    x_train, x_val, x_test, y_train, y_val, y_test, mask_train, mask_val, mask_test = split_data(options, X, Y)
    net, loss_fun = train(options, x_train, x_val, y_train, y_val, mask_train, mask_val)
    test(options, net, loss_fun, x_test, y_test, mask_test)
    
if __name__ == "__main__":
    main()    
