import matplotlib 
matplotlib.use('Agg') 
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
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
    
    parser.add_argument("--mem_file", default = '../../texture_nets/abstact_art_swA.txt', 
                        help = "File with memorability mesurments")
    parser.add_argument("--mem_file_external", default = '../../texture_nets/abstact_art_external_swA.txt', 
                        help = "File with memorability mesurments")
    parser.add_argument("--observed_part_of_samples", default = 1,
              type = float, help = "Fraction of seed that is used for training")
    parser.add_argument("--content_img_dir", default = '../../lamem/images/',
                        help = "directory with content images")
    parser.add_argument("--random_state", default = 0, type = int,
                        help = "Seed for train/test/val split and generating and selecting observed_part_of_samples")
    parser.add_argument("--train_size", default = 0.8,  type = float, 
                        help = "Part of data that is used for training")
    parser.add_argument("--val_size", default = 0.1,  type = float, 
                        help = "Part of data that is used for early stoping")
    parser.add_argument("--test_size", default = 0.1,  type = float, 
                        help = "Part of data that is used for testing")
    parser.add_argument("--model", default = 'alex-swA.npy', help = "Trained network")
    parser.add_argument("--network", default = 'alex', 
                        help = "File with network definition, should define build_model transform_train, transform_test, transform_val")
    parser.add_argument("--batch_size", default = 64, type = int,
                        help = "Size of the batch")
    parser.add_argument("--result_folder", default = "results_swA",
                        help = "Folder where results will be stored")
      
    options = parser.parse_args()

    return options.__dict__

def load_data(options, mem_file):
    def read_images(image_names, directory):
        images = []
        for fname in image_names:
            image = skimage.io.imread(os.path.join(directory, fname))
            if len(image.shape) == 2:
                image = gray2rgb(image)
            images.append(image)
        return images
    
    df = pd.read_csv(mem_file)
    pivot = pd.pivot(df['content_img_name'], df['seed_img_name'], df['diff_mem_score'])
    image_names = np.array(pivot.index)
    X = read_images(image_names, options['content_img_dir'])
    Y = np.array(pivot)

    return X, Y, image_names, pivot.columns, df

def split_data(options, X, Y, image_names):
    x_train, x_val, y_train, y_val, names_train, names_val = train_test_split(X, Y, image_names, test_size = options['val_size'],
                                                  random_state = options['random_state'])
    x_train, x_test, y_train, y_test, names_train, names_test = train_test_split(x_train, y_train, names_train,
                                                        test_size = options['test_size'] / (1 - options['val_size']),
                                                        random_state = options['random_state'])
    
    np.random.seed(options['random_state'])
    mask_train = np.random.binomial(1, p = options['observed_part_of_samples'], size = y_train.shape)
    mask_test = np.ones_like(y_test)
    mask_val = np.ones_like(y_val)

    return x_train, x_val, x_test, y_train, y_val, y_test, mask_train, mask_val, mask_test, names_train, names_val, names_test


def define_net(options, num_seeds):
    import train
    net = train.create_net(options, num_seeds)
    input_image = T.tensor4('input', dtype = 'float32')
    out_score = T.matrix('out', dtype = 'float32')
    mask = T.matrix('weight', dtype = 'float32')

    def acc_loss(ground, predicted, mask):
        return (mask * T.gt(ground * predicted,  0)).sum() / mask.sum()

    def square_loss(ground, predicted, mask):
        return (mask * ((ground - predicted) ** 2)).sum() / mask.sum()

    y_det = lasagne.layers.get_output(net['out'], input_image, deterministic = True)

    loss_test = square_loss(out_score, y_det, mask).mean()

    loss_acc_test = acc_loss(out_score, y_det, mask)
    
    loss_fun = theano.function([input_image, out_score, mask], [loss_test, loss_acc_test, y_det], allow_input_downcast=True)
    return net, loss_fun 


def pass_dataset(options, x, y, mask, fun, dataset):
    net_file = __import__(options['network'])
    transform_func_dict = {"train" : net_file.transform_train, 'test' : net_file.transform_test, 'val' : net_file.transform_val}
    def iterate_minibatches(x, y, mask, batchsize, dataset = 'train', shuffle = False):
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
    predictions = []
    for inputs, targets, batch_mask in iterate_minibatches(x, y, mask, options['batch_size'], dataset = dataset):
        loss, loss_acc, pr = fun(inputs, targets, batch_mask)
        err += loss
        err_acc += loss_acc
        batches += 1
        predictions.append(pr)
    return err / batches, err_acc / batches, np.vstack(predictions)


def test(options, net, loss_fun, x_test, y_test, mask_test):
    sq, acc, predictions = pass_dataset(options, x_test, y_test, mask_test, loss_fun, 'test')
    log_string = "%f,%f" % (sq, acc)
    print (log_string)
    return predictions

def compute_baseline(y_train, y_test, mask_train):
    predictions = ((y_train * mask_train).sum(axis = 0) / mask_train.sum(axis = 0)).reshape((1, y_train.shape[1]))
    predictions = np.broadcast_to(predictions, y_test.shape)
    square_gap = ((y_test - predictions) ** 2).mean()
    accuracy = (y_test * predictions > 0).mean()
    print ("%f,%f" % (square_gap, accuracy))
    return predictions

 
def save_frame(options, results, image_names, seed_names, postfix, prefix = None, baseline = False):
    frame = pd.DataFrame(results, index = pd.Index(image_names, name = 'image_id'), columns = seed_names)
    if prefix is None:
        prefix = (options['network'] + '_' if not baseline else '') + str(int(options['observed_part_of_samples'] * 100)) + '_'
    frame.to_csv(os.path.join(options['result_folder'], prefix + postfix + '.csv'))

def save_scores_df(options, image_names, df, file_name):
    scores_df = df[df['content_img_name'].isin(image_names)][['content_img_name', 'in_mem_score']].drop_duplicates() 
    scores_df.to_csv(os.path.join(options['result_folder'], file_name), index = False)

def main():
    options = get_cmd_options()
    X, Y, image_names, seed_names, df = load_data(options, options['mem_file']) 
    _, Y_external, _, _, df_external = load_data(options, options['mem_file_external'])
    _, _, x_test, y_train, _, y_test, mask_train, _, mask_test, _, _, names_test = split_data(options, X, Y, image_names)
    _, _, _, _, _, y_test_external, _, _, _, _, _, _ = split_data(options, X, Y_external, image_names)
   

    save_scores_df(options, names_test, df, 'mem_scores_ex.csv')
    save_scores_df(options, names_test, df_external, 'mem_scores_in.csv')

    save_frame(options, y_test_external, names_test, seed_names, 'gt_ex', '')
    save_frame(options, y_test, names_test, seed_names, 'gt_in', '')    

    print ("Baseline external test")
    predictions = compute_baseline(y_train, y_test_external, mask_train)
    save_frame(options, predictions, names_test, seed_names, 'base_ex', baseline = True)
    print ("Baseline internal test")
    predictions = compute_baseline(y_train, y_test, mask_train)
    save_frame(options, predictions, names_test, seed_names, 'base_in', baseline = True)
    net, loss_fun = define_net(options, y_train.shape[1])
    best_params = np.load(options['model'])
    lasagne.layers.set_all_param_values(net['out'], best_params)
    print ("Network external test")
    predictions = test(options, net, loss_fun, x_test, y_test_external, mask_test)
    save_frame(options, predictions, names_test, seed_names, 'nn_ex')   
    print ("Network internal test")
    predictions = test(options, net, loss_fun, x_test, y_test, mask_test)
    save_frame(options, predictions, names_test, seed_names, 'nn_in')
   
if __name__ == "__main__":
    main()    
