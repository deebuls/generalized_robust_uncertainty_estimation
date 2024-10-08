import numpy as np
import tensorflow as tf
import time
import datetime
import os
import sys
import h5py
from pathlib import Path

import edl
import nll
from .util import normalize, gallery

class Likelihood:
    def __init__(self, model, opts, loss_type, dataset="", learning_rate=1e-3, tag="", save_files=True, noisy_data=False):

        self.model = model
        self.dataset = dataset
        if "depth" == dataset:
            self.l = opts['l']
            self.drop_prob = opts['drop_prob']
            self.mse = not opts['sigma']
            self.lam = opts['lam']

        self.loss_type = loss_type

        if "gaussian" == loss_type:
            self.loss_function = nll.losses.gaussian_nll_loss
        elif "laplace" == loss_type:
            self.loss_function = nll.losses.laplace_nll_loss
        elif "Generalized" == loss_type:
            self.loss_function = nll.losses.generalized_nll_loss
        elif "mse" == loss_type:
            self.loss_function = edl.losses.MSE

        self.optimizer = tf.optimizers.Adam(learning_rate)

        self.min_rmse = float('inf')
        self.min_nll = float('inf')
        self.min_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = None
        if noisy_data:
            noisy_data_str="noisy"
        else:
            noisy_data_str="clean"

        if save_files:
            self.save_dir = os.path.join('save','{}_{}_{}_{}_{}'.format(current_time, dataset, noisy_data_str, loss_type, learning_rate))
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

            train_log_dir = os.path.join('logs', '{}_{}_{}_{}_{}_train'.format(current_time, dataset, loss_type,learning_rate, tag))
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_log_dir = os.path.join('logs', '{}_{}_{}_{}_{}_val'.format(current_time, dataset, loss_type,learning_rate, tag))
            self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        print("Trainer : {} \t Loss Type: {} \t Learning rate: {} ".format(trainer, loss_type, learning_rate))

    @tf.function
    def run_train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.model(x, training=True) #forward pass
            if "Generalized" == self.loss_type:
                mu, alpha, beta  = tf.split(y_hat, 2, axis=-1)
                loss = self.loss_function(mu, y, alpha, beta)
            else:
                mu, var = tf.split(y_hat, 2, axis=-1)
                loss = self.loss_function(mu, y, var)

        grads = tape.gradient(loss, self.model.variables) #compute gradient
        self.optimizer.apply_gradients(zip(grads, self.model.variables))

        return loss, y_hat

    @tf.function
    def evaluate(self, x, y):
        pred = self.model(x, training=False) #forward pass
        if "Generalized" == self.loss_type:
            mu, alpha, beta  = tf.split(y_hat, 2, axis=-1)
            loss = self.loss_function(mu, y, alpha, beta)
        else:
            mu, var = tf.split(y_hat, 2, axis=-1)
            loss = self.loss_function(mu, y, var)

        rmse = edl.losses.RMSE(y, mu)
	
        if "depth" == self.dataset:
            ###################### Check this later
            #tau = self.l**2 * (1-self.drop_prob) / (2. * self.lam) # https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html
            #var = tau**-1
            #nll = edl.losses.Gaussian_NLL(y, mu, tf.sqrt(var))
            ###################### Check this later should we just give sigma directly?
            nll = edl.losses.Gaussian_NLL(y, mu, tf.sqrt(var))
        else:
            nll = loss

        return mu, var, loss, rmse, nll

    def save_train_summary(self, loss, x, y, y_hat):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', tf.reduce_mean(loss), step=self.iter)
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, y_hat)), step=self.iter)
            idx = np.random.choice(int(tf.shape(x)[0]), 9)
            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(y_hat,idx).numpy())], max_outputs=1, step=self.iter)

    def save_val_summary(self, loss, x, y, mu, var):
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.iter)
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, mu)), step=self.iter)
            idx = np.random.choice(int(tf.shape(x)[0]), 9)
            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(mu,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_var", [gallery(normalize(tf.gather(var,idx)).numpy())], max_outputs=1, step=self.iter)
 
    def get_batch(self, x, y, batch_size):
        idx = np.random.choice(x.shape[0], batch_size, replace=False)
        if isinstance(x, tf.Tensor):
            x_ = x[idx,...]
            y_ = y[idx,...]
        elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
            idx = np.sort(idx)
            x_ = x[idx,...]
            y_ = y[idx,...]
 
            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
            y_divisor = 255. if y_.dtype == np.uint8 else 1.0
 
            x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
            y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)
        else:
            print("unknown dataset type {} {}".format(type(x), type(y)))
        return x_, y_
 

#    def get_batch(self, x, y, batch_size):
#        idx = np.random.choice(x.shape[0], batch_size, replace=False)
#        if isinstance(x, tf.Tensor):
#            x_ = x[idx,...]
#            y_ = y[idx,...]
#        elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
#            idx = np.sort(idx)
#            x_ = x[idx,...]
#            y_ = y[idx,...]
#
#            if "depth" == self.dataset:
#                # Adding noise to labels
#                #idx_height = np.random.randint(1, y.shape[1], size=(batch_size,200))
#                #idx_width = np.random.randint(1, y.shape[2], size=(batch_size,200))
#                #y_[:, idx_height, idx_width] = 0
#                for i in range(10):
#                    first_value_height = np.random.choice(y.shape[1]-20)
#                    last_value_height = first_value_height + np.random.randint(1,20)
#                    first_value_width = np.random.choice(y.shape[2]-20)
#                    last_value_width = first_value_width + np.random.randint(1,20)
#                    
#                    if (0 == i):
#                        idx_height = np.random.randint(first_value_height, last_value_height, size=(batch_size,20))
#                        idx_width = np.random.randint(first_value_width, last_value_width, size=(batch_size,20))
#                    else:
#                        idx_height = np.append( idx_height, np.random.randint(first_value_height, last_value_height, size=(batch_size,20)) , axis=1)
#                        idx_width = np.append( idx_width, np.random.randint(first_value_width, last_value_width, size=(batch_size,20)) , axis=1)
#                
#                y_[:, idx_height, idx_width] = 0
#
#            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
#            y_divisor = 255. if y_.dtype == np.uint8 else 1.0
#
#            x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
#            y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)
#        else:
#            print("unknown dataset type {} {}".format(type(x), type(y)))
#        return x_, y_


    def save(self, name):
        if self.save_dir:
            self.model.save(os.path.join(self.save_dir, "{}.h5".format(name)))

    def train(self, x_train, y_train, x_test, y_test, y_scale, batch_size=128, iters=10000, verbose=True):
        tic = time.time()
        for self.iter in range(iters):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss, y_hat = self.run_train_step(x_input_batch, y_input_batch)

            #if self.iter % 10 == 0:
            #    self.save_train_summary(loss, x_input_batch, y_input_batch, y_hat)

            if self.iter % 100 == 0:
                x_test_batch, y_test_batch = self.get_batch(x_test, y_test, min(100, x_test.shape[0]))
                mu, var, vloss, rmse, nll = self.evaluate(x_test_batch, y_test_batch)
                nll += np.log(y_scale[0,0])
                rmse *= y_scale[0,0]

                self.save_val_summary(vloss, x_test_batch, y_test_batch, mu, var)

                if rmse.numpy() < self.min_rmse:
                    self.min_rmse = rmse.numpy()
                    self.save(f"model_rmse_{self.iter}")

                if nll.numpy() < self.min_nll:
                    self.min_nll = nll.numpy()
                    self.save(f"model_nll_{self.iter}")

                if vloss.numpy() < self.min_vloss:
                    self.min_vloss = vloss.numpy()
                    self.save(f"model_vloss_{self.iter}")

                if verbose: print("[{}] \t RMSE: {:.4f} \t NLL: {:.4f} \t train_loss: {:.4f} \t t: {:.2f} sec".format(self.iter, self.min_rmse, self.min_nll, loss, time.time()-tic))
                tic = time.time()


        return self.model, self.min_rmse, self.min_nll
