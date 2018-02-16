import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True, validation_size=0)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
z = tf.placeholder(tf.float32, [None, 100])


def d_net(ix, iy):
    d_input = tf.concat([ix, iy], axis=1)
    d1 = tf.layers.dense(d_input, units=128,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         name='d1')
    d2 = tf.layers.dense(d1, units=1,
                         activation=tf.nn.sigmoid,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         name='d2')
    return d2


def g_net(iz, iy):
    g_input = tf.concat([iz, iy], axis=1)
    d1 = tf.layers.dense(g_input, units=128,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         name='d1')
    d2 = tf.layers.dense(d1, units=784,
                         activation=tf.nn.sigmoid,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         name='d2')
    return d2


def sample_z(m, n):
    #return np.random.uniform(0., 1., size=(m, n)) # don't work1
    return np.random.uniform(-1., 1., size=(m, n))


def generate_imgs(s, g, n, e, b):
    r, c = 2, 5
    noise = n(10, 100)
    gimg = s.run(g, feed_dict={z: noise, y: np.eye(10, 10)})
    g_i = 0
    fig, axes = plt.subplots(nrows=r, ncols=c) # , sharex=True, sharey=True
    for i in range(r):
        for j in range(c):
            fake = gimg[g_i].reshape((28, 28))
            axes[i, j].imshow(fake, cmap='gray')
            axes[i, j].axis('off')
            g_i += 1
    plt.savefig('dense_cgan_results/samples/g_{0}_{1}.png'.format(e, b))
    plt.clf()
    plt.close('all')


with tf.variable_scope('g'):
    g_out = g_net(z, y)

with tf.variable_scope('d') as scope:
    d_real = d_net(x, y)
    scope.reuse_variables()
    d_fake = d_net(g_out, y)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))

gan_vars = tf.trainable_variables()
d_params = [v for v in gan_vars if v.name.startswith('d/')]
g_params = [v for v in gan_vars if v.name.startswith('g/')]

opt_d = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_params)
opt_g = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_params)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 20
    batch_size = 100
    batches = mnist.train.num_examples // batch_size
    losses = []

    disc_k = 1
    sample_each = 100

    try:
        for e in range(epochs):
            for b in range(batches):
                for _ in range(disc_k):
                    batch_data = mnist.train.next_batch(batch_size)
                    batch_x = batch_data[0]
                    batch_y = batch_data[1].astype(np.float32)
                    batch_noise = sample_z(batch_size, 100)
                    disc_loss, _ = sess.run([d_loss, opt_d], feed_dict={x: batch_x, z: batch_noise, y: batch_y})

                batch_noise = sample_z(batch_size, 100)
                batch_y = mnist.train.next_batch(batch_size)[1].astype(np.float32)
                adv_loss, _ = sess.run([g_loss, opt_g], feed_dict={z: batch_noise, y: batch_y})

                print('epoch {0}/{1} batch {2}/{3} discriminator loss {4:.4f} adversarial loss {5:.4f}'.format(
                    e + 1, epochs, b + 1, batches,
                    disc_loss, adv_loss
                ))
                losses.append((disc_loss, adv_loss))
                if b % sample_each == 0:
                    generate_imgs(sess, g_out, sample_z, e + 1, b + 1)
    except KeyboardInterrupt:
        pass
    losses = np.array(losses)
    plt.plot(losses[:, 0], 'b-')
    plt.plot(losses[:, 1], 'r-')
    plt.savefig('dense_cgan_results/loss.png')