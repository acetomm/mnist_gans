import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True, validation_size=0)

noise_len = 100

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
z = tf.placeholder(tf.float32, [None, noise_len])


def d_net(ix, iy):
    conv1 = tf.layers.conv2d(ix, filters=8, kernel_size=[3, 3], strides=[1, 1],
                             activation=tf.nn.relu, padding='same', name='conv1') # try tf.nn.sigmoid

    iy_tiled = tf.tile(iy, [1, 28*28])
    iy_reshaped = tf.reshape(iy_tiled, [-1, 28, 28, 10])
    conv1_comp = tf.concat([conv1, iy_reshaped], axis=3)

    maxp1 = tf.layers.max_pooling2d(conv1_comp, pool_size=[2, 2], strides=[2, 2], padding='same')

    conv2 = tf.layers.conv2d(maxp1, filters=16, kernel_size=[3, 3], strides=[2, 2],
                             activation=tf.nn.relu, padding='same', name='conv2')
    maxp2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], padding='same')

    d_out = tf.layers.dense(tf.reshape(maxp2, [-1, 4 * 4 * 16]), activation=tf.nn.sigmoid, units=1, name='dense1')
    return d_out


def g_net(iz, iy):
    iz_comp = tf.concat([iz, iy], axis=1)
    g_input = tf.reshape(iz_comp, [-1, 1, 1, 100 + 10])
    '''
    conv1 = tf.layers.conv2d_transpose(inputs=g_input, filters=16, kernel_size=[3, 3], strides=[1, 1], padding='valid')
    conv1_a = tf.nn.relu(conv1)

    conv2 = tf.layers.conv2d_transpose(inputs=conv1_a, filters=8, kernel_size=[5, 5], strides=[2, 2], padding='same')
    conv2_a = tf.nn.relu(conv2)

    conv3 = tf.layers.conv2d_transpose(inputs=conv2_a, filters=8, kernel_size=[5, 5], strides=[2, 2], padding='same')
    conv3_a = tf.nn.relu(conv3)

    conv4 = tf.layers.conv2d_transpose(inputs=conv3_a, filters=1, kernel_size=[5, 5], strides=[2, 2], padding='same')
    g_out = tf.image.resize_images(tf.nn.sigmoid(conv4), size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    '''

    conv1 = tf.layers.conv2d_transpose(inputs=g_input, filters=8, kernel_size=[7, 7], strides=[1, 1], padding='valid')
    conv1_a = tf.nn.relu(conv1)

    conv2 = tf.layers.conv2d_transpose(inputs=conv1_a, filters=12, kernel_size=[5, 5], strides=[2, 2], padding='same')
    conv2_a = tf.nn.relu(conv2)

    conv3 = tf.layers.conv2d_transpose(inputs=conv2_a, filters=1, kernel_size=[5, 5], strides=[2, 2], padding='same')
    g_out = tf.nn.sigmoid(conv3)

    return g_out


def sample_z(m, n):
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
    plt.savefig('conv_cgan_results/samples/g_{0}_{1}.png'.format(e, b))
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

    adv_loss, disc_loss = 0, 0
    try:
        for e in range(epochs):
            for b in range(batches):
                if adv_loss < (1.2*disc_loss):
                    disc_k = 30
                else:
                    disc_k = 1
                for _ in range(disc_k):
                    batch_data = mnist.train.next_batch(batch_size)
                    batch_x = batch_data[0].reshape([-1, 28, 28, 1])
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
    plt.clf()
    plt.close('all')
    losses = np.array(losses)
    plt.plot(losses[:, 0], 'b-')
    plt.plot(losses[:, 1], 'r-')
    plt.savefig('conv_cgan_results/loss.png')