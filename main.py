# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys, random, cv2, os, shutil
import mnist_data

reload(sys)
sys.setdefaultencoding('utf-8')

mnist_data.prepare()

def train():
    IMAGE_SIZE = 28
    IMAGE_CHANNELS = 1
    BATCH_SIZE = 32

    # 1. build model
    images = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNELS), name="images")
    labels = tf.placeholder(dtype=tf.int32, shape=(None), name="labels")

    with tf.variable_scope('Model'):
        W = tf.Variable(tf.truncated_normal(shape=[IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNELS, 10], stddev=0.1), name='weight')
        B = tf.Variable(tf.truncated_normal(shape=[10], stddev=0.1), name='bias')
        output = tf.matmul(images, W)+B
    
    predict = tf.argmax(output, 1, name="predict")
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output))
    opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 2. load data
    train_data = {}
    with open("./mnist/train-labels.csv") as f:
        raw_data = f.readlines()
        for parsed in raw_data:
            parsed = parsed.strip().split(",")
            train_data[parsed[0]] = parsed[1]

    image_paths = train_data.keys()
    random.shuffle(image_paths)

    # 3. train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    g_data_idx = 0
    iteration = 0
    data_images = np.zeros([BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNELS])
    data_labels = np.zeros([BATCH_SIZE])

    if os.path.exists("./assets"):
        shutil.rmtree("./assets", ignore_errors=True)
    builder = tf.saved_model.builder.SavedModelBuilder("./assets")
        
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        while True:
            data_idx = 0
            while True:
                if g_data_idx >= len(image_paths):
                    builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING])
                    builder.save()
                    exit()
                if data_idx == BATCH_SIZE:
                    break
                image_path = image_paths[g_data_idx]
                image = cv2.imread(os.path.join("mnist", image_path), cv2.IMREAD_GRAYSCALE)/255.
                data_images[data_idx] = np.reshape(image, [IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNELS])
                data_labels[data_idx] = int(train_data[image_path])
                g_data_idx += 1
                data_idx += 1
                   
            outputs = sess.run([opt, loss, predict], feed_dict={images:data_images, labels:data_labels})
            loss_value = outputs[1]
            predict_value = outputs[2]
            if iteration % 100 == 0:
                print "%03d : %0.2f" % (iteration, loss_value)
            iteration += 1

train()
