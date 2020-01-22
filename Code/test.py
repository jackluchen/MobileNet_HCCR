import tensorflow as tf
import numpy as np
import cv2
import os
from nets.mobilenet import MobileNetV2
from HWDataReader import HWDataReader


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def inference(input_tf, n_classes):
    net = MobileNetV2(n_classes=n_classes, depth_rate=1.0, is_training=False)
    output = net.build_graph(input_tf)
    return output


def main():
    # test datasets
    datasets = HWDataReader(batch_size=1, hw_root='/Users/hyl/Desktop/ML/test')
    img_tf, label_tf = datasets.get_img_and_label()
    img_tf = tf.cast(img_tf, tf.float32)
    label_tf = tf.expand_dims(tf.one_hot(label_tf, 3755), axis=0)
    # logits
    logits_tf = inference(tf.expand_dims(img_tf, axis=0), 3755)
    print(label_tf)
    print(logits_tf)
    # accuracy
    correct_pred = tf.equal(tf.argmax(logits_tf, 1), tf.argmax(label_tf, 1))
    accuracy_tf = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, 'model/model-160000')
        acc_sum = 0
        for i in range(len(datasets)):
            acc = sess.run(accuracy_tf)
            acc_sum = acc_sum + acc
            if i > 0 and i % 100 == 0:
                print('progress:%.2f%%, accuracy=%f' % (i * 100.0 / len(datasets), acc_sum * 1.0 / i))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
