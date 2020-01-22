import tensorflow as tf
import os
import cv2
import numpy as np
import random
import math


class HWDataReader:
    def __init__(self, batch_size, hw_root, img_size=(64, 64)):
        self.hw_root = hw_root
        self.batch_size = batch_size
        self.label_tf = self.read_labels()
        self.img_size = img_size

    def read_labels(self):
        label_list = []
        self.n_classes = 3755
        for img_name in os.listdir(self.hw_root):
            img_path = self.hw_root + '/' + img_name
            label_list.append(img_path + ' ' + img_name)
        self.size = len(label_list)
        label_list_tf = tf.convert_to_tensor(label_list, dtype=tf.string)
        [label_tf] = tf.train.slice_input_producer([label_list_tf])
        return label_tf

    def __len__(self):
        return self.size

    def get_img_and_label(self):
        kvs = tf.string_split([self.label_tf], delimiter=' ')
        img_path = kvs.values[0]
        label_str = kvs.values[1]
        label_tf = label_str
        # label_tf = tf.cast(label_tf, dtype=tf.float32)
        img_data_tf = tf.read_file(img_path)
        img_tf = tf.image.decode_png(img_data_tf, channels=3)

        def resize_with_padding(img):
            im_h, im_w, im_c = img.shape
            wh = max(im_h, im_w)
            wh_img = np.ones((wh, wh, im_c), dtype=np.uint8) * 255
            left = int((wh - im_w) / 2)
            top = int((wh - im_h) / 2)
            wh_img[top:top + im_h, left:left + im_w] = img
            wh_img = cv2.resize(wh_img, self.img_size)
            return wh_img

        print(img_tf.shape)
        img_tf = tf.py_func(resize_with_padding, [img_tf], tf.uint8)
        print(img_tf.shape)
        w, h = self.img_size
        img_tf.set_shape((h, w, 3))
        print(img_tf.shape)
        return img_tf, label_tf


if __name__ == '__main__':
    dr = HWDataReader(batch_size=10, hw_root='/Users/hyl/Desktop/ML/train')
    img_tf, label_tf = dr.get()
    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # img = sess.run(data_tf)
        # cv2.imwrite('tmp/1111.jpg',img.astype(np.uint8) )
        img, label = sess.run([img_tf, label_tf])
        print(img.shape)
        for i in range(10):
            print(i, '-->', label[i])
            #image = img[i]
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('/Users/hyl/Desktop/temp/%d.jpg' % i, ((img[i] + 0.5) * 255).astype(np.uint8))
            #cv2.imwrite('tmp_pic/' + 'tmp/%d.jpg' % i, image)

        coord.request_stop()
        coord.join(threads)
