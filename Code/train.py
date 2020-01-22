import tensorflow as tf
import numpy as np
import cv2
from nets.mobilenet import MobileNetV2
from HWDataReader import HWDataReader
import os

BATCH_SIZE = 256
MAX_STEPS = 95000
RETRAIN_STEP = None
FLAG = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def inference(input_tf, n_classes):
    net = MobileNetV2(n_classes=n_classes, depth_rate=1.0, is_training=True)
    output = net.build_graph(input_tf)
    return output


# def load_model(sess,step):
#     if step is None:
#         return 0
#     else:
#         saver = tf.train.Saver()
#         saver.restore(sess,'model/model-%d'%step)
#         print('load model from model/model-%d'%step)
#         return step


def load_model(sess):
    ckpt = tf.train.get_checkpoint_state('model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('load model from ', ckpt.model_checkpoint_path)
        last_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        last_step = int(last_step)
        return last_step
    else:
        return 0


def get_loss(logits, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss)
    return loss


def main(dir_):
    start_step = 0
    if not RETRAIN_STEP is None:
        start_step = RETRAIN_STEP
    global_step = tf.Variable(start_step, trainable=False, name='gloabel_step')
    datasets = HWDataReader(batch_size=BATCH_SIZE, hw_root=dir_)
    img_batch, onehot_batch = datasets.get()
    # logits
    logits_tf = inference(img_batch, datasets.n_classes)
    # loss
    loss_tf = get_loss(logits_tf, onehot_batch)

    # accuracy
    correct_pred = tf.equal(tf.argmax(logits_tf, 1), tf.argmax(onehot_batch, 1))
    accuracy_tf = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # learning rate
    decay_steps = int(len(datasets) / BATCH_SIZE * 2.5)
    lr_tf = tf.train.exponential_decay(0.001, global_step, decay_steps, 0.94, staircase=True)

    # optimizer
    opt = tf.train.AdamOptimizer(lr_tf)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss_tf, global_step=global_step)
    saver = tf.train.Saver()
    with tf.name_scope('scalar') as scope:
        tf.summary.scalar('loss', loss_tf)
        tf.summary.scalar('accuracy', accuracy_tf)
        tf.summary.scalar('lr', lr_tf)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        summary_op = tf.summary.merge(summaries)
        summary_writer = tf.summary.FileWriter('./model', graph=tf.get_default_graph())
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.get_variable_scope().reuse_variables()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # load_model(sess, RETRAIN_STEP)
        start_step = load_model(sess)

        if FLAG == 1:
            t_max_steps = MAX_STEPS * 2
        else:
            t_max_steps = MAX_STEPS
        for i in range(start_step + 1, t_max_steps+1):
            _, loss, accuracy, summary, lr = sess.run([train_op, loss_tf, accuracy_tf, summary_op, lr_tf])
            summary_writer.add_summary(summary, i)

            print('step:%d  loss=%f   acuracy=%f  lr=%f' % (i, loss, accuracy, lr))

            if i % 1000 == 0:
                saver.save(sess, 'model/model-%d' % i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main('/dev/shm/ML/train')
