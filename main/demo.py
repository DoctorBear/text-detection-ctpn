# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import math
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_string('test_data_path', '../data/demo/', '')
tf.app.flags.DEFINE_string('output_path', '../data/res/', '')
tf.app.flags.DEFINE_string('gpu', '-1', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    # os.walk will recursively go through all nodes(directories) in this path
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


# fixme this method should be move to class uimg after module merging
# this method only pads to the bottom side and right side while another method pads in all border
def pad_right_and_below(img, new_height, new_width, padding_val=255):
    height, width = img.shape[:2]
    assert height <= new_height and width <= new_width
    top, left, bottom, right = 0, 0, new_height - height, new_width-width
    out_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_val)
    return out_img


# todo Replace resize below with cut to avoid lose of pixels
def pad_image(img, filter_size=600, stride=400):
    '''
    Cut just like filter in cnn, size of filter and stride is needed first!
    The overlap is needed to avoid character cut!
    If one side is not equal to 600 + 400 * n(n>=0), pad it too.
    :param img: picure
    :param filter_size: size of images after cut
    :param stride: just like stride in cnn
    :return images and their offset(height and width)
    '''
    height, width = img.shape
    new_height = int(math.ceil((height-filter_size)/stride)*stride) + filter_size if height > filter_size-1 else filter_size
    new_width = math.ceil((width-filter_size)/stride)*stride + filter_size if width > filter_size-1 else filter_size
    re_im = pad_right_and_below(img, new_height=new_height, new_width=new_width)
    return re_im


# todo merge parts model detects
def merge_text_part(parts):
    pass



# todo Replace resize with cut to avoid lose of pixels
def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    # 在较大边不超过1200的情况下，按照较小边到600缩放,较小边=600，600<=较大边<1200
    # 否则按照较大变1200缩放，即较大边=1200,较小边<=600
    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)

    # 当需要缩小时，返回False
    # if im_scale < 1:
    #     return False

    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    # 上取整两边至整除16
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    print(new_h, end=':')
    print(new_w)

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imencode('.jpg', re_im)[1].tofile('data/res/1.jpg')
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def main(argv=None):
    # todo 重新读一遍这里，并且重构，把训练部分单独拎出来，然后结合cut而非resize
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, (rh, rw) = resize_image(im)

                # resize_time = time.time()
                # print("Resize cost time: {:.2f}s".format(resize_time - start))

                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})
                # net_time = time.time()
                # print("Net cost time: {:.2f}s".format(net_time - resize_time))

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                # proposal_time = time.time()
                # print("Proposal cost time: {:.2f}s".format(proposal_time - net_time))

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                print("All cost time: {:.2f}s".format(cost_time))

                h1, w1, c1 = im.shape
                panel = np.ones((h1, w1, c1), np.uint8)*255

                actual_boxes = []

                for i, box in enumerate(boxes):
                    print(box[0])
                    print(box[1])
                    print(box[2])
                    print(box[5])

                    # actual = [box[0] // rw, box[1] // rh, box[2] // rw + 1, box[5] // rh + 1]
                    # todo not sure whether there should be padding, padding may let the noise in,
                    # todo but without padding, part of the valuable info may be lost
                    actual = [box[0] // rw - 5, box[1] // rh - 2, box[2] // rw + 6, box[5] // rh + 2]
                    actual[0] = int(actual[0]) if actual[0] > -1 else 0
                    actual[1] = int(actual[1]) if actual[1] > -1 else 0
                    actual[2] = int(actual[2]) if actual[2] < w1 else w1-1
                    actual[3] = int(actual[3]) if actual[3] < h1 else h1-1

                    panel[actual[1]:actual[3]+1, actual[0]:actual[2]+1, :] = im[actual[1]:actual[3]+1, actual[0]:actual[2]+1, :]

                    actual_boxes.append(actual)
                    # cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                    #               thickness=2)
                # img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), panel[:, :, ::-1])

                with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                          "w") as f:
                    for i, box in enumerate(actual_boxes):
                        line = ",".join(str(box[k]) for k in range(4))
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)


if __name__ == '__main__':
    tf.app.run()
