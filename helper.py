#encoding=utf-8
import tensorflow as tf
from matplotlib.image import imread
import numpy as np
from PIL import Image
import scipy.misc as misc
import os
import signal
from sklearn.utils import shuffle
from datetime import datetime
import os


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def normalize(samples,ax=2):
    """
    并且灰度化: 从三色通道 -> 单色通道     省内存 + 加快训练速度
    (R + G + B) / 3
    将图片从 0 ~ 255 线性映射到 -1.0 ~ +1.0
    @samples: numpy array
    """
    a = np.add.reduce(samples, keepdims=True, axis=ax)  # shape (图片数，图片高*图片宽，通道数)
    a = a // 3
    return a
    # return samples / 128.0 - 1.0

class read_datasets:
    """flag=o:nothing to be done
            1:store and load .npy data file
            2:TFRecords file
    """
    def __init__(self,flag=0,num_epoch=2):
        self.dir="/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/"
        self.flag=flag
        self.filename = self.dir + 'train_dataset_bytes_0404.tfrecords'
        self.num_epoch=num_epoch

    def save_train_dataSets(self):
        image_path=self.dir+"train/"
        if self.flag==1:
            image_list=[]
            for img_name in os.listdir(image_path):  # 当前目录下所有图片，一次一张
                image_array=Image.open(image_path + img_name)
                image_list.append(image_array.getdata())
            image_list=image_list[:4096]
            # print (np.shape(image_list))#4096 40000 3
            image_list=normalize(image_list)#4096,40000#有问题 数据一直为0
            # image_list=list(np.reshape(image_list,[-1,120000]))
            image_list = np.reshape(image_list, [-1, 40000])
            label_list=np.loadtxt(self.dir+"train_labels")[0:4096]
            print(np.shape(image_list))
            print(np.shape(label_list))
            train_dataSets=np.hstack((image_list,label_list))
            train_dataSets=shuffle(np.array(train_dataSets))
            if os.path.exists(self.dir+"../../train_dataSets_with_shuffle_1_channels.npy"):
                os.system("rm /home/naruto/PycharmProjects/knifey_spoony_demo/train_dataSets_with_shuffle_1_channels.npy")
            np.save(self.dir+"../../train_dataSets_with_shuffle_1_channels.npy", train_dataSets)
            # train_dataSets = image_list.append(label_list)
            del image_list,label_list
            print("the data has saved")

    def load_train_dataSets(self):
        if self.flag==1:
            train_dataSets=np.load(self.dir+"../../train_dataSets_with_shuffle_1_channels.npy","r+")
            train_dataSets=shuffle(np.array(train_dataSets))
            print ("the samples is 1 channel ,and the value is between [0,255]")
            return train_dataSets[:,:40000],train_dataSets[:,40000:]

    def convert2TFRecord(self):
        if self.flag==2:
            i=0
            classes = {'forky', 'knifey', 'spoony'}  #
            writer = tf.python_io.TFRecordWriter(self.filename)  # 要生成的文件
            for index, name in enumerate(classes):
                class_path = self.dir + name+'/'
                for img_name in os.listdir(class_path):
                    if os.path.splitext(img_name)[1] == '.jpg':#以为该文件将爱中还有test文件夹，且下面的文件是soft link
                    # if not os.path.isdir(img_name):
                        img_path = class_path + img_name  # 每一个图片的地址
                        img = Image.open(img_path)
                        # print(np.shape(img))#(200, 200, 3)
                        # img_raw=np.array(img.getdata())
                        img = img.resize((200, 200))#，分辨率，和shape不一样
                        img_raw = img.tobytes()  # 将图片转化为二进制格式
                        # img_raw = img_raw.tostrings() #mnist数据可这样处理
                        #onehot
                        # label=[0]*3
                        # label[index]=1
                        # label = np.array(label).tobytes()
                        # label=bytes(label)
                        # label=label.tobytes()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            #有三种数据类型float int64（int） byte（array，string tobyte）
                            }))  # example对象对label和image数据进行封装

                        writer.write(example.SerializeToString())  # 序列化为字符串
                        i+=1
            print("the datasets  is:{} examples".format(i))
            writer.close()
            print("TFrecords file is done!,the path is {}".format(self.filename))
    def convert2TFRecord_test(self):
        if self.flag==2:
            self.dir = "/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/test/"

            self.filename="/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/test_dataset_0426.tfrecords"
            i=0
            classes = {'forky', 'knifey', 'spoony'}  #
            writer = tf.python_io.TFRecordWriter(self.filename)  # 要生成的文件
            for index, name in enumerate(classes):
                class_path = self.dir + name+'/'
                for img_name in os.listdir(class_path):
                    if os.path.splitext(img_name)[1] == '.jpg':#以为该文件将爱中还有test文件夹，且下面的文件是soft link
                    # if not os.path.isdir(img_name):
                        img_path = class_path + img_name  # 每一个图片的地址
                        img = Image.open(img_path)
                        # print(np.shape(img))#(200, 200, 3)
                        # img_raw=np.array(img.getdata())
                        img = img.resize((200, 200))#，分辨率，和shape不一样
                        img_raw = img.tobytes()  # 将图片转化为二进制格式
                        # img_raw = img_raw.tostrings() #mnist数据可这样处理
                        #onehot
                        # label=[0]*3
                        # label[index]=1
                        # label = np.array(label).tobytes()
                        # label=bytes(label)
                        # label=label.tobytes()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            #有三种数据类型float int64（int） byte（array，string tobyte）
                            }))  # example对象对label和image数据进行封装

                        writer.write(example.SerializeToString())  # 序列化为字符串
                        i+=1
            print("the datasets  is:{} examples".format(i))
            writer.close()
            print("TFrecords file is done!,the path is {}".format(self.filename))


    def read_and_decode(self):
        #tf 将读取数据可计算分成两个进程，文件队列和内存队列，减少GPU等待的开销，可以传递多个文件list
        # FIFOQueue和RandomShuffleQueue
        filename_queue = tf.train.string_input_producer(
            [self.filename], num_epochs=self.num_epoch)#epoch默认为none
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),#和编码对应
                                           })  # 将image数据和label取出来
        #返回单条记录tensor 区别parse_example
        img = tf.decode_raw(features['image_raw'],tf.uint8)

        img = tf.reshape(img, [200, 200, 3])  # reshape 根据图像和具体模型确定
        img=tf.cast(img, tf.float32) * (1. / 255) - 0.5 #中心化有助于快速训练，且必须为float，mnist demo中to_int64在1.1中存在bug

        label = tf.cast(features['label'],tf.int32)
        print("reading TFRecords file is done!")
        return img, label

    # def loaddata():
    #     cache_file_path=os.path.join(data_dir,"knifey_spoony.cache")
    #     if os.path.exists(cache_file_path):
    #         with open(cache_file_path,"rb") as f:
    #             obj=pickle.load(cache_file_path)
    #             print ("")
    #     else:
    #         obj=dataSets(data_dir)
    #         with open(cache_file_path,mode="wb") as f:
    #             pickle.dump(obj,f)
    #     return obj

    # with tf.Session() as sess:  # 开始一个会话
    #     init_op = tf.initialize_all_variables()
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(20):
    #         example, l = sess.run([image, label])  # 在会话中取出image和label
    #         img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
    #         img.save(cwd + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
    #         print(example, l)
    #     coord.request_stop()
    #     coord.join(threads)
    # image = cv2.imread(images[i])
    # image = cv2.resize(image, (208, 208))
    # b, g, r = cv2.split(image)
    # rgb_image = cv2.merge([r, g, b])  # this is suitable
    # image_raw = rgb_image.tostring()


