#encoding=utf-8
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from imp import reload
from tensorflow.python import debug as tf_debug#
from sklearn.utils import shuffle
from helper import DelayedKeyboardInterrupt,read_datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("nvidia-smi | grep vpython3 | awk '{print $3}' | xargs kill -9")
os.system("lsof -i:6006 | grep tensorboa | awk '{print $2}' | xargs kill -9")
os.system("export TERM=linux ; export TERMINFO=/etc/terminfo")

# 读取tfrecords one_hot=false
# read_datasets(2).convert2TFRecord_test()

# debug = True

iterations=1000
batch_size = 128
learning_rate = 0.01#和初始化的权重很重要，直接关系你预测起始值
train_epochs=100
test_epochs=1
init_stddev=0.1
display_step=10

samples_channels = 3
one_hot_num = 3
height = width = 200
tarin_samples_num=4170

layer1_output_num = 8
layer2_output_num = 16
layer3_output_num = 32
layer4_output_num = 64
strides = [1, 1, 1, 1]
ksize = [1, 2, 2, 1]
fsize = 3

cost_list = []
corr_rate_list = []

output_dir="/home/naruto/PycharmProjects/knifey_spoony_demo/data/output"
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=init_stddev))#很重要
#session一定要包含在这个Graph中
with tf.Graph().as_default():
    is_training = tf.placeholder(dtype=tf.bool, shape=())
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()
    # writer = tf.summary.FileWriter(output_dir)
    # merge = tf.summary.merge_all()
    train_images_, train_labels_ = read_datasets(num_epoch=train_epochs).read_and_decode()
    test_images_ , test_labels_  = read_datasets(num_epoch=test_epochs).read_and_decode()
    # train_images_：[200, 200, 3]
    # train_labels_:() 都是tensor
    # print(tf.Tensor.get_shape(train_images_))
    # Image.fromarray(np.reshape(np.matrix(train_images_.eval())[0])).show()
    train_images, train_labels = tf.train.shuffle_batch([train_images_, train_labels_], batch_size=batch_size, num_threads=2,capacity=2000,min_after_dequeue=500)
    test_images, test_labels = tf.train.shuffle_batch([test_images_, test_labels_], batch_size=batch_size, num_threads=2,capacity=500,min_after_dequeue=256)
    # train_images：【256,200,200,3】
    # parse_labels：(256,)
    # min_after_dequeue：越大shuffle力度越大
    # 不是线程越多越快，甚至更多的线程反而会使效率下降。
    # Ensures a minimum amount of shuffling of examples.
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    #
    # image = tf.image.per_image_standardization(image)

    # # 随机裁剪大小
    # distorted_image = tf.random_crop(tf.cast(image, tf.float32), [IMAGE_SIZE, IMAGE_SIZE, 3])
    # # 随机水平翻转
    # distorted_image = tf.image.random_flip_left_right(distorted_image)
    # # 随机调整亮度
    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # # 随机调整对比度
    # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # # 对图像进行白化操作，即像素值转为零均值单位方差
    # float_image = tf.image.per_image_standardization(distorted_image)

#dropout ，fc的relu解决梯度被relu激活过大，导致训练一会后损失不变，预测值都为0
    DROPOUT=tf.constant(0.75)
    layer1_w = init_weights([fsize, fsize, samples_channels , layer1_output_num])
    layer2_w = init_weights([fsize, fsize, layer1_output_num, layer2_output_num])
    layer3_w = init_weights([fsize, fsize, layer2_output_num, layer3_output_num])
    layer4_w = init_weights([fsize,fsize,  layer3_output_num, layer4_output_num])
    fc1_w = init_weights([6400,1024])
    fc2_w = init_weights([1024,128])
    out_w = init_weights([128, one_hot_num])
    #valid不补
    conv1_o = tf.nn.relu(tf.nn.conv2d(train_images, layer1_w, strides, padding='VALID'),name="conv1_o")#x+2=200 198 stride=2 #2x+1<=200
    pool1_o = tf.nn.max_pool(conv1_o, ksize, ksize, padding="VALID",name="pool1_o")  # 99
    conv2_o = tf.nn.relu(tf.nn.conv2d(pool1_o, layer2_w, strides, padding='VALID',name="conv2_o"))#97
    pool2_o = tf.nn.max_pool(conv2_o, ksize, ksize, padding="VALID",name="pool2_o")  # 48
    conv3_o = tf.nn.relu(tf.nn.conv2d(pool2_o, layer3_w, strides, padding='VALID',name="conv3_o"))#46
    pool3_o = tf.nn.max_pool(conv3_o, ksize, ksize, padding="VALID",name="pool3_o")  # 23
    conv4_o = tf.nn.relu(tf.nn.conv2d(pool3_o, layer4_w, strides, padding='VALID',name="conv4_o"))#21
    pool4_o = tf.nn.max_pool(conv4_o, ksize, ksize, padding="VALID",name="pool4_o")  # 10*10*64
    pool4_o = tf.reshape(pool4_o, shape=[-1, 64*10*10])

    fc_1 = tf.nn.relu(tf.matmul(pool4_o, fc1_w))
    fc_1 = tf.nn.dropout(fc_1,DROPOUT)
    fc_2 = tf.nn.relu(tf.matmul(fc_1,fc2_w))
    fc_2 = tf.nn.dropout(fc_2,DROPOUT)

    output=tf.matmul(fc_2,out_w)
    train_labels_onehot=tf.one_hot(indices=train_labels, depth=one_hot_num, on_value=1, off_value=0)
    # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_1, labels=train_labels))
    #sparse时传递是一个值，本实验中一直报错

    with tf.name_scope("cost"):
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=train_labels_onehot))
        tf.summary.scalar("cost",cost)

    optimezer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
    # optimezer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.name_scope("corr_rate"):
        corr_rate=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1),tf.argmax(train_labels_onehot,1)),tf.float32))
        tf.summary.scalar("corr_rate",corr_rate)

    with tf.Session() as sess:
        start = datetime.now()
        x = tf.cond(is_training, lambda: train_images, lambda: test_images)
        y = tf.cond(is_training, lambda: train_labels, lambda: test_labels)
        writer = tf.summary.FileWriter(output_dir)
        merge = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables

        # start = global_step.eval()  # get last global_step
        # print("Start from:", start)
        # if debug:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # 调试
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Coordinator类可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常。
        # QueueRunner类用来协调多个工作线程同时将多个tensor压入同一个队列中。
        coord = tf.train.Coordinator()#创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)#启动所有的QueueRunner
        # enqueue_threads = tf.train.create_threads(sess, coord=coord, start=True)
        try:
          for step in range(iterations):
              if coord.should_stop():
                  break
              STEP,summary,pre, _, COST, CORR_RATE = sess.run([global_step.assign(step),merge,output, optimezer, cost, corr_rate], {is_training :True})
              # TIME = (datetime.now() - start).seconds 这里定义起始时间不管用，最终结果batch time都是0
              # global_step.assign(step).eval()  # set and update(eval) global_step with index, i
              saver.save(sess, ckpt_dir + "/model.ckpt", global_step=STEP)
              if  (step+1)% display_step == 0:
                  # STEP,test_cost, test_corr_rate = sess.run(
                  #     [global_step.assign(step), cost, corr_rate], {is_training: False})
                  # test_start = datetime.now()

                  test_cost, test_corr_rate = sess.run(
                      [cost, corr_rate], {is_training: False})
                  #不同的tensor显示方法
                  # print("the predict values is {}".format(output.eval()))
                  # print("the true values is {}".format(tf.Tensor.eval(train_labels)))
                  print("after %s iteration ,the cost values is %8.4f,the  corr_rate is %8.4f"%(step + 1, COST, CORR_RATE))
                  print("after %s iteration ,the test  coss  is %8.4f,test corr_tate is %8.4f"%(step+1, test_cost,test_corr_rate))
                  TIME = (datetime.now() - start).seconds
                  print("the train time:{} seconds".format(TIME))
                  cost_list.append(COST)
                  corr_rate_list.append(CORR_RATE)
                  writer.add_summary(summary,step)
                  # writer.add_summary(step,COST)
                  # writer.add_summary(step,CORR_RATE)
        except tf.errors.OutOfRangeError:
          print('OutOfRangeError ')
          print("after {} iteration ,the coss values is {},the corr_rate is {}".format(step+1, COST, CORR_RATE))

        finally:
          # When done, ask the threads to stop.
          TIME = (datetime.now() - start).seconds
          print("the train time:{} seconds".format(TIME))
          print(" training done;after {} iteration ,{} epochs,the coss values is {},the corr_rate is {}".format(step+1,(step+1)*batch_size//tarin_samples_num, COST, CORR_RATE))

          # timestamp = str(datetime.date(datetime.now()))
          timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
          np.save(output_dir+timestamp + "_cost_lst.npy", cost_list)
          np.save(output_dir+timestamp + "_corr_rate.npy", corr_rate_list)
          # plt.plot(cost_list)
          plt.figure(1)
          plt.subplot(211)
          plt.title("")
          plt.ylabel('corr_rate')
          plt.plot(corr_rate_list,'k')
          plt.subplot(212)
          plt.title("")
          plt.ylabel('cost')
          plt.plot(cost_list,'r--')
          plt.savefig(output_dir+timestamp+"cost_corr.png")
          # print(cost_list)
          plt.show()
          coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
# 坑1：一直报数据类型不匹配，如conv只能传递float类型
# 坑2：形状不匹配，VALID和SAME区别
# 坑3：debug的配置
# r -f has_inf_or_nan
# ni -t Discrim/add_2
# 坑4：自己将数据规范化到【-1,1】之间，但是由于初始化的weights（方差stdev）过大，导致输出值非常大（几十万），训练三轮以后就输出为0，应该是激活率低，原因是学习率大
# 导致relu后就传递很大梯度（减去很大的值），很容易导致某个节点失活，从而激活率下降，最终就使得每个节点都为0，从而输出就为0；
# 坑5：那么问题来了，为啥不是一开始第一轮训练完毕后就输出为0，
# 后来想了，并非所有的一开始就输出非常大，但是通过几次学习后就累积后误差还是很大导致输出为0

# 总结
# 该实验中，对初始值选择较敏感，容易导致输出很大值（stdev）
# 学习率敏感性：因为relu会学习大的梯度，learning rate大就更容易失活（发散），损失不变，很大可能是梯度消失，学习不到内容

# 想法
# 一开始FC层没有加relu以及dropout，导致的输出很大，然后训练几轮后就输出为0，
# 方案一 可以用sigmoid+较小的学习率
# 方案二 用clip gredient+ relu 来限制
#

# TODO:
# checkpoint 生效
# BN
# 可视化summary的问题，因为常用的方法都是训练一个batch加入到summary中，而验证的时候我是在全部验证集上做的，怎么办？下次再说！加入2个placeholder就行了

# tf.add_to_collection('logits',bottom)
# 可以通过 print(out.name) 来看看
# meta_path = './model/checkpoint/model.ckpt.meta'
# model_path = './model/checkpoint/model.ckpt'
# saver = tf.train.import_meta_graph(meta_path)  # 导入图
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
#     saver.restore(sess, model_path)  # 导入变量值
#     graph = tf.get_default_graph()
#     prob_op = graph.get_operation_by_name('prob')  # 这个只是获取了operation， 至于有什么用还不知道
# prediction = graph.get_tensor_by_name('prob:0')  # 获取之前prob那个操作的输出，即prediction
# print(ress.run(prediciton, feed_dict={...}))  # 要想获取这个值，需要输入之前的placeholder （这里我编辑文章的时候是在with里面的，不知道为什么查看的时候就在外面了...）
# print(sess.run(
#     graph.get_tensor_by_name('logits_classifier/weights:0')))  # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重

# 因为tensorflow是在图上进行计算，要驱动一张图进行计算，必须要送入数据，如果说数据没有送进去，那么sess.run()，就无法执行，tf也不会主动报错，提示没有数据送进去，其实tf也不能主动报错，因为tf的训练过程和读取数据的过程其实是异步的。tf会一直挂起，等待数据准备好。现象就是tf的程序不报错，但是一直不动，跟挂起类似

# https://blog.csdn.net/lujiandong1/article/details/53385092



            
            
            



