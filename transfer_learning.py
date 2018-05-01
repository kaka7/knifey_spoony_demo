#encoding=utf-8
# 从某一层开始获取权重并开始transferlearning
# 输入图像的大小建议和pre_process model要求的输入大小一致，效果会更好
# 从处理方式的灵活性上推荐函数式模型，sequential只是特例
# 各层的激活函数优先选sigmoid ，即便是最后一层的softmax?
# 将最后一层的输出作为输入的ＴＬ实验３ 起始准确率不到２０　而本实验直接将池化层的结果作为输入　然后ｓｏｆｔｍａｘｘ获得的起始准确率达６０％

#总结一下：　ｔransfer learning :利用现有的模型，将特征层剥离，然后fine-truning
# １可以直接将输出层作为输入
# ２将中间某层（最好是特征层，比如卷积后，ＦＣ已经到达了抽象的感知事物的理解程度）或多层（trainable）
# 另外，还有就是自己写将多次结果的数据写到ｈ５文件中，然后读取

# keras 2.1.5
# 报错：StopIteration: unsupported operand type(s) for /=: 'JpegImageFile' and 'float'
# 降级到2.1.3

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("nvidia-smi | grep vpython3 | awk '{print $3}' | xargs kill -9")
os.system("lsof -i:6006 | grep tensorboa | awk '{print $2}' | xargs kill -9")


height=224
train_batch_size=64
test_batch_size=64
FLAG = None
#assert

train_data_dir='/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/train_data_link/'
test_data_dir='/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/test/'
gen = ImageDataGenerator()
generator_train = gen.flow_from_directory(train_data_dir, target_size=(height, height),batch_size=train_batch_size)
generator_test = gen.flow_from_directory(test_data_dir, target_size=(height, height),batch_size=test_batch_size)
# base_model = VGG16(include_top=True, weights='imagenet')需要下载，速度慢，
# include_top=true时下载也比较慢，需要测试等于true时的模型结构
# input_tensor=Input((height, height, 3))报错ValueError: The shape of the input to "Flatten" is not fully defined (got (None, None, 2048).
#  Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.，是因为此时用的Ｍｏｄｅｌ需要预先指定ｉｎｐｕｔ
base_model = ResNet50(input_tensor=Input((height, height, 3)),weights='imagenet', include_top=False)
base_model.summary()
for i ,layer in enumerate(base_model.layers):
    print("layer_{0}:\t{1}\t{2}".format(i,layer.trainable, layer.name))

#这个地方比较坑，调试比较费力，按照网上的操作,base_model.output的输出是tensor，以后都是tensor， 传递给其他层，切记，
# TypeError: 'NoneType' object is not callable
# 网上还将func 模型传给sequential（add）会出问题，#ValueError: Variable bn_conv1/moving_mean/biased already exists, disallowed.
# Did you mean to set reuse=True in VarScope? Originally defined at:应该是版本问题，报错说变量存在，设置需要重用
x=Flatten()(base_model.layers[172].output)#不必指定Input,或者base_model的output，就是最后一层
# transfer_layer = model.get_layer('activation_48')
# #获取权重tensor 切记和layer的区别,否则weights维度为none
#【7,7,512】直接FC到3也不错，单独只polling操作也可
# x=GlobalAveragePooling2D()(base_model.layers[172].output)
x = Dropout(0.25)(x)
x = Dense(3, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()

# ＃这里不是ｆａｌｓｅ那么就是整个网络重新训练易导致ＯＯＭ
# for layer in base_model.layers:
#     layer.trainable = False
#
# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
# history_ft = model.fit_generator(
#     generator_train,#可自定义
#     # samples_per_epoch=4170,  # nb_train_samples
#     steps_per_epoch=4170,  # nb_train_samples#每轮epoch遍历的samples
#     # validation_data=generator_test,#可自定义
#     # nb_epoch=100,
#     epochs=100
#     # nb_val_samples=nb_val_samples
# )

for layer in base_model.layers[:172]:
   layer.trainable = False
for layer in base_model.layers[172:]:
   layer.trainable = True
for i ,layer in enumerate(model.layers):
    print("layer_{0}:\t{1}\t{2}".format(i,layer.trainable, layer.name))

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 节省内存，类似tf的queue
history_ft = model.fit_generator(
    generator_train,#可自定义
    # samples_per_epoch=4170,  # nb_train_samples，Basically steps_per_epoch = samples_per_epoch/batch_size
    # steps_per_epoch=10,  # nb_train_samples#每轮epoch遍历的samples
    validation_data=generator_test,#可自定义
    nb_epoch=10,
    verbose=1,#控制显示方式，冗长
    validation_steps=530//64,
    workers=8,
    # use_multiprocessing=True,
    # epochs=100
    # nb_val_samples=530 # nb_val_samples`->`validation_steps
)

# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# train_y=tf.one_hot(train_data_dir.classes,depth=3)
# sess.run(train_y)
#
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)






