#实现两个版本的transfer learning 

* transfer_learning with keras on my own data.ipynb:keras 2.13
* transfer_learning with tf on my own data.ipynb:tensorflow1.4 GPU 版本


##transfer_learning with tf on my own data.ipynb
* 用于杀掉调试时的进程和tensorboard端口，释放GPU资源
* os.environ["CUDA_VISIBLE_DEVICES"] = "0"
* os.system("nvidia-smi | grep vpython3 | awk '{print $3}' | xargs kill -9")
* os.system("lsof -i:6006 | grep tensorboa | awk '{print $2}' | xargs kill -9")
* os.system("export TERM=linux ; export TERMINFO=/etc/terminfo")
* 实现TFrecord方式保存文件,自定义自己的数据，官网给的教程不能满足需求，坑多
* 实现使用queue的方式在有限GPU资源下使用大量的训练数据流式读取数据训练模型，且TF提供很好的数据增强api，目前没用上
* TF DEBUG功能：使用很简单，找到NAN也容易，但是寻找原因需要对模型有很好的认识，本人就碰到，原因找了两天，后来发现
* 添加checkpoint tensorboard等功能

*  坑1：一直报数据类型不匹配，如conv只能传递float类型
*  坑2：形状不匹配，VALID和SAME区别
*  坑3：debug的配置
*  r -f has_inf_or_nan
*  ni -t Discrim/add_2
*  坑4：自己将数据规范化到【-1,1】之间，但是由于初始化的weights（方差stdev）过大，导致输出值非常大（几十万），训练三轮以后就输出为0，应该是激活率低，原因是学习率大
*  导致relu后就传递很大梯度（减去很大的值），很容易导致某个节点失活，从而激活率下降，最终就使得每个节点都为0，从而输出就为0；
*  坑5：那么问题来了，为啥不是一开始第一轮训练完毕后就输出为0，
*  后来想了，并非所有的一开始就输出非常大，但是通过几次学习后就累积后误差还是很大导致输出为0

*  总结
*  该实验中，对初始值选择较敏感，容易导致输出很大值（stdev）
*  学习率敏感性：因为relu会学习大的梯度，learning rate大就更容易失活（发散），损失不变，很大可能是梯度消失，学习不到内容

*  想法
*  一开始FC层没有加relu以及dropout，导致的输出很大，然后训练几轮后就输出为0，
*  方案一 可以用sigmoid+较小的学习率
*  方案二 用clip gredient+ relu 来限制
*  感谢网友给力方案，同时在tf中训练和测试，通过传递参数来表示

        x = tf.cond(is_training, lambda: train_images, lambda: test_images)
        y = tf.cond(is_training, lambda: train_labels, lambda: test_labels)
        STEP,summary,pre, _, COST, CORR_RATE = sess.run([global_step.assign(step),merge,output, optimezer, cost, corr_rate], {is_training :True})
        test_cost, test_corr_rate = sess.run([cost, corr_rate], {is_training: False})

##transfer_learning with keras on my own data.ipynb
* 不得不说keras的高效，适合项目前期快速验证想法，api太简洁啦，但是还不够稳定，被迫从2.15转到2.13 说多了都是泪啊
* 从某一层开始获取权重并开始transferlearning
* 输入图像的大小建议和pre_process model要求的输入大小一致，效果会更好
* 从处理方式的灵活性上推荐函数式模型，sequential只是特例
* 各层的激活函数优先选sigmoid ，即便是最后一层的softmax?
* 将最后一层的输出作为输入的ＴＬ实验３ 起始准确率不到２０　而本实验直接将池化层的结果作为输入　然后ｓｏｆｔｍａｘｘ获得的起始准确率达６０％*
* 总结一下：　ｔransfer learning :利用现有的模型，将特征层剥离，然后fine-truning
* １可以直接将输出层作为输入
* ２将中间某层（最好是特征层，比如卷积后，ＦＣ已经到达了抽象的感知事物的理解程度）或多层（trainable）
* 另外，还有就是自己写将多次结果的数据写到ｈ５文件中，然后读取*
* keras 2.1.5
* 报错：StopIteration: unsupported operand type(s) for /=: 'JpegImageFile' and 'float'  降级到2.1.3

## help.py
* 包括数据下载，TFrecords转换和读取

