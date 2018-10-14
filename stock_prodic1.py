import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#_________读入数据______________
filepath = 'D:\project\stock_prodict\dataset_1.csv'
modelpath = 'D:\project\stock_prodict\model.model'
f = open(filepath)
df = pd.read_csv(f)
data = np.array(df['最高价'])
data = data[::-1]
#plt.figure()#显示数据
#plt.plot(data)
#plt.show()
normalize_data = (data - np.mean(data, dtype=tf.float32))/np.std(data, dtype=tf.float32) #归一化数据
normalize_data = normalize_data[:, np.newaxis]
print(type(normalize_data))
#生成训练集
time_step = 20    #时间步
rnn_unit = 10     #隐藏层单元数
batch_size = 60   #一次喂入数据量
input_size = 1    #输入维度
output_size = 1   #输出维度
learning_step = 0.001   #学习率
train_step = 5000      #训练次数
train_x, train_y = [], []   #训练集
#生成训练集
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i : i + time_step]
    y = normalize_data[i + 1 : i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
#_______神经网络变量__________
batch_x = tf.placeholder(tf.float32, [None,time_step, input_size])
batch_y = tf.placeholder(tf.float32, [None,time_step, output_size])

w_in = tf.Variable(tf.random_normal([input_size, rnn_unit]), dtype=tf.float32)
b_in = tf.Variable(tf.random_normal([rnn_unit, ]), dtype=tf.float32)

w_out = tf.Variable([rnn_unit, output_size], dtype=tf.float32)
b_out = tf.Variable([output_size, ], dtype=tf.float32)
#神经网络
def lstm(batch):
    input = tf.reshape(x, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit]) #隐藏层输出reshape后作为cell的输入
    #cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    output_rnn, final_state = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn,[-1,time_step])
    pred = tf.maximum(output,w_out) + b_out
    return pred,final_state
#训练模型
def train_lstm():
    global batch_size
    pred,_ = lstm(batch_size)
    loss = tf.reduce_mean(tf.reshape(pred,[-1])-tf.reshape(y,[-1]))
    train_op = tf.train.AdamOptimizer(learning_step).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_step):
            step = 0
            start = 0
            end = start + batch_size
            while(end < len(train_x)):
                _,loss_ = sess.run([train_op,loss],feed_dict={batch_x:train_x[start:end], batch_y:train_y[start:end]})
                start = start + batch_size
                end = start + batch_size
                #saver model
                if step%100 == 0:
                    print('轮',i,'   step',step,'   loss=',loss_)
                    print('save model',saver.save(sess,'model.model'))
                    step = step + 1

#train model
train_lstm()

def prediction():
    pred,_=lstm(1)
    saver = tf.Saver(tf.global_variables())
    with tf.Session() as sess:
        #restore model
        model_file = tf.train.latest_checkpoint(modelpath)
        saver.restore(sess,model_file)
        #取一个样本作为测试样本
        pre_seq = train_x[5000,:,:]
        predict = []
        #之后的预测结果
        for i in range(1000):
            next_seq = sess.run(pred,feed_dict={batch_x:pre_seq})
            predict.append(next_seq[-1])
            pre_seq = np.vstack((pre_seq[1:],next_seq[-1]))
        plt.figure()
        plt.plot(list(range(len(normalize_data))),normalize_data,color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()

prediction()
