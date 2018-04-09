import cluster
from numpy import *
import tensorflow as tf
def replace(a):
	if a>0:
		return 0
	else:
		return 1
def readSlices(file):
	slices = open(file)
	slices = list(map(int,slices.read().split()))
	slices = list(map(replace,slices))
	return array(slices).reshape(128,128,128,1)

	
def weight_variable(shape):  
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1), name='weight')  

def conv_2d(x, w):  
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding="VALID")  

def evaluate(y, y_):  
    y = tf.arg_max(input=y, dimension=1)  
    y_ = tf.arg_max(input=y_, dimension=1)  
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(y, y_), tf.float32)) 

def bias_variable(shape):  
    return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), name='bias')  
if __name__=='__main__':
	slices = readSlices('dogOfTheBullKiller.slice')
	
	tfX = tf.placeholder(tf.float32, [1, 128,128,1])
	w_convX1 = weight_variable(shape=[12, 12, 1, 3])  
	b_convX1 = bias_variable(shape=[3])  
	convX1_out = tf.nn.relu(conv_2d(tfX,w_convX1)+b_convX1)

	w_convX2 = weight_variable(shape=[12, 12, 3, 9])  
	b_convX2 = bias_variable(shape=[9])  
	convX2_out = tf.nn.relu(conv_2d(convX1_out,w_convX2)+b_convX2)

	w_convX3 = weight_variable(shape=[10, 10, 9, 27])  
	b_convX3 = bias_variable(shape=[27])  
	convX3_out = tf.nn.relu(conv_2d(convX2_out,w_convX3)+b_convX3)

	w_convX4 = weight_variable(shape=[10, 10, 27, 54])  
	b_convX4 = bias_variable(shape=[54])  
	convX4_out = tf.nn.relu(conv_2d(convX3_out,w_convX4)+b_convX4)

	w_convX5 = weight_variable(shape=[10, 10, 54, 108])  
	b_convX5 = bias_variable(shape=[108])  
	convX5_out = tf.nn.relu(conv_2d(convX4_out,w_convX5)+b_convX5)

	w_convX6 = weight_variable(shape=[10, 10, 108, 216])  
	b_convX6 = bias_variable(shape=[216])  
	convX6_out = tf.nn.relu(conv_2d(convX5_out,w_convX6)+b_convX6)







	w_convT1 = weight_variable(shape=[10, 10, 108, 216])
	b_convT1= bias_variable(shape=[108])
	convT1_out = tf.nn.relu(tf.nn.conv2d_transpose(convX6_out,w_convT1,output_shape=[1,79,79,108],strides=[1,1,1,1],padding="VALID")+b_convT1)

	w_convT2 = weight_variable(shape=[10, 10, 54, 108])
	b_convT2= bias_variable(shape=[54])
	convT2_out = tf.nn.relu(tf.nn.conv2d_transpose(convT1_out,w_convT2,output_shape=[1,88,88,54],strides=[1,1,1,1],padding="VALID")+b_convT2)

	w_convT3 = weight_variable(shape=[10, 10, 18, 54])
	b_convT3= bias_variable(shape=[18])
	convT3_out = tf.nn.relu(tf.nn.conv2d_transpose(convT2_out,w_convT3,output_shape=[1,97,97,18],strides=[1,1,1,1],padding="VALID")+b_convT3)

	w_convT4 = weight_variable(shape=[10, 10, 9, 18])
	b_convT4= bias_variable(shape=[9])
	convT4_out = tf.nn.relu(tf.nn.conv2d_transpose(convT3_out,w_convT4,output_shape=[1,106,106,9],strides=[1,1,1,1],padding="VALID")+b_convT4)

	w_convT5 = weight_variable(shape=[12, 12, 3, 9])
	b_convT5= bias_variable(shape=[3])
	convT5_out = tf.nn.relu(tf.nn.conv2d_transpose(convT4_out,w_convT5,output_shape=[1,117,117,3],strides=[1,1,1,1],padding="VALID")+b_convT5)

	w_convT6 = weight_variable(shape=[12, 12, 1, 3])
	b_convT6= bias_variable(shape=[1])
	y_pred = tf.nn.relu(tf.nn.conv2d_transpose(convT5_out,w_convT6,output_shape=[1,128,128,1],strides=[1,1,1,1],padding="VALID")+b_convT6)
	#print('*************************')
	#print(y_pred)


	tfy = tf.placeholder(tf.float32, [1, 128,128,1])
	Loss = tf.nn.l2_loss(y_pred - tfy)
	Step_train = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss=Loss)

	initialized_variables = tf.initialize_all_variables()



	sess = tf.Session()
	saver=tf.train.Saver()
	saver.restore(sess,"/20170501/1/surface15.ckpt")

	a=[]#第一根轴切片 以a的方向作为标准方向
	b=[]
	c=[]
	for i in range(128):
		a.append(sess.run(fetches=y_pred, feed_dict={tfX:slices[i,0:128,0:128,0:1].reshape((1,128,128,1)), tfy:slices[i,0:128,0:128,0:1].reshape((1,128,128,1))}))
		b.append(sess.run(fetches=y_pred, feed_dict={tfX:slices[0:128,i,0:128,0:1].reshape((1,128,128,1)), tfy:slices[i,0:128,0:128,0:1].reshape((1,128,128,1))}))
		c.append(sess.run(fetches=y_pred, feed_dict={tfX:slices[0:128,0:128,i,0:1].reshape((1,128,128,1)), tfy:slices[i,0:128,0:128,0:1].reshape((1,128,128,1))}))
		print(i+1)
	#b,c的shape为（128，1，128，128，1）  a为xyz b为yxz c为zxy
	a=array(a).reshape(128,128,128)
	b=array(b).reshape(128,128,128).transpose((1,0,2))
	c=array(c).reshape(128,128,128).transpose((1,2,0))
	for x in range(128):
		for y in range(128):
			for z in range(128):

				a[x, y, z] = 1 if (a[x, y, z] > 0.9) else 0
				b[x, y, z] = 1 if (b[x, y, z] > 0.9) else 0
				c[x, y, z] = 1 if (c[x, y, z] > 0.9) else 0
				if(b[x,y,z]==1 or c[x,y,z]==1):
					a[x,y,z]=1
		print(x)


	for i in range(128):
		output=array(a[i]).reshape((1,16384))
		f=open("result/a"+str(i)+".txt","w")
		outputy=[]
		for j in range(16384):
			if output[0, j] > 0.8:
				outputy.append(1)
			else:
				outputy.append(0)
		pre=str(outputy)
		pre=pre.replace("[","")
		pre=pre.replace("]","")+"\n"


		f.write(pre)
		f.close()







		output=array(slices[i,0:128,0:128,0:1]).reshape((1,16384))
		f=open("result/b"+str(i)+".txt","w")
		outputy=[]
		for j in range(16384):
				outputy.append(output[0,j])

		pre=str(outputy)
		pre=pre.replace("[","")
		pre=pre.replace("]","")+"\n"


		f.write(pre)
		f.close()








