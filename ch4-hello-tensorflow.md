# Hello Tensorflow

在虛擬環境安裝好tensorflow以後  
簡單介紹tensorflow(簡寫tf)的程式架構  


```python
import tensorflow as tf

# -----------------------------------------------------
# Define tensorflow model
# https://gist.github.com/eerwitt/518b0c9564e500b4b50f
# -----------------------------------------------------
# 定義 graph (tensor 和 flow)
str = "Hello Tensorflow"

final_image_array = []
# 執行 graph
with tf.Session() as sess:
	# 執行tensorflow時要先初始化(初學者照抄即可!)
	init = tf.global_variables_initializer()
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord) # 建立多執行緒

	# exec tf flow
	print(str)

	# 停止多執行緒(初學者照抄前可!)
	coord.request_stop()
	coord.join(threads)
```

要寫tf時，都要先定義flow  
定義好flow以後，接下來才會執行  

以本例來說，我們的flow就是定義一個字串`str`  
真正要執行程式，其實是在`with tf.Session() as sess:`這段裡  
裡面分為三部份「初始化」、「執行flow」、「結束」  
其中初始化和結束對初學者來說照抄即可  
中間的`print(str)`才是需要去關注的  
