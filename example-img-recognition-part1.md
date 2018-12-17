# 簡單範例 - 判斷圖片是否為貓 - part1

整個機器學習的流程就是  
Step1: 產生X, Y矩陣，稱為Feature, Label矩陣  
Step2: 學習模型  
Step3: 驗證模型  

以下我們將以「教機器學會判斷圖片是否為貓」為例

#### Feature(X), Label(Y)矩陣

第一步，必須有很多貓的圖片(X)，並且逐一標示是否為貓(Y)  
這個步驟因為是已經知道input(貓的圖片)/output(是否為貓)  
所以也稱為監督式學習(Supervised Learning)
```
Feature: X = [x1, x2, ...]
Label: Y = [y1, y2, ...]
```
**X比較複雜**  
如果一張圖的維度是(30, 30)，因為是彩色(RGB)的關係  
所以矩陣為(30, 30, 3)  
現有n張圖，所以X陣列維度為(n, 30, 30, 3)  

**Y相對較簡單**  
Y的維度就是 (n, 1), 由true,false(或1,0)組成

**以下sample code**
```python
import tensorflow as tf
import numpy as np

def load_image_features(img_paths=[]):
	# -----------------------------------------------------
	# Define tensorflow model
	# https://gist.github.com/eerwitt/518b0c9564e500b4b50f
	# -----------------------------------------------------
	# 定義 graph (tensor 和 flow)
	filename_queue = tf.train.string_input_producer(img_paths, shuffle=False)
	image_reader = tf.WholeFileReader()
	file_name, file_content = image_reader.read(filename_queue)
	decoded_image = tf.image.decode_png(file_content, channels=3)
	resized_image = tf.image.resize_images(decoded_image, [30, 30])

	final_image_ary = []
	# 執行 graph
	with tf.Session() as sess:
		# 執行tensorflow時要先初始化(初學者照抄即可!)
		init = tf.global_variables_initializer()
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord) # 建立多執行緒

		# image to RGB
		for i in range(len(img_paths)):
			img = sess.run(resized_image)
			final_image_ary.append(img)

		# 將最後的陣列轉換為 numpy 格式的陣列，以便存檔
		final_image_ary = np.array(final_image_ary, dtype=np.uint8)
		# 存檔
		final_image_ary.tofile("./your/image/training/array.features")

		# 停止多執行緒(初學者照抄前可!)
		coord.request_stop()
		coord.join(threads)

	return final_image_array
```

在Hello Tensorflow章節已經有提到撰寫的架構，本例依然遵守這個架構  

**第一步，定義graph**  
在 `with tf.Session() as sess` 前，先定義要做什麼  
本例即使用tensorflow內建的image reader先「讀圖(.decode_png)」  
為了減少運算量，我們還可以「縮圖(.resize_images)」  
最後得到`resized_image`這個變數，tensorflow稱之為`graph`  

**第二步，執行graph**  
因為有很多張圖，所以用一個for迴圈跑  
每次跑的內容就是先前定義好的graph  
這裡我們將最後的結果append到final_image_ary裡  

**第三步，存檔**
圖片讀取之後可能需要存檔  
Python的Numpy套件提供了簡易的方法: `.tofile(path)`  
但必須先將final_image_ary轉成numpy的格式後，再存檔，即  
```python
final_image_ary = np.array(final_image_ary, dtype=np.uint8)
final_image_ary.tofile("./your/image/training/array.features")
```