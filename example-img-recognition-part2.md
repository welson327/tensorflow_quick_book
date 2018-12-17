# 簡單範例 - 判斷圖片是否為貓 - part2

在part1已經學到如何得到Feature與Label陣列
本節就準備使用這個陣列進行訓練

#### 訓練模型


**直接看sample code**
```python
"""Train MLP Model
Using MLP Model to Train Picture Recognition Model.
"""
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np


# 設定 np 亂數種子
np.random.seed(10)

# 載入訓練資料集
n = 10000
img_feature = np.fromfile("./your/image/training/array.features", dtype=np.uint8)
img_feature = img_feature.reshape(n, 30, 30, 3)
img_label = np.fromfile("./your/image/training/array.labels", dtype=np.uint8)
img_label = img_label.reshape(n, 1)

# 打散資料集
indexs = np.random.permutation(img_label.shape[0])
rand_img_feature = img_feature[indexs]
rand_img_label = img_label[indexs]

# 資料正規化
# 將 feature 數字轉換為 0~1 的浮點數，能加快收斂，並提升預測準確度
# 把維度 (n,30,30,3) => (n, 30*30*3)後，再除255
img_feature_normalized = rand_img_feature.reshape(n, 30*30*3).astype('float32') / 255

# 將 label 轉換為 onehot 表示
img_label_onehot = np_utils.to_categorical(rand_img_label)

# 建立一個線性堆疊模型
model = Sequential()

# 建立輸入層與隱藏層
model.add(Dense(input_dim = 30*30*3, # 輸入層神經元數
                units = 1000, # 隱藏層神經元數
                kernel_initializer = 'normal', # 權重和誤差初始化方式:normal，使用常態分佈產生出始值
                activation = 'relu')) # 激活函數:relu函數，忽略掉負數的值

# 建立輸出層
model.add(Dense(units = 2, # 輸出層神經元數 (即[True, False])
                kernel_initializer = 'normal',
                activation = 'softmax')) # 激活函數:softmax函數，使輸出介於 0~1 之間

# 定義訓練方式
model.compile(loss='categorical_crossentropy', # 損失函數
             optimizer='adam', # 最佳化方法
             metrics=['accuracy']) # 評估方式:準確度

# 顯示模型摘要
print(model.summary())

# 開始訓練模型
train_history = model.fit(x=img_feature_normalized, # 指定 feature
                          y=img_label_onehot, # 指定 label 
                          validation_split=0.2, # 分80%訓練，20%驗證
                          epochs=5, # 執行 5 次訓練
                          batch_size=200, # 批次訓練，每批次 200 筆資料
                          verbose=2) # 顯示訓練過程

# 儲存模型
model.save("./your/image/training/models.dat")
```

**第一步：feature, label陣列**  
使用`np.fromfile()`讀檔即可  
讀完檔案後記得`reshape()`成當初你建立的陣列大小  
因為訓練的資料愈平均愈好，所以可用`np.random.permutation()`打散資料  

**第二步：正規化**  
我們不希望算到最後正無限大或負無限大這種無意義的數字  
所以建議資料都要正規化成0~1之間，讓運算時可以得到比較恰當的數字  
  
假設一張圖是`(30, 30)`的維度，考慮RGB，我們會存成`(30, 30, 3)`  
在丟進訓練系統時，會將每一張圖用一維陣列表示，即`(30*30*3, 1)`  
因為要正規化RGB，所以為除255。即：
```
.reshape(n, 30*30*3).astype('float32') / 255
```

每張圖對應的label陣列也需處理成機器學習的格式(稱為One-Hot Encoding)  
例如判斷數字的圖片，label由`[1,2,3,...,9,0]`組成，但這樣的資料機器無法辨識  
所以必須:  
將1變成[1,0,0,0,0,0,0,0,0,0],  
將2變成[0,1,0,0,0,0,0,0,0,0],  
...  
將10變成[0,0,0,0,0,0,0,0,0,1].  
如此機器學習才能處理。最後我們的one-shot資料就成了:  
```
label = [
  [0,0,0,0,1,0,0,0,0,0],
  [0,0,0,1,0,0,0,0,0,0],
  [0,0,1,0,0,0,0,0,0,0],
  ...
]
```
這樣代表第一張圖是數字5，第二張圖是數字4，以此類推  
可以簡單理解成：第一張是數字5的機率為100%，其他數字的機率是0%  

那以上的工作，都可以交給keras的`np_utils.to_categorical()`函數直接完成  
它很神奇的是，會判斷label array的所有內容，最後正規化成我們要的  


**第三步：定義神經網路**  
其實keras已經幫我們封裝了tensorflow，所以code讀起來就很簡單  
只要幾個步驟就可以完成定義  
```python
# 建立模型
model = Sequential()
# 定義input維度為30*30*3，即一張圖。定義隱藏層為1000個神經元
model.add(Dense(input_dim=30*30*3, units=1000, ...))
# 定義output神經元數為2，代表正規化後的label arry，即[1,0]
model.add(Dense(units=2))
# 定義訓練方式
model.compile()
# 開始訓練
model.fit()
# 存檔
model.save("/your/path")
```
很多不懂的參數可以先照抄就可以了。



#### 訓練結果不理想？

此時您應該會發現，好像不論參數怎麼調整，結果都不是很理想  
原因在於本例是講解一個通式而已  
真正在進行圖形訓練，我們會使用CNN演算法。請見下節。  

