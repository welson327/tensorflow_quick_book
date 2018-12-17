# 圖片訓練 - by CNN

把上節的model改為下面，就可以了

```python
# 建立一個線性堆疊模型
model = Sequential()

#建立第1層券積層，透過濾鏡產生32個影像特徵
model.add(Conv2D(filters=32, kernel_size=(3,3),
                 input_shape=(img_feature.shape[1],img_feature.shape[2],img_feature.shape[3]),
                 activation='relu',
                 padding='same'))

#在第1層券積層加入Dropout層，避免overfitting
model.add(Dropout(rate=0.25))

#建立第1層池化層，將32*32影像，縮小為16*16影像
model.add(MaxPooling2D(pool_size=(2,2)))

#建立第2層券積層，透過濾鏡產生64個影像特徵
model.add(Conv2D(filters=64, kernel_size=(3,3),
                 activation='relu',
                 padding='same'))

#在第2層券積層加入Dropout層，避免overfitting
model.add(Dropout(rate=0.25))

#建立第2層池化層，將16*16影像，縮小為8*8影像
model.add(MaxPooling2D(pool_size=(2,2)))

#建立平坦層，將64個8*8影像轉換為一維向量，64*8*8=4096個數字
model.add(Flatten())
model.add(Dropout(rate=0.25))

#建立有1920個神經元的隱藏層
model.add(Dense(1920, activation='relu'))
model.add(Dropout(rate=0.25))

#建立有2個神經元的輸出層
model.add(Dense(2, activation='softmax'))

# 定義訓練方式
model.compile(loss='categorical_crossentropy', # 損失函數
             optimizer='adam', # 最佳化方法
             metrics=['accuracy']) # 評估方式:準確度

# 顯示模型摘要
print(model.summary())

# 開始訓練模型
train_history = model.fit(x=stk_feature_normalized, # 指定 feature
                         y=stk_label_onehot, # 指定 label 
                         validation_split=0.2, # 分80%訓練，20%驗證
                         epochs=5, # 執行 5 次訓練
                         batch_size=200, # 批次訓練，每批次 200 筆資料
                         verbose=2) # 顯示訓練過程

# 儲存模型
model.save("/your/model/path")
```


#### 什麼是CNN演算法
可以參考下面
https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_convolutional_neural_networks_work.html
