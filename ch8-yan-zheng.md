# 驗證

真正的驗證是需要單獨寫一直程式來驗證的
本節只稍微介紹一個參數

在訓練時，我們給了一個參數`validation_split=0.2`
```python
# 開始訓練模型
train_history = model.fit(x=img_feature_normalized, # 指定 feature
                          y=img_label_onehot, # 指定 label 
                          validation_split=0.2, # 分80%訓練，20%驗證
                          epochs=5, # 執行 5 次訓練
                          batch_size=200, # 批次訓練，每批次 200 筆資料
                          verbose=2) # 顯示訓練過程
```
這代表80%進行訓練，20%進行驗證
換句話說，如果資料有10000筆，則8000筆拿來學習，2000筆拿來驗證
所以資料的平均分散性就很重要了

執行CNN算法後，會看到
![](/assets/螢幕快照 2017-12-11 下午4.24.43.png)

以本圖為例  
* 首先看到的表格就是類神經的定義。
* 最下方顯示`acc: 0.9006`代表訓練集的精準度是90.06%，`val_acc: 0.9011`代表測試集的精準度是90.11%。
* 本例共執行5次訓練，acc和val_acc的值差異愈小愈好。如果愈差愈大，代表訓練過頭，專有名詞是overfitting。
