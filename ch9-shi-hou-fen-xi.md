# 分析

前節我們提到，在產生模型時，可以預留一些data進行預估模擬  
實際上我們還是會希望由外部檔案進行model的測試  

```python
from keras.models import load_model
from keras.utils import np_utils
import pandas as pd

# 載入測試資料(正規化的feature,label陣列)
feature_ary = np.fromfile("/test_feature_ary/path", dtype=np.uint8)
feature_ary = feature_ary.reshape(n, h, w, 3)
label_ary = np.fromfile("/test_label_ary/path", dtype=np.uint8)
label_ary = label_ary.reshape(n, 1)
feature_ary_normalized = feature_ary.astype('float32') / 255
label_ary_onehot = np_utils.to_categorical(label_ary)

# 載入訓練好的model
model = load_model("/your/model/path")
```

得到model以後，可以先察看待驗證資料的accuracy及預估值
```python
evalution = model.evaluate(feature_ary_normalized, label_ary_onehot)
prediction = model.predict_classes(feature_ary_normalized)
print("Accuracy = ", evalution[1])
print("前10筆預測結果：", prediction[:10])
```
結果如下:  
![](/assets/螢幕快照 2017-12-21 下午6.06.50.png)

進一步分析，看幾張正確，幾張錯誤，可使用pandas的cross table
```python
label_ary_reshaped = label_ary.reshape(n)
table = pd.crosstab(label_ary_reshaped, prediction, rownames=['label'], colnames=['predict'])
print(table)
```
結果如下:  
![](/assets/螢幕快照 2017-12-21 下午6.12.01.png) 

由上表可知，預測正確的有258+64張，預測錯誤的有71+3張  



如果想取得每一張圖預測的數值(即每個label值可能的機率)  
```python
predict_prob = model.predict(feature_ary_normalized)
print("第{}張貼圖每種標籤的預測機率: {}".format(i, predict_prob[i]))
...
```
![](/assets/螢幕快照 2017-12-25 上午11.46.55.png)

上圖可知:  
第3張貼圖預測結果是 `[0.82102931, 0.17897071]`  
對應到當初設計的one-shot encoding，就是0的機率是82%, 1的機率是17%，所以predict_classes的結果為0  