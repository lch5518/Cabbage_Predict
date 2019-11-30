# 배추가격 예측하기
2019 한남대학교 빅데이터 경진대회 최우수상

## 데이터
* [농림축산식품 공공데이터포털](http://data.mafra.go.kr/opendata/data/indexOpenDataDetail.do?data_id=20151117000000000534&filter_ty=R&getBack=&sort_id=&s_data_nm=&instt_id=&cl_code=&shareYn=<br>)<br> 
* [기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)


### 농수산데이터 불러오기
<pre>
<code>
  cabbagedf = []
  cabbagedf = pd.DataFrame(cabbagedf)
  for j in range(16,20):
      for i in range(1,13):
          if(j==19 and i==9):break
          if i<10:filename = "원천조사가격정보_20"+ str(j) + "0" + str(i) + ".csv"
          else:filename = "원천조사가격정보_20"+ str(j) + str(i) + ".csv"
          cabbageTemp = pd.read_csv(filename, encoding = 'cp949', engine = 'python')
          cabbagedf = cabbagedf.append(cabbageTemp)
          print(filename)
</code>
</pre>


### 배추만 뽑아오기(도매-상등급 기준)
<pre>
<code>
  cbg_name = cabbagedf[cabbagedf['조사가격품목명'] == '배추']
  cbg_pt = cbg_name[cbg_name['조사구분명'] == '도매가격']
  cbg_1 = cbg_pt[cbg_pt['조사등급명'] == '상(1등급)']
  cbg_nd = cbg_1[['조사일자','당일조사가격']].groupby('조사일자', as_index=False).agg(np.mean) # 전국으로
</code>
</pre>

### 기상데이터 불러오기
<pre>
<code>
  weather = pd.read_csv("기상.csv", encoding = 'cp949', engine = 'python')
  weather_use = weather.groupby('일시', as_index=False).agg(np.mean) # 전국으로
  weather_use.isnull().sum()
  weather_use = weather_use.fillna(0)
</code>
</pre>

### 데이터 합치기(merge)
<pre>
<code>
  cabbge_res = pd.merge(cbg_nd, weather_use, left_on = '조사일자', right_on = '일시')
</code>
</pre>

### 주요변수 선택(R에서 AIC함수이용-후진제거법)
<pre>
<code>
  cabbge_select = cabbge_res[['일시','평균기온(°C)','최고기온(°C)','일강수량(mm)','최대 순간 풍속(m/s)',
              '최대 풍속(m/s)','평균 풍속(m/s)','풍정합(100m)','평균 이슬점온도(°C)','평균 상대습도(%)',
               '평균 증기압(hPa)','평균 현지기압(hPa)','평균 해면기압(hPa)','1시간 최다일사량(MJ/m2)',
              '합계 일사량(MJ/m2)','평균 중하층운량(1/10)','당일조사가격']]

  cabbge_select_use = cabbge_select[['평균기온(°C)','최고기온(°C)','평균 현지기압(hPa)','평균 중하층운량(1/10)','당일조사가격']]
</code>
</pre>
<br><br><br>
## 케라스 이용하기
### 목표변수 분리(가격)
<pre>
<code>
  data = np.asarray(cabbge_select_use[cabbge_select != cabbge_select_use['당일조사가격']])
  targets = np.asarray(cabbge_select_use['당일조사가격'])
</code>
</pre>

### Train/Test 데이터 나누기
<pre>
<code>
  cnt = 1
  price_train = []
  price_test = []

  for i in targets:
      if cnt <= 528:
          price_train.append(i)
      else:
          price_test.append(i)
      cnt += 1

  train_targets = np.asarray(price_train)
  test_targets = np.asarray(price_test)
  
  cnt = 1
  price_train = []
  price_test = []

  for i in data:
      if cnt <= 528:
          price_train.append(i)
      else:
          price_test.append(i)
      cnt += 1

  train_data = np.asarray(price_train)
  test_data = np.asarray(price_test)
</code>
</pre>

### 정규화
<pre>
<code>
  mean = train_data.mean(axis=0)
  train_data -= mean
  std = train_data.std(axis=0)
  train_data /= std

  test_data -= mean
  test_data /= std
</code>
</pre>

### 모델구축
<pre>
<code>
  from keras import models
  from keras import layers

  def build_model():
      # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다
      model = models.Sequential()
      model.add(layers.Dense(64, activation='relu',
                             input_shape=(train_data.shape[1],)))
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(1))
      model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])#metrics=['mae']
      return model
</code>
</pre>


### 교차검증(k-flod)
<pre>
<code>
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('처리중인 폴드 #', i)
    # 검증 데이터 준비: k번째 분할
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 훈련 데이터 준비: 다른 분할 전체
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 케라스 모델 구성(컴파일 포함)
    model = build_model()
    # 모델 훈련(verbose=0 이므로 훈련 과정이 출력되지 않습니다)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # 검증 세트로 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
</code>
</pre>

### 테스트
<pre>
<code>
from keras import backend as K

# 메모리 해제
K.clear_session()
</code>
</pre>

<pre>
<code>
num_epochs = 300
all_mae_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)
    # 검증 데이터 준비: k번째 분할
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 훈련 데이터 준비: 다른 분할 전체
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 케라스 모델 구성(컴파일 포함)
    model = build_model()
    # 모델 훈련(verbose=0 이므로 훈련 과정이 출력되지 않습니다)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    #mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(val_mse)
</code>
</pre>

<pre>
<code>
# 새롭게 컴파인된 모델을 얻습니다
model = build_model()
# 전체 데이터로 훈련시킵니다
model.fit(train_data, train_targets,
          epochs=100, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
</code>
</pre>

<pre>
<code>
test_mae_score
</code>
</pre>

<pre>
<code>
x = []
y = []
for i in range(351):
    xhat = test_data[i:i+1] # 평균기온, 최고온도, 평균 현지기압, 평균 중하층운량
    yhat = model.predict(xhat)
    x.append(test_targets[i])
    y.append(yhat[0][0])

res = pd.DataFrame(data = {'실제가격' : x,'예측가격' : y})
res.head()
</code>
</pre>


<pre>
<code>
import matplotlib.pyplot as plt
# matplotlib 폰트설정
# plt.rc('font', family='NanumGothicOTF') # For MacOS
plt.rc('font', family='NanumGothic') # For Windows
print(plt.rcParams['font.family'])
plt.rcParams["figure.figsize"] = (12,8)
plt.title("배추가격 예측하기")
plt.xlabel('날짜')
plt.ylabel('가격')
plt.plot(res['실제가격'], lw=1, label='실제가격')
plt.plot(res['예측가격'], ls='--',lw=0.9, label='예측가격')
plt.legend(loc=1)
plt.show()
</code>
</pre>
