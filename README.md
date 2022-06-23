# 🎬 Movie Recommendation

## ❗ 주제 설명

- 시간 순으로 정렬된 영화 시청 이력에서 중간의 일부 데이터가 누락된 상황일 때, 그 누락된 아이템들과 마지막 아이템을 예측



## 👋 팀원 소개

|[강신구](https://github.com/Kang-singu)|[김백준](https://github.com/middle-100)|[김혜지](https://github.com/h-y-e-j-i)|[이상연](https://github.com/qwedsazxc456)|[전인혁](https://github.com/inhyeokJeon)|
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [![Avatar](https://user-images.githubusercontent.com/58590260/163955612-1e3c1752-9c68-4cb1-af8f-c99b99625750.jpg)](https://github.com/Kang-singu) |  [![Avatar](https://user-images.githubusercontent.com/58590260/163910764-69f88ef5-5d66-4cec-ab17-a53b12463c7d.jpg)](https://github.com/middle-100) | [![Avatar](https://user-images.githubusercontent.com/58590260/163910721-c067c68a-9612-4e70-a464-a4bb84eea97e.jpg)](https://github.com/h-y-e-j-i) | [![Avatar](https://user-images.githubusercontent.com/58590260/163955925-f5609908-6984-412f-8df6-ae490517ddf4.jpg)](https://github.com/qwedsazxc456) | [![Avatar](https://user-images.githubusercontent.com/58590260/163956020-891ce159-3233-469d-a83c-4c0926ec438a.jpg)](https://github.com/inhyeokJeon) |



## 🔨 Installation

- numpy==1.22.2
- pandas==1.4.1
- pytz==2021.3
- python-dateutil==2.8.2
- scipy==1.8.0
- six==1.16.0
- torch==1.10.2
- tqdm==4.62.3
- typing_extensions==4.1.1
- Python==3.8.5

```python
$ pip install -r $ROOT/level2-movie-recommendation-level2-recsys-06/requirements.txt
```

## 🏢 Structure

```bash
level2-movie-recommendation-level2-recsys-06
|-- EASE
|   |-- README.md
|   |-- models.py|   
|   `-- run_ease.py
|-- EDA
|   `-- EDA_hyeji.ipynb
|-- Ensemble
|   `-- ensemble.ipynb
|-- README.md
|-- RecVAE
|   |-- README.md
|   |-- model.py
|   |-- preprocesing.py
|   |-- run.py
|   `-- utils.py
`-- requirements.txt
```
## 👩‍🔬 연구과정

<p align="center"><img src="https://user-images.githubusercontent.com/58590260/175431028-d01fcc87-c977-43b0-aa1b-e2769b6f8669.png" width=1000></p>


## 🔎 EDA

- **Train**
 - User, Item, Time으로 나뉘어져 있습니다.
 - 유저는 6807명, 아이템(영화)는 31360개 종류입니다.
 - 가장 적게 본 영화 개수는 16개, 가장 많이 본 영화 개수는 2912개입니다.
 - 시간 시청 시각은 2005-04-11 ~ 2015-03-31까지 입니다.
- **Titles**
 - ‘영화이름(연도)’의 형식을 가지고 있습니다.
 - train의 아이템(영화) 컬럼을 기준으로 merge한 결과, null값이 존재하지 않았습니다. 따라서 결측치가 없습니다.
- **Years**
  - 1902년부터 2014년까지 있습니다.\
  ![image](https://user-images.githubusercontent.com/58590260/175431075-4ba032c8-232e-46ec-a6b1-80d17f255452.png)
  - 그래프로 그려본 결과, 2010년대의 작품이 많은 것은 알 수 있습니다.
 - train의 아이템(영화) 컬럼을 기준으로 merge한 결과, null값이 존재했습니다. 그러나 영화이름에 연도가 있기 때문에 데이터를 채워 넣을 수 있습니다.
- **Genres**
    - 총 18개의 장르가 있습니다.
    - 사람들은 드라마 장르를 제일 선호하고, 누아르 장르를 제일 선호하지 않습니다.\
    ![image](https://user-images.githubusercontent.com/58590260/175431196-14c51c2f-f48e-4fe6-b9d7-d585f11269c1.png)
    - train의 아이템(영화) 컬럼을 기준으로 merge한 결과, null값이 존재하지 않았습니다.
- **Writers**
    - 2989명의 작가와 5648개의 영화 정보가 있습니다.
    - 최대 47개, 최소 2개의 영화를 집필하였습니다.
    - 한 영화에 최대 24명, 최소 1명의 작가가 집필하였습니다.
    - train의 아이템(영화) 컬럼을 기준으로 merge한 결과이 존재했습니다. 총 1159개의 영화에 작가의 정보가 없었습니다.
- **Directors**
    - 1340명의 감독과 5503개의 영화 정보가 있습니다.
    - 최대 44개, 최소 2개의 작품에 참여하였습니다
    - 한 영화에 최대 14명, 최소 1명의 감독이 참여하였습니다.
    - train의 아이템(영화) 컬럼을 기준으로 merge한 결과이 존재했습니다. 총 1304개의 영화에 감독의 정보가 없었습니다.

### ❗ EDA 결과

- Writer와 Directors는 많은 결측치로 인해 위험 부담이 존재했습니다. 따라서 Side information는 **Title, Years, Genres**를 사용해보자는 결론이 나왔습니다.
- 시청자의 최근 취향만 반영하기 위해 **최근에 시청한 영화 데이터만 사용**하여 실험해보기로 했습니다.
- 영화의 갯수에 따라 모델의 성능에 영향이 있을 수도 있으니, 앙상블 시, **시청한 영화의 갯수에  따라 모델의 가중치을 달리 주는 실험을 진행해보기로 했습니다.**
- 영화를 많이 본 사람들이 모델 훈련에 방해가 될 수 도 있으니, **영화를 많이 본 사람을 제외하고** 모델을 학습시켜보기로 했습니다.

## 🏆 Modeling
### 1️⃣ Model

- S3Rec, BERT4Rec, Multi-VAE, RecVAE, CF, H+Vamp, EASE
- SOTA 모델인 S3Rec을 사용하여 recall@10 0.09의 성능을 나타났습니다.
- EDA결과에 따라 **sequential 모델은 현재 task에 적합하지 않다고 판단하여** memory based방법을 **jaccard 유사도를 이용**하여 실험하여 성능을 향상하였습니다
- 성능을 더 향상 시키기 위해 **Deep learning 기반 AutoEncoder 모델인 RecVAE, H+VAMP, Multi-VAE를 사용**하여 실험을 진행한 결과 **RecVAE가 성능이 제일 좋았습니다.**
- Side information 을 사용하기 위해 DeepFM을 적용 시켰지만 성능이 좋지 않았습니다.
- **Neighborhood CF기반 EASE모델을 실험해본 결과, 높은 성능을 보였습니다.**

### 2️⃣ Hyperparameter tuning
- Wandb와 제출을 통해 모델의 하이퍼 파라미터 튜닝을 하였습니다.

#### **2-1. optimizer**
<p align="center"><img src="https://user-images.githubusercontent.com/58590260/175431358-57e8a2ef-56d5-47ef-9635-81d69a64b9f5.png" width=700></p>

- Wandb를 통해 Recall@10의 결과가 가장 높았던 NAdam, Adamax, AdamW, RAdam, RMSprop을 여러 모델에 사용하여 실험하였습니다.
- 그 결과, RecVAE와 MultiVAE는 optimizer가 RAdam일 때 성능이 가장 좋았습니다.

#### 2-2. batch size

- 대부분의 모델이 batch size가 16이나 32일 때 좋은 성능을 보였습니다.
- BERT4Rec은 batch size가 16일 때, H+Vamp와 Multi-VAE는 32일 때 성능이 좋았습니다.

#### 2-3. 그 외의 파라미터

- EASE는 lambda가 500일 때 가장 좋은 성능을 보였습니다.
- RecVAE는 laten dim이 250, hidden dim이 600, beta가 0.4일 때 가장 좋은 성능을 보였습니다.
- H+Vamp는 beta가 0.3, Gated가 True일 때 성능이 가장 좋았습니다.

#### 2-3. Ensemble
- 모델들의 rating matrix을 정규화 작업을한 뒤 앙상블을 진행했습니다.
- 결과적으로, **EASE를 가중치로 사용하는 VASP 논문의 아이디어를 적용한 RecVAE, EASE의 앙상블의 성능이 제일 좋았습니다**. 두모델의 matrix에서 양수인 rating score에만 계산하고, 이 외에는 0으로 처리하였습니다.


## 🖼️ 실행 결과

| 모델명 | Recall@10 | 최종 순위 |
| --- | --- | --- |
| RecVAE + EASE 앙상블 | 0.1630 | private 6등 |


## 📜 참고자료
1. Diane Bouchacourt, Ryota Tomioka, Sebastian Nowozin, 2017. Multi-Level Variational Autoencoder: Learning Disentangled Representations from Grouped Observations
2. Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara, 2018. Variational Autoencoders for Collaborative Filtering
3. Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
4. Wang-Cheng Kang, Julian McAuley, 2018. Self-Attentive Sequential Recommendation
5. Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang, 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer
6. Ilya Shenbin, Anton Alekseev, Elena Tutubalina, Valentin Malykh, Sergey I. Nikolenko, 2019. RecVAE: a New Variational Autoencoder for Top-N Recommendations with Implicit Feedback
7. Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data
8. Daeryong Kim, Bongwon Suh, 2019. Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms
9. Pavel Kordik, Vojtech Vancura, 2021. Deep Variational Autoencoder with Shallow Parallel Path for Top-N Recommendation (VASP)

 
