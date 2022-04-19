# Movie Recommendation Baseline Code

영화 추천 대회를 위한 베이스라인 코드입니다. 다음 코드를 대회에 맞게 재구성 했습니다.

- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec

## Installation

```
pip install -r requirements.txt
```

## How to run

1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py
   ```

# level2-movie-recommendation-level2-recsys-06

## ❗ 주제 설명

- 시간 순으로 정렬된 영화 시청 이력에서 중간의 일부 데이터가 누락된 상황일 때, 그 누락된 아이템들과 마지막 아이템을 예측



## 👋 팀원 소개

|[강신구](https://github.com/Kang-singu)|[김백준](https://github.com/middle-100)|[김혜지](https://github.com/h-y-e-j-i)|[이상연](https://github.com/qwedsazxc456)|[전인혁](https://github.com/inhyeokJeon)|
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/Kang-singu) |  [![Avatar](https://user-images.githubusercontent.com/58590260/163910764-69f88ef5-5d66-4cec-ab17-a53b12463c7d.jpg)](https://github.com/middle-100) | [![Avatar](https://user-images.githubusercontent.com/58590260/163910721-c067c68a-9612-4e70-a464-a4bb84eea97e.jpg)](https://github.com/h-y-e-j-i) | [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/qwedsazxc456) | [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/inhyeokJeon) |



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


## ✍ Function Description

`model.py`: EfficientNet-b4와  GoogLeNet을 Ensemble하여 모델링

`dataset.py`: data augmentation, labeling 등 model training에 사용되는 dataset 생성

`loss.py`: cross entropy, f1 score, arcface를 이용해 loss 값을 계산

`train.py`: model을 사용자가 지정한 parameter에 따라 실행하여 training


## 🏢 Structure

```bash
level1-image-classification-level1-recsys-09
│
├── README.md
├── requirements.txt
├── EDA
│   ├── data_EDA.ipynb
│   ├── image_EDA.ipynb
│   └── torchvision_transforms.ipynb
└── python
    ├── dataset.py
    ├── loss.py
    ├── model.py
    └── train.py
```


## ⚙️ Training 명령어
### RecVAE
```python
python run.py --optimizer ‘RAdam’
```
|명령어|타입|설명|기본값|
|------|---|---|---|
|--dataset|str|random seed||
|--hidden-dim|int|number of epochs to train|600|
|--latent-dim|int|dataset augmentation type|300|
|--batch-size|int|data augmentation type|500|
|--beta|float|data augmentation type|CustomAugmentation|
|--gamma|float|data augmentation type|0.005|
|--lr|float|data augmentation type|5e-4|
|--n-epochs|int|data augmentation type|50|
|--n-enc_epochs|int|data augmentation type|3|
|--n-dec_epochs|int|data augmentation type|1|
|--not-alternating|bool|data augmentation type|False|

### EASE
```python
python run_ease.py --lambda 700’
```
|명령어|타입|설명|기본값|
|------|---|---|---|
|--data|str|random seed|"/opt/ml/input/data/train/train_ratings.csv"|
|--output_dir-dim|str|number of epochs to train|"/workspace/output/"|
|--output_file_name|str|dataset augmentation type|"submission_lambda500_top50.csv"|
|--lambda_|float|data augmentation type|500|

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
