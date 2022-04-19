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
|  [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/Kang-singu) |  [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/middle-100) | [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/h-y-e-j-i) | [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/qwedsazxc456) | [![Avatar](https://user-images.githubusercontent.com/92902312/163906431-df327c34-1518-41ee-9bd7-2a525750be94.png)](https://github.com/inhyeokJeon) |



## 🔨 Installation

- torch == 1.6.0
- torchvision == 0.7.0
- tensorboard == 2.4.1
- pandas == 1.1.5
- opencv-python == 4.5.1.48
- scikit-learn ~= 0.24.1
- matplotlib == 3.2.1
- efficientnet_pytorch

```python
$ pip install -r $ROOT/level1-image-classification-level1-recsys-09/requirements.txt
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

```python
python train.py --model 'Ensemble' --TTA True --name 'final model' --epoch 3
```

![image](https://user-images.githubusercontent.com/44939208/157379480-737623fe-8237-47bc-8c4a-03897a8fd3e9.png)


## 🖼️ 실행 결과

| 모델명 | F1-Score | Accuracy | 최종 순위 |
| --- | --- | --- | --- |
| EfficientNet-b4 + GoogLeNet | 0.7269 | 77.3016 | private 35등 |


## 📜 참고자료

[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

[GoogLeNet](https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html)
