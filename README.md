# π¬ Movie Recommendation

## β μ£Όμ  μ€λͺ

- μκ° μμΌλ‘ μ λ ¬λ μν μμ²­ μ΄λ ₯μμ μ€κ°μ μΌλΆ λ°μ΄ν°κ° λλ½λ μν©μΌ λ, κ·Έ λλ½λ μμ΄νλ€κ³Ό λ§μ§λ§ μμ΄νμ μμΈ‘



## π νμ μκ°

|[κ°μ κ΅¬](https://github.com/Kang-singu)|[κΉλ°±μ€](https://github.com/middle-100)|[κΉνμ§](https://github.com/h-y-e-j-i)|[μ΄μμ°](https://github.com/qwedsazxc456)|[μ μΈν](https://github.com/inhyeokJeon)|
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [![Avatar](https://user-images.githubusercontent.com/58590260/163955612-1e3c1752-9c68-4cb1-af8f-c99b99625750.jpg)](https://github.com/Kang-singu) |  [![Avatar](https://user-images.githubusercontent.com/58590260/163910764-69f88ef5-5d66-4cec-ab17-a53b12463c7d.jpg)](https://github.com/middle-100) | [![Avatar](https://user-images.githubusercontent.com/58590260/163910721-c067c68a-9612-4e70-a464-a4bb84eea97e.jpg)](https://github.com/h-y-e-j-i) | [![Avatar](https://user-images.githubusercontent.com/58590260/163955925-f5609908-6984-412f-8df6-ae490517ddf4.jpg)](https://github.com/qwedsazxc456) | [![Avatar](https://user-images.githubusercontent.com/58590260/163956020-891ce159-3233-469d-a83c-4c0926ec438a.jpg)](https://github.com/inhyeokJeon) |



## π¨ Installation

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

## π’ Structure

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
## π©βπ¬ μ°κ΅¬κ³Όμ 

<p align="center"><img src="https://user-images.githubusercontent.com/58590260/175431028-d01fcc87-c977-43b0-aa1b-e2769b6f8669.png" width=1000></p>


## π EDA

- **Train**
 - User, Item, TimeμΌλ‘ λλμ΄μ Έ μμ΅λλ€.
 - μ μ λ 6807λͺ, μμ΄ν(μν)λ 31360κ° μ’λ₯μλλ€.
 - κ°μ₯ μ κ² λ³Έ μν κ°μλ 16κ°, κ°μ₯ λ§μ΄ λ³Έ μν κ°μλ 2912κ°μλλ€.
 - μκ° μμ²­ μκ°μ 2005-04-11 ~ 2015-03-31κΉμ§ μλλ€.
- **Titles**
 - βμνμ΄λ¦(μ°λ)βμ νμμ κ°μ§κ³  μμ΅λλ€.
 - trainμ μμ΄ν(μν) μ»¬λΌμ κΈ°μ€μΌλ‘ mergeν κ²°κ³Ό, nullκ°μ΄ μ‘΄μ¬νμ§ μμμ΅λλ€. λ°λΌμ κ²°μΈ‘μΉκ° μμ΅λλ€.
- **Years**
  - 1902λλΆν° 2014λκΉμ§ μμ΅λλ€.\
  ![image](https://user-images.githubusercontent.com/58590260/175431075-4ba032c8-232e-46ec-a6b1-80d17f255452.png)
  - κ·Έλνλ‘ κ·Έλ €λ³Έ κ²°κ³Ό, 2010λλμ μνμ΄ λ§μ κ²μ μ μ μμ΅λλ€.
 - trainμ μμ΄ν(μν) μ»¬λΌμ κΈ°μ€μΌλ‘ mergeν κ²°κ³Ό, nullκ°μ΄ μ‘΄μ¬νμ΅λλ€. κ·Έλ¬λ μνμ΄λ¦μ μ°λκ° μκΈ° λλ¬Έμ λ°μ΄ν°λ₯Ό μ±μ λ£μ μ μμ΅λλ€.
- **Genres**
    - μ΄ 18κ°μ μ₯λ₯΄κ° μμ΅λλ€.
    - μ¬λλ€μ λλΌλ§ μ₯λ₯΄λ₯Ό μ μΌ μ νΈνκ³ , λμλ₯΄ μ₯λ₯΄λ₯Ό μ μΌ μ νΈνμ§ μμ΅λλ€.\
    ![image](https://user-images.githubusercontent.com/58590260/175431196-14c51c2f-f48e-4fe6-b9d7-d585f11269c1.png)
    - trainμ μμ΄ν(μν) μ»¬λΌμ κΈ°μ€μΌλ‘ mergeν κ²°κ³Ό, nullκ°μ΄ μ‘΄μ¬νμ§ μμμ΅λλ€.
- **Writers**
    - 2989λͺμ μκ°μ 5648κ°μ μν μ λ³΄κ° μμ΅λλ€.
    - μ΅λ 47κ°, μ΅μ 2κ°μ μνλ₯Ό μ§ννμμ΅λλ€.
    - ν μνμ μ΅λ 24λͺ, μ΅μ 1λͺμ μκ°κ° μ§ννμμ΅λλ€.
    - trainμ μμ΄ν(μν) μ»¬λΌμ κΈ°μ€μΌλ‘ mergeν κ²°κ³Όμ΄ μ‘΄μ¬νμ΅λλ€. μ΄ 1159κ°μ μνμ μκ°μ μ λ³΄κ° μμμ΅λλ€.
- **Directors**
    - 1340λͺμ κ°λκ³Ό 5503κ°μ μν μ λ³΄κ° μμ΅λλ€.
    - μ΅λ 44κ°, μ΅μ 2κ°μ μνμ μ°Έμ¬νμμ΅λλ€
    - ν μνμ μ΅λ 14λͺ, μ΅μ 1λͺμ κ°λμ΄ μ°Έμ¬νμμ΅λλ€.
    - trainμ μμ΄ν(μν) μ»¬λΌμ κΈ°μ€μΌλ‘ mergeν κ²°κ³Όμ΄ μ‘΄μ¬νμ΅λλ€. μ΄ 1304κ°μ μνμ κ°λμ μ λ³΄κ° μμμ΅λλ€.

### β EDA κ²°κ³Ό

- Writerμ Directorsλ λ§μ κ²°μΈ‘μΉλ‘ μΈν΄ μν λΆλ΄μ΄ μ‘΄μ¬νμ΅λλ€. λ°λΌμ Side informationλ **Title, Years, Genres**λ₯Ό μ¬μ©ν΄λ³΄μλ κ²°λ‘ μ΄ λμμ΅λλ€.
- μμ²­μμ μ΅κ·Ό μ·¨ν₯λ§ λ°μνκΈ° μν΄ **μ΅κ·Όμ μμ²­ν μν λ°μ΄ν°λ§ μ¬μ©**νμ¬ μ€νν΄λ³΄κΈ°λ‘ νμ΅λλ€.
- μνμ κ°―μμ λ°λΌ λͺ¨λΈμ μ±λ₯μ μν₯μ΄ μμ μλ μμΌλ, μμλΈ μ, **μμ²­ν μνμ κ°―μμ  λ°λΌ λͺ¨λΈμ κ°μ€μΉμ λ¬λ¦¬ μ£Όλ μ€νμ μ§νν΄λ³΄κΈ°λ‘ νμ΅λλ€.**
- μνλ₯Ό λ§μ΄ λ³Έ μ¬λλ€μ΄ λͺ¨λΈ νλ ¨μ λ°©ν΄κ° λ  μ λ μμΌλ, **μνλ₯Ό λ§μ΄ λ³Έ μ¬λμ μ μΈνκ³ ** λͺ¨λΈμ νμ΅μμΌλ³΄κΈ°λ‘ νμ΅λλ€.

## π Modeling
### 1οΈβ£ Model

- S3Rec, BERT4Rec, Multi-VAE, RecVAE, CF, H+Vamp, EASE
- SOTA λͺ¨λΈμΈ S3Recμ μ¬μ©νμ¬ recall@10 0.09μ μ±λ₯μ λνλ¬μ΅λλ€.
- EDAκ²°κ³Όμ λ°λΌ **sequential λͺ¨λΈμ νμ¬ taskμ μ ν©νμ§ μλ€κ³  νλ¨νμ¬** memory basedλ°©λ²μ **jaccard μ μ¬λλ₯Ό μ΄μ©**νμ¬ μ€ννμ¬ μ±λ₯μ ν₯μνμμ΅λλ€
- μ±λ₯μ λ ν₯μ μν€κΈ° μν΄ **Deep learning κΈ°λ° AutoEncoder λͺ¨λΈμΈ RecVAE, H+VAMP, Multi-VAEλ₯Ό μ¬μ©**νμ¬ μ€νμ μ§νν κ²°κ³Ό **RecVAEκ° μ±λ₯μ΄ μ μΌ μ’μμ΅λλ€.**
- Side information μ μ¬μ©νκΈ° μν΄ DeepFMμ μ μ© μμΌ°μ§λ§ μ±λ₯μ΄ μ’μ§ μμμ΅λλ€.
- **Neighborhood CFκΈ°λ° EASEλͺ¨λΈμ μ€νν΄λ³Έ κ²°κ³Ό, λμ μ±λ₯μ λ³΄μμ΅λλ€.**

### 2οΈβ£ Hyperparameter tuning
- Wandbμ μ μΆμ ν΅ν΄ λͺ¨λΈμ νμ΄νΌ νλΌλ―Έν° νλμ νμμ΅λλ€.

#### **2-1. optimizer**
<p align="center"><img src="https://user-images.githubusercontent.com/58590260/175431358-57e8a2ef-56d5-47ef-9635-81d69a64b9f5.png" width=700></p>

- Wandbλ₯Ό ν΅ν΄ Recall@10μ κ²°κ³Όκ° κ°μ₯ λμλ NAdam, Adamax, AdamW, RAdam, RMSpropμ μ¬λ¬ λͺ¨λΈμ μ¬μ©νμ¬ μ€ννμμ΅λλ€.
- κ·Έ κ²°κ³Ό, RecVAEμ MultiVAEλ optimizerκ° RAdamμΌ λ μ±λ₯μ΄ κ°μ₯ μ’μμ΅λλ€.

#### 2-2. batch size

- λλΆλΆμ λͺ¨λΈμ΄ batch sizeκ° 16μ΄λ 32μΌ λ μ’μ μ±λ₯μ λ³΄μμ΅λλ€.
- BERT4Recμ batch sizeκ° 16μΌ λ, H+Vampμ Multi-VAEλ 32μΌ λ μ±λ₯μ΄ μ’μμ΅λλ€.

#### 2-3. κ·Έ μΈμ νλΌλ―Έν°

- EASEλ lambdaκ° 500μΌ λ κ°μ₯ μ’μ μ±λ₯μ λ³΄μμ΅λλ€.
- RecVAEλ laten dimμ΄ 250, hidden dimμ΄ 600, betaκ° 0.4μΌ λ κ°μ₯ μ’μ μ±λ₯μ λ³΄μμ΅λλ€.
- H+Vampλ betaκ° 0.3, Gatedκ° TrueμΌ λ μ±λ₯μ΄ κ°μ₯ μ’μμ΅λλ€.

#### 2-3. Ensemble
- λͺ¨λΈλ€μ rating matrixμ μ κ·ν μμμν λ€ μμλΈμ μ§ννμ΅λλ€.
- κ²°κ³Όμ μΌλ‘, **EASEλ₯Ό κ°μ€μΉλ‘ μ¬μ©νλ VASP λΌλ¬Έμ μμ΄λμ΄λ₯Ό μ μ©ν RecVAE, EASEμ μμλΈμ μ±λ₯μ΄ μ μΌ μ’μμ΅λλ€**. λλͺ¨λΈμ matrixμμ μμμΈ rating scoreμλ§ κ³μ°νκ³ , μ΄ μΈμλ 0μΌλ‘ μ²λ¦¬νμμ΅λλ€.


## πΌοΈ μ€ν κ²°κ³Ό

| λͺ¨λΈλͺ | Recall@10 | μ΅μ’ μμ |
| --- | --- | --- |
| RecVAE + EASE μμλΈ | 0.1630 | private 6λ± |


## π μ°Έκ³ μλ£
1. Diane Bouchacourt,Β Ryota Tomioka,Β Sebastian Nowozin, 2017. Multi-Level Variational Autoencoder: Learning Disentangled Representations from Grouped Observations
2. Dawen Liang,Β Rahul G. Krishnan,Β Matthew D. Hoffman,Β Tony Jebara, 2018. Variational Autoencoders for Collaborative Filtering
3. Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
4. Wang-Cheng Kang,Β Julian McAuley, 2018. Self-Attentive Sequential Recommendation
5. Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang, 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer
6. Ilya Shenbin,Β Anton Alekseev,Β Elena Tutubalina,Β Valentin Malykh,Β Sergey I. Nikolenko, 2019. RecVAE: a New Variational Autoencoder for Top-N Recommendations with Implicit Feedback
7. Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data
8. Daeryong Kim,Β Bongwon Suh, 2019. Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms
9. Pavel Kordik, Vojtech Vancura, 2021. Deep Variational Autoencoder with Shallow Parallel Path for Top-N Recommendation (VASP)

 
