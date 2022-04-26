# Movie Recommendation EASE Model Code

Embarrassingly Shallow Autoencoders for Sparse Data

This model is cool because it is a closed-form solution.

Similiarity matrix is calculated on CPU with numpy.

model paper : https://arxiv.org/pdf/1905.03375v1.pdf

paperwithcode : https://paperswithcode.com/paper/190503375

referred github : https://github.com/Darel13712/ease_rec

## 🏃 run
1. Run and make csv file
```
python run_ease.py \
--data "data_path" --output_dir "output_path" \
----output_file_name "output_file_name" --labmda_ "lambda"
```

2. make all predicted item per user (OPTION)\
```--all_predict True```
   
3. if you want to use YEAR column (OPTION)\
```--use_year True```

|명령어|타입|설명|기본값|
|------|---|---|---|
|--data|str|data path|"/opt/ml/input/data/train/train_ratings.csv"|
|--output_dir|str|output dir|"/workspace/output/"|
|--output_file_name|str|output name|"submission_lambda500.csv"|
|--lambda_|float|regularization|500|


## 🍽 Input & output
### 🍴 Input
```csv file``` with columns ```user_id``` and ```item_id``` both for fit and predict.

It may also use ```ratings``` from column rating if ```implicit``` parameter is set to ```False```.

### 🥄 Output
```csv file``` with columns user_id, item_id


## 🖼️ final model
```python
python run_ease.py --lambda 500
```