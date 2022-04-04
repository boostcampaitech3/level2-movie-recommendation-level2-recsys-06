# MultiVAE(hyeji)
- 가장 좋은 모델 : MultiVAE-Ml_data_random_train_all_100.pt
   - python run_train.py --optimizer 'RAdam' --batch_size 32 --train_all True  --seed 960708 
- TODO tree에서 XXX로 표시한 부분이 수정한 부분입니다.
## run_train.py
- 결과를 뽑기 위한 evaluate_submission 함수가 추가되었습니다.
- args.parser가 추가됐습니다.
- dafault값은 모두 기존 코드와 동일한 기본값입니다.
```python
parser.add_argument("--wandb", type=bool, default=False, help="wandb") # wandb 사용 여부
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)') # optimizer 설정
parser.add_argument('--data_process', type=int, default=0,  help='data process') # 최근 데이터를 얼마나 사용할 것인가
parser.add_argument('--data_random_process', type=int, default=0,  help='data random process') # 데이터를 어느 비율만큼 랜덤으로 뽑을 것인가
parser.add_argument('--train_all', type=bool, default=False,  help='use all training set') # 훈련데이터를 모두 쓸 것인지
```
- 모델 이름 방식은 다음과 같습니다.
```python
if args.data_process != 0 : # 최근 데이터 일부만 사용하는 경우
   # 모델이름-데이터이름_data_최근 데이터 갯수
   checkpoint = f"{args_str}_data_{args.data_process}.pt"
elif args.data_random_process != 0 : # 최근 데이터 일부만 사용하는 경우
   # 모델이름-데이터이름_data_random_비율
   checkpoint = f"{args_str}_data_random_{args.data_random_process}.pt"
elif args.train_all == True:
   checkpoint = f"{args_str}_data_random_train_all_100.pt"
else : # 기본값
   # 모델이름-데이터이름_epoch 수_optimizer.pt
   checkpoint = f"{args_str}_{args.epochs}_{args.optimizer}.pt"
```
- submission 파일 이름 방식은 다음과 같습니다.
```python
if args.data_process != 0: # 최근 데이터 일부만 사용한 경우
   # submission_data_최근 데이터 갯수.csv
   result.to_csv(os.path.join(args.output_dir, f'submission_data_{args.data_process}.csv'), index=False)
   print("export submission : ", os.path.join(args.output_dir, f'submission_data_{args.data_process}.csv'))
elif args.data_random_process != 0:
   result.to_csv(os.path.join(args.output_dir, f'submission_data_random_{args.data_random_process}.csv'), index=False)
   print("export submission : ", os.path.join(args.output_dir, f'submission_data_random_{args.data_random_process}.csv'))
else : # 기본값             
   # submission_epoch 수_optimizer.csv
   result.to_csv(os.path.join(args.output_dir, f'submission_{args.epochs}_{args.optimizer}.csv'), index=False)
   print("export submission : ", os.path.join(args.output_dir, f'submission_{args.epochs}_{args.optimizer}.csv'))
```
## export_submission_data.py
- 모델 평가 input에 넣기 위해 train_rating.csv을 변환하는 코드입니다.
- /opt/ml/input/data/eval/submission_data2.csv 에 저장됩니다.

## train_data_processing.py, train_data_processing.ipynp
- 최근 데이터만 뽑아서 훈련, 검증, 테스트셋을 만듭니다.
- py인 경우 뽑을 데이터 갯수를 터미널 창에 입력하면 "start"가 뜨면서 처리를 시작합니다
- ipynp는 MOVIE_COUNT 함수의 값을 원하는 데이터 갯수로 바꾸시면 됩니다.

## train_data_random_processing.py, train_data_random_processing.ipynp
- 입력받은 비율로 랜덤으로 뽑아서 훈련, 검증, 테스트셋을 만듭니다.
- py인 경우 뽑을 비율을 터미널 창에 입력하면 "start"가 뜨면서 처리를 시작합니다
- ipynp는 MOVIE_COUNT 함수의 값을 원하는 비율로 바꾸시면 됩니다.

## train_data_all_part_processing.ipynb
- 데이터를 모두 훈련데이터로 변환합니다

## train_data_all_part_processing.ipynp
- 데이터 중 일부만 훈련데이터로 사용합니다.