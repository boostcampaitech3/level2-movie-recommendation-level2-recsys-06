python preprocessing.py --dataset 데이터의 위치 --output_dir 저장할 곳의 위치 --heldout_users 전체데이터 사용의 경우 0으로 해주시고
python run.py 로 실행하면 됩니다.

|명령어|타입|설명|기본값|
|------|---|---|---|
|--dataset|str|preprocessing한 데이터의 경로||
|--output_dir|str|hidden dim||
|--min_users_per_item|int|latent dim|0|
|--heldout_users|int|batch size||
