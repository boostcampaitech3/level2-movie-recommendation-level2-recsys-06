python preprocessing.py 을 실행한 후
python run.py 로 실행하면 됩니다.

preprocessing의 argparser

|명령어|타입|설명|기본값|
|------|---|---|---|
|--dataset|str|preprocessing할 데이터||
|--output_dir|str|preprocessing된 데이터의 저장 위치||
|--heldout_users|int|전체 데이터 사용은 0||

run의 argparser

|명령어|타입|설명|기본값|
|------|---|---|---|
|--dataset|str|preprocessing한 데이터의 경로||
|--hidden-dim|int|hidden dim|600|
|--latent-dim|int|latent dim|300|
|--batch-size|int|batch size|500|
|--beta|float|beta의 크기 0~1사이|None|
|--gamma|float|gamma 0으로 beta 사용|0.005|
|--lr|float|learning rate|5e-4|
|--n-epochs|int|epoch수|50|
|--n-enc_epochs|int|encoding 부분 dropout|3|
|--n-dec_epochs|int|decoding 부분 dropout|1|
|--not-alternating|bool|--n-enc_epochs/--n-dec_epochs 사용여부|False|
