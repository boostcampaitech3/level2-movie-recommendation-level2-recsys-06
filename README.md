# RecVAE
1.  preprocesing.py에서--dataset, --output_dir 주소 설정하고 --heldout_user 0 으로 설정하면
2. train.csv라는 파일과 unique_sid.txt , unique_uid.txt 파일이 생성됨
3. txt파일을 csv파일로 변환 후,
4. run,py로 가셔서 --dataset을 설정하고
5. --hidden-dim, --latent-dim, --batch-size 바꾸시면서 --beta값 0에서 1사이값 --gamma는 0으로 하시면 됩니다. 
6. preprocesing.py에서 random seed 안쓰게 해놔서 하실분은 주석 없애시고 run에서 마지막줄 주석 없애시면 결과 csv파일 나옵니다
