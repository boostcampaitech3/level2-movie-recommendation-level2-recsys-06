import pandas as pd
import argparse
from models import EASE, EASE2
import wandb

def WandB():
    wandb.init(
        # 필수
        project="MovieLens",  # project Name
        entity="recsys-06",  # Repository 느낌 변경 X
        name="EASE_TEST",  # -> str : ex) "모델_파라티머_파라미터_파라미터", 훈련 정보에 대해 알아보기 쉽게
        notes="with years",  # -> str commit의 메시지 처럼 좀 더 상세한 설명 log
        group="EASE",
        # 추가 요소
        # tags -> str[] baseline, production등 태그 기능.
        # save_code -> bool 코드 저장할지 말지 default false
        # group -> str : 프로젝트내 그룹을 지정하여 개별 실행을 더 큰 실험으로 구성, k-fold교차, 다른 여러 테스트 세트에 대한 모델 훈련 및 평가 가능.

        #  more info
        # https://docs.wandb.ai/v/ko/library/init
    )

def main():
    # WandB()
    parser = argparse.ArgumentParser()
    # for EASE2 "/workspace/output/user_item_last_year_final.csv"
    parser.add_argument("--data", default="/opt/ml/input/data/train/train_ratings.csv", type=str)
    parser.add_argument("--output_dir", default="/workspace/output/", type=str)
    parser.add_argument("--output_file_name", default="submission_lambda500_top50.csv", type=str)
    parser.add_argument("--use_year", default=False, type=bool)
    parser.add_argument("--all_predict", default=False, type=bool)

    parser.add_argument("--lambda_", default=500, type=float)

    args = parser.parse_args()
    # wandb.config.update(args)

    train_df = pd.read_csv(args.data)
    if args.use_year:
        ease = EASE2()
    else:
        ease = EASE()
    users = train_df["user"].unique()
    items = train_df["item"].unique()
    ease.fit(train_df, lambda_=args.lambda_)

    result_df = ease.predict(train_df, users, items, 10)

    result_df[["user", "item"]].to_csv(
        args.output_dir + args.output_file_name, index=False
    )

    if args.all_predict:
        ease.make_all_predicted()


if __name__ == "__main__":
    main()