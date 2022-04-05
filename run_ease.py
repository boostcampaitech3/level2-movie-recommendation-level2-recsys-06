import pandas as pd
import argparse
from models import EASE


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/opt/ml/input/data/train/train_ratings.csv", type=str)
    parser.add_argument("--output_dir", default="/workspace/TorchEASE/output/", type=str)
    parser.add_argument("--output_file_name", default="submission4.csv", type=str)

    args = parser.parse_args()

    train_df = pd.read_csv(args.data)
    ease = EASE()
    users = train_df["user"].unique()
    items = train_df["item"].unique()
    ease.fit(train_df)
    result_df = ease.predict(train_df, users, items, 10)
    result_df[["user", "item"]].to_csv(
        args.output_dir + args.output_file_name, index=False
    )


if __name__ == "__main__":
    main()