import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
import wandb
from utils import ndcg_k, recall_at_k


class Trainer:
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
    ):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        post_fix_org = {
            "Epoch": epoch,
            "RECALL@5": float("{:.8f}".format(recall[0])),
            "NDCG@5": float("{:.8f}".format(ndcg[0])),
            "RECALL@10": float("{:.8f}".format(recall[1])),
            "NDCG@10": float("{:.8f}".format(ndcg[1])),
        }
        wandb.log(post_fix_org, step=epoch)
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    # process 5 Loss 계산 과정
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # process 5-1 Loss 함수에 들어오는 입력값
        # [batch seq_len hidden_size] Cube 모양
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # process 5-2 계산을 쉽게 하기위해 펼치기
        # [batch*seq_len, hidden_size] 2 차원 matrix 모양 펼침(view를 사용해 펼침)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len, hidden_size]

        # process 5-3 예측한 영화와 실제 영화의 유사도 구하기
        # 각 요소를 곱한 수 다 더해준다.
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)

        # process 5-4 Padding을 무시하고 계산하기 위해 Mask 생성.
        # 실제 interaction만 추출.
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]

        # process 5-5 Cross Entropy 계산하기
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget  # sigmoid -> 1에 가까울 수록 좋다 -> pos_logits은 양수로 크게
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget  # 0에 가까울 수록 좋다 -> neg_logits은 음수로 크게, 1e-24 -> log(0) 방지용
        ) / torch.sum(istarget)

        return loss

    # process 6-1 마지막 유저에 대한 임베딩과 영화에 대한 유사도를 구해 전체 영화에 대한 점수를 구한다.
    # [Users, hidden_state] X [hidden_state, Movices]
    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # TODO matmul 대신 cosine 유사도로 해볼까?
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class PretrainTrainer(Trainer):
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
    ):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def pretrain(self, epoch, pretrain_dataloader):
        desc = (
            f"AAP-{self.args.aap_weight}-"
            f"MIP-{self.args.mip_weight}-"
            f"MAP-{self.args.map_weight}-"
            f"SP-{self.args.sp_weight}"
        )

        pretrain_data_iter = tqdm.tqdm(
            enumerate(pretrain_dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
            total=len(pretrain_dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            (
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            ) = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            )
            # process 10 Pretraining 마무리.
            # 모든 Loss를 합쳐서 한번에 고려, 이때 argparse로 loss 각각의 가중치를 조절 가능
            joint_loss = (
                    self.args.aap_weight * aap_loss
                    + self.args.mip_weight * mip_loss
                    + self.args.map_weight * map_loss
                    + self.args.sp_weight * sp_loss
            )

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        losses = {
            "epoch": epoch,
            "aap_loss_avg": aap_loss_avg / num,
            "mip_loss_avg": mip_loss_avg / num,
            "map_loss_avg": map_loss_avg / num,
            "sp_loss_avg": sp_loss_avg / num,
        }
        print(desc)
        print(str(losses))
        return losses


class FinetuneTrainer(Trainer):
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
    ):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            wandb.watch(self.model)
            # process 4 본격적인 훈련
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch  # _ -> user_ids 인듯
                # process 4-1 배치단위 훈련
                # user_ids = [Batch]
                # input_ids = [Batch, Seq Len]
                # target_pos = [Batch, Seq Len]
                # target_neg = [Batch, Seq Len]
                # answers = [Batch, 1]

                # [Batch, Seq Len, Hidden Size] Hidden Size 에 대한 그림설명 - 오피스아워 8분 50초.
                sequence_output = self.model.finetune(input_ids)

                # process 5
                loss = self.cross_entropy(sequence_output, target_pos, target_neg) # Binary cross_entropy


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                # print("rec_avg_loss : ",rec_avg_loss, type(rec_avg_loss))
                # print("rec_avg_loss : ",rec_cur_loss, type(rec_cur_loss))
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }
            post_fix_org = {
                "epoch": epoch,
                "rec_avg_loss": float("{:.8f}".format(rec_avg_loss / len(rec_data_iter))),
                "rec_cur_loss": float("{:.8f}".format(rec_cur_loss)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                wandb.log(post_fix_org, step=epoch)
                print(str(post_fix))

        else:
            self.model.eval()
            pred_list = None
            answer_list = None
            # process 6 예측 과정
            for i, batch in rec_data_iter:

                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, target_neg, answers = batch
                recommend_output = self.model.finetune(input_ids)

                recommend_output = recommend_output[:, -1, :]  # 예측을 위한 마지막 영화

                rating_pred = self.predict_full(recommend_output) # 예측 6

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0  # 봤던 영화를 제거한다.

                ind = np.argpartition(rating_pred, -10)[:,
                      -10:]  # 10 개를 구한다 단 순위는 보장 못한다. np.argpartition -> index return

                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)),
                                  ::-1]  # 10개 뽑은 것을 정렬한다. -> index return

                batch_pred_list = ind[  # 예측한 10개
                    np.arange(len(rating_pred))[:, None], arr_ind_argsort
                ]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(
                        answer_list, answers.cpu().data.numpy(), axis=0
                    )

            if mode == "submission":
                return pred_list
            else:
                return self.get_full_sort_score(epoch, answer_list, pred_list)
