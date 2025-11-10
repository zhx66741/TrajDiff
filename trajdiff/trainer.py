import time
import torch
import os
import numpy as np
from TKDE_3_copy_3.utils import tools
from TKDE_3_copy_3.utils.top_k import top_k, reformat_top_k



class Trainer:
    def __init__(self, model, ddbm, optimizer, loss, device, args):
        self.model = model
        self.model = self.model.to(device)
        self.ddbm = ddbm
        self.optimizer = optimizer
        self.loss = loss
        self.loss = self.loss.to(device)
        self.device = device

        self.args = args
        self.pretrain_epoch = args.pretrain_epoch
        self.finetune_epoch = args.finetune_epoch
        self.target_measure = args.target_measure
        self.emd_metric = args.emb_sim_metric
        self.load_checkpoint = args.load_checkpoint

        if args.dataset == "porto" or args.dataset == "geolife":
            if self.target_measure == "sspd":
                self.checkpoint_path = os.getcwd() + '/exp/checkpoint/model_sspd_porto_geolife.pt'
            else:
                self.checkpoint_path = os.getcwd() + '/exp/checkpoint/model_haus_DFD_porto_geolife.pt'  # lstm
        elif args.dataset == "tdrive":
            if self.target_measure == "sspd":
                self.checkpoint_path = os.getcwd() + '/exp/checkpoint/model_sspd_{}.pt'.format(args.dataset)
            else:
                self.checkpoint_path = os.getcwd() + '/exp/checkpoint/model_haus_DFD_{}.pt'.format(args.dataset)
        else:
            ValueError("Unsupported dataset or target measure: {} and {}".format(args.dataset, self.target_measure))


        self.train_identifier = f"{self.target_measure}_{self.emd_metric}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"

        print(
            f"[Trainer Initialization] Target Metric: {self.target_measure.upper()}, Embedding Metric: {self.emd_metric.upper()}")
        print(f"[Trainer Initialization] Identifier: {self.train_identifier}")

    def run(self, train_dataloader, eval_dataloader, test_dataloader):
        if self.load_checkpoint == 1:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        best_score = 0
        for epoch in range(self.finetune_epoch):
            self.epoch = epoch
            self.train(train_dataloader)
            # evaluation 
            eval_score = self.eval(eval_dataloader, stage="Eval")
            if eval_score[1] > best_score:
                best_score = eval_score[1]
                print(f"[Epoch {self.epoch}] Best Eval Score Updated, Test Model on Test Dataset...")
                # test when a better model is found
                test_score = self.eval(test_dataloader, stage="Test")


    def pre_train(self, pretrain_dataloader):
        dataloader = pretrain_dataloader.get_dataloader()
        best_loss = 10000
        best_epoch = 0
        bad_counter = 0
        bad_patience = 5
        for epoch in range(self.pretrain_epoch):
            start_time = time.time()
            self.model.train()
            loss_ep = []
            for batch in dataloader:
                t0, tT = batch
                self.optimizer.zero_grad()
                t0 = tuple(item.to(self.device) for item in t0)
                tT = tuple(item.to(self.device) for item in tT)
                loss_ = self.ddbm.training_bridge_losses(self.model, t0, tT)
                loss_.backward()
                self.optimizer.step()
                loss_ep.append(loss_.item())
            loss_ep_avg = np.mean(loss_ep)
            print("[Training] ep={}, avg_loss={:.7f}, @={:.7f}".format(epoch, loss_ep_avg, time.time() - start_time))

            if best_loss > loss_ep_avg:
                best_epoch = epoch
                best_loss = loss_ep_avg
                bad_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print('model_saved')
            else:
                bad_counter += 1
            if bad_counter == bad_patience or (epoch + 1) == self.pretrain_epoch:
                print("[Training] END! best_epoch={}, best_loss_train={:.7f}".format(best_epoch, best_loss))
                break
        return None

    def train(self, train_dataloader):
        dataloader = train_dataloader.get_dataloader()
        start_time = time.time()
        self.model.train()
        epoch_loss = 0
        for batch in dataloader:
            batch = tuple(item.to(self.device) for item in batch)
            target_trajs_merc, target_trajs_xy, target_trajs_mask, sub_simi =batch

            emb = self.model.forward(target_trajs_merc.float(), target_trajs_xy.float(), None, target_trajs_mask)
            loss = self.loss.loss_compute(emb, sub_simi)

            epoch_loss += loss.item()  # total loss of this epoch
            #################### Gradient descent and backpropagation  ####################
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)  # Prevent gradient explosion.
            self.optimizer.step()

        end_time = time.time()
        print(f"[Epoch {self.epoch}] Epoch Loss:{epoch_loss}, Train Time:{end_time - start_time}")

    def eval(self, eval_dataloader, stage="Eval"):
        dataloader = eval_dataloader.get_dataloader()
        dis_matrix = eval_dataloader.get_dis_matrix()
        # 1. Obtain trajectory representation vector
        self.model.eval()
        emds = torch.zeros(len(dataloader.dataset), self.model.args.hidden_dim).to(self.device)  # batch_size * hidden_dim
        with torch.no_grad():
            for batch in dataloader:
                moved_batch = tuple(item.to(self.device) for item in batch)
                merc, xy, mask, IDs, traj_len = moved_batch
                traj_vecs = self.model.forward(merc.float(), xy.float(), None, mask)
                emds[IDs] = traj_vecs

        # 2. calculate top-k acc
        topk_acc = top_k(emds.cpu(), dis_matrix, metric=self.emd_metric)

        if stage == "Eval":
            print(f"[Epoch {self.epoch}] {stage} {self.target_measure.upper()}@Top-k Acc:{topk_acc}")

        if stage == "Test":
            print(f"[Epoch {self.epoch}] |-> {stage} {self.target_measure.upper()}@Top-k Acc:{topk_acc}")

        return topk_acc
