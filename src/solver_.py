import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model_ import MMIM

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        print(hp)
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta

        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = model = MMIM(hp)
        
        self.device = torch.device("cuda")
        model = model.cuda()
        print("Model is on device: ", self.device)

        # criterion
        if self.hp.dataset == "ur_funny" or self.hp.n_class > 2:
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        elif self.hp.n_class == 2:
            self.criterion = criterion = nn.BCEWithLogitsLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")
        
        # optimizer
        self.optimizer={}

        if self.is_train:
            mmilb_param = []
            main_param = []
            bert_param = []

            for name, p in model.named_parameters():
                # print(name)
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else: 
                        main_param.append(p)
                
                for p in (mmilb_param+main_param):
                    if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        nn.init.xavier_normal_(p)

        # self.optimizer_mmilb = getattr(torch.optim, self.hp.optim)(
        #     mmilb_param, lr=self.hp.lr_mmilb, weight_decay=hp.weight_decay_club)
        
        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        #self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5, verbose=True)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        log_file = f"/data/wang/junh/results/MMIM/{self.hp.dataset}_{self.hp.modality}_{self.hp.n_class}_{self.hp.lr_main}_{self.hp.d_vh}_{self.hp.d_vout}_best_performance.log"
        model = self.model
        #optimizer_mmilb = self.optimizer_mmilb
        optimizer_main = self.optimizer_main

        #scheduler_mmilb = self.scheduler_mmilb
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        mem_size = 1

        def train(model, optimizer, criterion, stage=1):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            nce_loss = 0.0
            ba_loss = 0.0
            start_time = time.time()

            left_batch = self.update_batch

            mem_pos_tv = []
            mem_neg_tv = []
            mem_pos_ta = []
            mem_neg_ta = []
            if self.hp.add_va:
                mem_pos_va = []
                mem_neg_va = []

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data

                # for mosei we only use 50% dataset in stage 1
                if self.hp.dataset == "mosei":
                    if stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break
                model.zero_grad()

                with torch.cuda.device(0):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()
                    if self.hp.dataset=="ur_funny":
                        y = y.squeeze()
                
                batch_size = y.size(0)

                if stage == 0:
                    y = None
                    mem = None
                elif stage == 1 and i_batch >= mem_size:
                    mem = {'tv':{'pos':mem_pos_tv, 'neg':mem_neg_tv},
                            'ta':{'pos':mem_pos_ta, 'neg':mem_neg_ta},
                            'va': {'pos':mem_pos_va, 'neg':mem_neg_va} if self.hp.add_va else None}
                else:
                    mem = {'tv': None, 'ta': None, 'va': None}

                preds = model(text, visual, audio, vlens, alens, 
                                                bert_sent, bert_sent_type, bert_sent_mask, y, mem)

                if self.hp.n_class == 2:
                    preds = preds.squeeze()
                    #preds = (preds > 0.5).float()
                    y = y.float()
                elif self.hp.n_class > 2:
                    #preds = preds.argmax(dim=-1)
                    y = y.long().squeeze()

                #if stage == 1:
                y_loss = criterion(preds, y)
                loss = y_loss
                loss.backward()

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer.step()

                # Update loss tracking
                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                # Logging
                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, avg_loss))
                    proc_loss, proc_size = 0, 0
                    start_time = time.time()

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0
        
            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                        if self.hp.dataset == 'iemocap':
                            y = y.long()
                    
                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)

                    # we don't need lld and bound anymore
                    preds = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                    if self.hp.n_class == 2:
                        preds = preds.squeeze()
                        #preds = (preds > 0.5).float()
                        y = y.float()
                    elif self.hp.n_class > 2:
                        #preds = preds.argmax(dim=-1)
                        y = y.long().squeeze()

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()
                    if self.hp.n_class == 2:
                        criterion = nn.BCEWithLogitsLoss()
                    elif self.hp.n_class > 2:
                        criterion = nn.CrossEntropyLoss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        best_valid = 1e8
        best_mae = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch
            model.to(torch.device('cuda'))

            # maximize likelihood
            if self.hp.contrast:
                train_loss = train(model, optimizer_mmilb, criterion, 0)

            # minimize all losses left
            train_loss = train(model, optimizer_main, criterion, 1)

            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)
            
            end = time.time()
            duration = end-start
            scheduler_main.step(val_loss)    # Decay learning rate by validation loss

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"] and self.hp.n_class == 1:
                        best_results_dict = eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'mosi' and self.hp.n_class == 1:
                        best_results_dict = eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        best_results_dict = eval_iemocap(results, truths)
                    elif self.hp.dataset in ["mosi", "mosei_senti", "mosei"] and self.hp.n_class > 1:
                        best_results_dict = eval_categorical_labels(results, truths, self.hp.n_class)
                    
                    best_results = results
                    best_truths = truths
                    #print(f"Saved model at pre_trained_models/MM.pt!")
                    #save_model(self.hp, model)
                    # **LOG BEST MODEL IMMEDIATELY**
                    with open(log_file, "a") as f:
                        f.write(f"\nEpoch: {epoch}\n")
                        
                        if best_results_dict:
                            f.write("Best Model Performance:\n")
                            for metric, value in best_results_dict.items():
                                f.write(f"{metric}: {value:.4f}\n")
                        
                        f.write("=" * 50 + "\n")
            elif test_loss < best_mae:
                patience = self.hp.patience
                best_epoch = epoch
                best_mae = test_loss
                if self.hp.dataset in ["mosei_senti", "mosei"] and self.hp.n_class == 1:
                    best_results_dict = eval_mosei_senti(results, truths, True)
                elif self.hp.dataset == 'mosi' and self.hp.n_class == 1:
                    best_results_dict = eval_mosi(results, truths, True)
                elif self.hp.dataset == 'iemocap':
                    best_results_dict = eval_iemocap(results, truths)
                elif self.hp.dataset in ["mosi", "mosei_senti", "mosei"] and self.hp.n_class > 1:
                    best_results_dict = eval_categorical_labels(results, truths, self.hp.n_class)
                
                best_results = results
                best_truths = truths
                #print(f"Saved model at pre_trained_models/MM.pt!")
                #save_model(self.hp, model)
                # **LOG BEST MODEL IMM
                with open(log_file, "a") as f:
                    f.write(f"\nEpoch: {epoch}\n")
                    
                    if best_results_dict:
                        f.write("Best Model Performance:\n")
                        for metric, value in best_results_dict.items():
                            f.write(f"{metric}: {value:.4f}\n")
                    
                    f.write("=" * 50 + "\n")
            else:
                patience -= 1
                if patience == 0:
                    break

        # # Save the best epoch and results to a file
        # with open(log_file, "a") as f:  # "a" for append mode, so it doesn't overwrite previous logs
        #     f.write(f"\nBest epoch: {best_epoch}\n")

        #     if self.hp.dataset in ["mosei_senti", "mosei"]:
        #         best_results_str = eval_mosei_senti(best_results, best_truths, True)
        #         f.write(f"Best MOSI/MOSEI Sentiment Results: {best_results_str}\n")
            
        #     elif self.hp.dataset == "mosi":
        #         self.best_dict = eval_mosi(best_results, best_truths, True)
        #         f.write(f"Best MOSI Results: {self.best_dict}\n")

        #     f.write("=" * 50 + "\n")
        # f.close()

        # print(f'Best epoch: {best_epoch}')
        # if self.hp.dataset in ["mosei_senti", "mosei"]:
        #     eval_mosei_senti(best_results, best_truths, True)
        # elif self.hp.dataset == 'mosi':
        #     self.best_dict = eval_mosi(best_results, best_truths, True)
        # elif self.hp.dataset == 'iemocap':
        #     eval_iemocap(results, truths)       
        # sys.stdout.flush()