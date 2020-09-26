from fastai.torch_core import *
from fastai.callback import *
from fastai.layers import *
from fastai.basic_train import LearnerCallback
import torch.nn.functional as F
import torch
from scipy import stats

def root_mean_squared_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Root mean squared error between `pred` and `targ`."
    pred,targ = pred.contiguous().view(-1),targ.contiguous().view(-1)
    return torch.sqrt(F.mse_loss(pred, targ))

def pearson_r(pred, targ):
    pred,targ = pred.contiguous().view(-1),targ.contiguous().view(-1)
    return stats.pearsonr(pred,targ)[0]

class Regression(Callback):
    "Stores predictions and targets to perform calculations on epoch end."
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])
        self.rmse = 0.0
        self.pr = 0.0
        self.batch_iter = 0
        self.batch_num = 0.0
        self.metrics_size = 10

    def on_batch_end(self, last_output, last_target, **kwargs):
        # assert last_output.numel() == last_target.numel(), "Expected same numbers of elements in pred & targ"
        self.batch_iter += 1         
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target[1].cpu()))
        #print('batch_end:', self.batch_iter, self.batch_num, self.rmse, self.pr)

        if self.batch_iter == self.metrics_size: 
            self.rmse += root_mean_squared_error(self.preds, self.targs)
            # self.pr += pearson_r(self.preds, self.targs)
            #print('cal rmse:', self.batch_iter, self.batch_num, self.rmse, self.pr)
            self.targs, self.preds = Tensor([]), Tensor([])
            self.batch_iter = 0
            self.batch_num += 1
            #print('iter end:', self.batch_iter, self.batch_num, self.rmse, self.pr)

class RMSE_Reg(Regression):
    def on_epoch_end(self, last_metrics, **kwargs):
        #print('RMSE epoch end:', self.batch_iter, self.batch_num, self.rmse, self.pr)
        if self.batch_iter != 0:
            self.rmse += root_mean_squared_error(self.preds, self.targs)
            #print('RMSE cal end batch:', self.batch_iter, self.batch_num, self.rmse, self.pr)
            self.targs, self.preds = Tensor([]), Tensor([])
            self.batch_iter = 0
            self.batch_num += 1
        return add_metrics(last_metrics, 1-self.rmse/self.batch_num)

class Pr_Reg(Regression):
    def on_epoch_end(self, last_metrics, **kwargs):
        print('Pr epoch end:', self.batch_iter, self.batch_num, self.rmse, self.pr)
        if self.batch_iter != 0:
            self.pr += pearson_r(self.preds, self.targs)
            print('Pr cal end batch:', self.batch_iter, self.batch_num, self.rmse, self.pr)
            self.targs, self.preds = Tensor([]), Tensor([])
            self.batch_iter = 0
            self.batch_num += 1
        return add_metrics(last_metrics, self.pr/self.batch_num)
    
def mean_squared_error_mine(pred, *targs)->Rank0Tensor:
    "Mean squared error between `pred` and `targ`."
    targ = targs[1]
    pred, targ = pred.contiguous().view(-1),targ.contiguous().view(-1)
    return F.mse_loss(pred, targ)