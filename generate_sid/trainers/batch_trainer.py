import torch
import time
import os
from modules.codebook_eval import codebook_evaluator
import torch.distributed as dist


class trainer():
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = self.args.device
        self.evaluator = codebook_evaluator()
    def train(self, train_ld, eval_ld,train_sampler):
        if self.args.train_from_pretrained:
            if hasattr(self.args, 'state_dict_save_path'):
                self.model, best_metric, best_epoch, optimizer, scheduler, model_config = self.load_from_state_dict(self.model, self.args)
                best_loss = sum([val for key, val in best_metric.items() if 'loss' in key])#sum(best_metric.values())# sum([task_metric['loss'] * self.args.task_config[task_name]['loss_weight'] for task_name, task_metric in best_metric.items()])
                print('checkpoint load successfully, start training with checkpoint {}, with loss {}...'.format(os.path.join(self.args.state_dict_save_path, f"checkpoint_best.pt"), best_loss))
            else:
                raise ValueError('parameter pretrained_model_path is not defined in args')
        else:
            best_loss = 1e8
            best_epoch = -1
        eval_metric = {}
        for epoch in range(best_epoch + 1, self.args.num_epochs + 1):
            train_sampler.set_epoch(epoch)
            for batch_data in train_ld:
                self.model.train()
                losses = self.model(batch_data)
                loss = sum(losses.values())#  + codebook_loss + self.loss_weight * commitment_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print(
                "training epoch {} : loss {:.4} lr {:.5} ".format(
                    epoch, loss.item(), self.optimizer.param_groups[0]['lr']))
            if eval_ld is not None and epoch % self.args.eval_steps == 0:
                eval_metric = self.eval(eval_ld)
                eval_loss = sum([val for key, val in eval_metric.items() if 'loss' in key])
                # eval_loss = sum(eval_metric.values())
                print("evaluating epoch {} total loss {}, all metrics {}".format(epoch, eval_loss, eval_metric))
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_epoch = epoch
                    best_metric = eval_metric
                    if self.args.rank == 0:
                        self.save_to_state_dict(self.model, self.optimizer, self.scheduler, best_epoch, best_metric, self.args, save_path=os.path.join(self.args.state_dict_save_path, f"checkpoint_best.pt"))
                print("best valid result {:.5} at epoch {}...".format(best_loss, best_epoch))

            if best_epoch >= 0 and epoch - best_epoch >= self.args.early_stop:
                break
            if epoch % self.args.save_per_epochs == 0:
                self.save_to_state_dict(self.model, self.optimizer, self.scheduler, epoch, eval_metric, self.args, save_path=os.path.join(self.args.state_dict_save_path, f"checkpoint_{epoch}.pt"))

        if self.args.device != torch.device("cpu"):  
            torch.cuda.synchronize(self.args.device)

       
        if eval_ld is None: 
            self.save_to_state_dict(self.model, self.optimizer, self.scheduler, epoch, {}, self.args, save_path=os.path.join(self.args.state_dict_save_path, f"checkpoint_last.pt"))

    def eval(self, dataloader):
        self.model.eval()
        metrics = {}
        ind = 0
        with torch.no_grad():
            for batch_data in dataloader:
                losses = self.model(batch_data)
                try:
                    quantized_encodings, quantized_indices = self.model.get_codes(batch_data)
                except:
                    quantized_encodings, quantized_indices = self.model.module.get_codes(batch_data)
                for key in losses:
                    if key not in metrics:
                        metrics[key] = [losses[key]]
                    else:
                        metrics[key].append(losses[key])
                codebook_size = quantized_encodings.shape[1]
                for key in ['perplexity', 'collision']:
                    result = self.evaluator.compute_perplexity(quantized_indices, codebook_size=codebook_size) if key == 'perplexity' else self.evaluator.compute_collision(quantized_indices)
                    if key not in metrics:
                        metrics[key] = [result]
                    else:
                        metrics[key].append(result)
                if ind % 100 == 0:
                    print("evaluating batch ind {} ".format(ind))
                ind += 1
            for key in metrics:
                metrics[key] = torch.mean(torch.as_tensor(metrics[key], dtype=torch.float32)).to(self.device)
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)  # Wait for all processes to finish computation
        eval_metrics = {}
        for key in metrics:
            eval_metrics[key] = {}
            eval_metrics[key] = reduce_value(metrics[key], self.args.world_size).item()
        return eval_metrics

    def save_to_state_dict(self, model, optimizer, scheduler, best_epoch, best_metric, args, save_path):
        try:
            model_state_dict = model.module.state_dict()
        except:
            model_state_dict = model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "config": args.model_config,
        }
        torch.save(checkpoint, save_path)
        print("checkpoint has been save to {} successfully...".format(save_path))

    def load_from_state_dict(self, model, args):
        checkpoint_path = os.path.join(args.state_dict_save_path, "checkpoint_best.pt")
        if not os.path.exists(checkpoint_path):
            raise ValueError('checkpoint {} not found'.format(checkpoint_path))
        else:
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
        try:
            try:
                model.load_state_dict(checkpoint["model"], strict=True)
            except:
                model.module.load_state_dict(checkpoint["model"], strict=True)
        except:
            try:
                model.load_state_dict(checkpoint["model"], strict=False)
            except:
                model.module.load_state_dict(checkpoint["model"], strict=False)
            print('model loaded success with not strict mode...')
        best_metric = checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]
        model_config = checkpoint["config"]
        return model, best_metric, best_epoch, optimizer, scheduler, model_config

def reduce_value(value, world_size, average=True):
    if world_size < 2:  
        return value

    with torch.no_grad():
        dist.all_reduce(value) 
        if average:  
            value /= world_size
    return value
