"""Meters."""
import time
import datetime
import torch
from utils import loss_printer


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        if torch.is_tensor(self.avg):
            self.avg = self.avg.item()
        return self.avg


class TimeMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.end_time = self.start_time
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, n=1):
        self.end_time = time.time()
        self.sum = self.end_time - self.start_time
        self.count += n
        self.avg = self.sum / self.count

    def update_count(self, count):
        self.end_time = time.time()
        self.sum = self.end_time - self.start_time
        self.count += count
        self.avg = self.sum / self.count

    def complete_time(self, remain_batch):
        self.remain_time = datetime.timedelta(seconds=int(self.avg * remain_batch))


class LossesMeter:
    """Computes and stores the average and current value"""

    def __init__(self, fmt='.04f'):
        self._loss = {}
        self.fmt = fmt

    def update(self, dict: dict):
        for key, value in dict.items():
            if key not in self._loss.keys():
                self._loss[key] = AverageMeter()
            else:
                self._loss[key].update(value)

    def print_avg(self):
        return loss_printer({key: meter.avg for key, meter in self._loss.items()}, fmt=self.fmt)

    def print_val(self):
        return loss_printer({key: meter.val for key, meter in self._loss.items()}, fmt=self.fmt)
