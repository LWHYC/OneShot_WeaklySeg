import torch
import shutil
import os

class Save_checkpoint(object):
    def __init__(self):
        self.pre_check_save_name = ''
        self.pre_best_save_name = ''
        self.delete_pre = False

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
        if self.delete_pre:
            if os.path.exists(self.pre_check_save_name):
                os.remove(self.pre_check_save_name)
        torch.save(state, filename)
        print('succeffcully save', filename)
        self.pre_check_save_name = filename
        if is_best:
            if self.delete_pre:
                if os.path.exists(self.pre_best_save_name):
                    os.remove(self.pre_best_save_name)
            self.pre_best_save_name = bestname
            shutil.copyfile(filename, bestname)
        self.delete_pre = True
