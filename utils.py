import torch
import os
import numpy as np
import tensorboard_logger

def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    epoch = state['epoch']
    checkpoint_file = os.path.join(directory, f'checkpoint_{epoch}.pth')
    best_model_file = os.path.join(directory, f'model_best_{epoch}.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        torch.save(state, best_model_file)


def load_best_checkpoint(args, model, optimizer):
    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file,encoding='latin1')
            # checkpoint_str = {key.decode('latin1'): value for key, value in checkpoint.items()}
            args.start_epoch = checkpoint['epoch']
            best_epoch_error = checkpoint['best_epoch_error']
            try:
                avg_epoch_error = checkpoint['avg_epoch_error']
            except KeyError:
                avg_epoch_error = np.inf
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.cuda:
                model.cuda()
            print(f"=> loaded best model '{best_model_file}' (epoch {checkpoint['epoch']})")
            return args, best_epoch_error, avg_epoch_error, model, optimizer
        else:
            print(f"=> no best model found at '{best_model_file}'")
    return None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

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


class Logger(object):
    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            # if the directory does not exist we create the directory
            tmp_log_dir = log_dir.replace("/","\\")
            tmp_log_dir = tmp_log_dir.replace(":",'-')
            os.makedirs(tmp_log_dir)
        else:
            # clean previous logged data under the same directory name
            tmp_log_dir = log_dir.replace("/","\\")
            tmp_log_dir = tmp_log_dir.replace(":",'-')
            self._remove(log_dir)
            
        # configure the project
        tensorboard_logger.configure(tmp_log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        tensorboard_logger.log_value(name, value, self.global_step)
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains


def main():
    pass


if __name__ == '__main__':
    main()
