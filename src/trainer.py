import time

import torch
import torch.nn as nn

import src.model as md
import src.utils as utils
from src.data_generator import DataGenerator


class Trainer:
    def __init__(self, device, config):
        # # 设置随机数种子
        # self.seed = int(time.time() * 256) % (2 ** 32)
        # utils.set_seed(self.seed)

        self.device = device
        self.name = config['name']
        self.conf = config['conf']
        self.data_dir = self.conf['data_dir']
        self.log_path = self.conf['log_dir'] + self.name + '.log'
        self.model_path = self.conf['model_dir'] + self.name + '.ptm'
        self.batch_size = self.conf['batch_size']
        self.learning_rate = self.conf['learning_rate']
        self.decay_rate = self.conf['decay_rate']
        self.decay_freq = self.conf['decay_freq.']
        self.max_hanging_epoch = self.conf['max_hanging_epoch']
        self.extend_dim = self.conf['extend_dim']

        self.resnet_shrink = self.conf['resnet_shrink']
        self.resnet_depth = self.conf['resnet_depth']

        print('ResNet Shrink: {}'.format(self.resnet_shrink))
        print('ResNet Depth: {}'.format(self.resnet_depth))

        print('Batch size: {}'.format(self.batch_size))
        print('Initial lr: {}'.format(self.learning_rate))
        print('Decay frequency: {}'.format(self.decay_freq))

        self.data_generator = DataGenerator(self.data_dir, self.batch_size, ext=self.extend_dim)
        self.model = md.ResNetClassifier(self.data_generator.channels, self.resnet_shrink, self.resnet_depth).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_freq, gamma=self.decay_rate)

        open(self.log_path, 'w', encoding='utf-8').close()

    def write_log(self, epoch, loss, dev_acc, test_acc, eps):
        with open(self.log_path, 'a', encoding='utf-8') as log:
            log_line = '{}-{}'.format(self.data_generator.test_pos, epoch)
            log_line += ' ' + 'loss={:.4f}'.format(loss)
            log_line += ' ' + 'dev_acc={:.4f}'.format(dev_acc)
            log_line += ' ' + 'test_acc={:.4f}'.format(test_acc)
            log_line += ' ' + 'eps={:.2f}'.format(eps)
            log.write(log_line + '\n')

    def train(self):
        max_acc = 0.0
        acc_list = []
        accu_loss = 0.0
        steps = 0
        epochs = 0
        hanging = 0
        start_time = time.time()
        while not self.data_generator.round_end:
            for examples, labels in self.data_generator.generate_train_data():
                examples = torch.from_numpy(examples).to(self.device)
                labels = torch.from_numpy(labels).to(self.device)
                output = self.model(examples)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                accu_loss += loss.detach().item()
                steps += 1
                if steps % 10 == 0:
                    print('\r{}'.format(steps), end='')
            epochs += 1
            time_cost = time.time() - start_time
            steps_per_sec = steps / time_cost
            average_loss = accu_loss / steps
            validate_acc = self.validate()
            current_acc = self.evaluate()
            print('\r[{}] loss={:.2f} dev={:.2f}% test={:.2f}% batches/s={:.2f}'.format(epochs, average_loss, validate_acc * 100, current_acc * 100, steps_per_sec))
            print('Current lr is: {}'.format(self.optimizer.param_groups[0]['lr']))
            self.write_log(epochs, average_loss, validate_acc, current_acc, steps_per_sec * self.batch_size)
            if current_acc > max_acc:
                max_acc = current_acc
                hanging = 0
            else:
                hanging += 1
            if hanging == self.max_hanging_epoch:
                self.data_generator.switch_train_test()
                self.reset()
                acc_list.append(max_acc)
                max_acc = 0.0
                epochs = 0
                hanging = 0
                if not self.data_generator.round_end:
                    print('\nCurrent record: ' + str([float('{:.4f}'.format(i)) for i in acc_list]) + '\n')
                    print('------------Fold-{}------------\n'.format(self.data_generator.test_pos))
                else:
                    print('\n------Result after {} fold------'.format(self.data_generator.n_fold))
            accu_loss = 0.0
            steps = 0
            start_time = time.time()
        print('Final record: ' + str([float('{:.4f}'.format(i)) for i in acc_list]))
        print('Average accuracy={:.2f}%'.format(sum(acc_list) * 100 / len(acc_list)))

    def evaluate(self):
        self.model.eval()
        match_num = 0
        for examples, labels in self.data_generator.generate_test_data():
            examples = torch.from_numpy(examples).to(self.device)
            output = torch.max(self.model(examples), 1)[1].cpu().detach().numpy()
            match_num += utils.match(labels, output)
        accuracy = match_num / (self.data_generator.test_num())
        self.model.train()
        return accuracy

    def validate(self):
        self.model.eval()
        match_num = 0
        for examples, labels in self.data_generator.generate_dev_data():
            examples = torch.from_numpy(examples).to(self.device)
            output = torch.max(self.model(examples), 1)[1].cpu().detach().numpy()
            match_num += utils.match(labels, output)
        accuracy = match_num / (self.data_generator.dev_num())
        self.model.train()
        return accuracy

    def reset(self):
        del self.model
        del self.criterion
        del self.optimizer
        del self.scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = md.ResNetClassifier(self.data_generator.channels, self.resnet_shrink, self.resnet_depth).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_freq, gamma=self.decay_rate)
