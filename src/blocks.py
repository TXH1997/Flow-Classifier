import math

import torch
from torch import nn
from torch.nn import init

from src import utils


class CNN(nn.Module):
    def __init__(self, emb_size, kernel_num, kernel_sizes):
        super(CNN, self).__init__()

        # configurations
        self.emb_size = emb_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        # modules
        self.conv1 = nn.Conv1d(self.emb_size, self.kernel_num, self.kernel_sizes[0])
        self.conv2 = nn.Conv1d(self.emb_size, self.kernel_num, self.kernel_sizes[1])
        self.conv3 = nn.Conv1d(self.emb_size, self.kernel_num, self.kernel_sizes[2])
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.bm1 = nn.BatchNorm1d(self.kernel_num)
        self.bm2 = nn.BatchNorm1d(self.kernel_num)
        self.bm3 = nn.BatchNorm1d(self.kernel_num)

    def forward(self, inputs):  # [batch_size, seq_len, emb_size]
        inputs = inputs.permute(0, 2, 1)  # [batch_size, emb_size, seq_len]
        pooled1 = torch.max(self.relu1(self.bm1(self.conv1(inputs)).permute(0, 2, 1)), 1)[0]  # [batch_size, kernel_num]
        pooled2 = torch.max(self.relu2(self.bm2(self.conv2(inputs)).permute(0, 2, 1)), 1)[0]
        pooled3 = torch.max(self.relu3(self.bm3(self.conv3(inputs)).permute(0, 2, 1)), 1)[0]
        outputs = torch.cat([pooled1, pooled2, pooled3], 1)  # [batch_size, 3 * kernel_num]

        return outputs


class FFNN(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size, output_size, dropout, out_weights_initializer=None):
        super(FFNN, self).__init__()
        # configurations
        assert num_hidden_layers >= 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.out_weights_initializer = out_weights_initializer

        # modules
        if self.num_hidden_layers == 0:
            self.fc_direct = nn.Linear(self.input_size, self.output_size)
        else:
            self.fc_in = nn.Linear(self.input_size, self.hidden_size)
            self.relu = nn.ReLU()
            for i in range(self.num_hidden_layers - 1):
                setattr(self, 'fc_hi' + str(i), nn.Linear(self.hidden_size, self.hidden_size))
                setattr(self, 'drop' + str(i), nn.Dropout(p=self.dropout, inplace=True))
                setattr(self, 'relu' + str(i), nn.ReLU())
            self.fc_out = nn.Linear(self.hidden_size, self.output_size)
            self.drop = nn.Dropout(p=self.dropout, inplace=True)
        self.init_params()

    def init_params(self):
        if self.num_hidden_layers == 0:
            if self.out_weights_initializer == 'orthogonal':
                with torch.no_grad():
                    w = torch.from_numpy(utils.block_orthonormal_initializer(self.input_size, self.output_size, 3))
                    self.fc_direct.weight.copy_(w.t())
            else:
                init.kaiming_normal_(self.fc_direct.weight)
            std = self.calculate_std(name='fc_direct')
            init.normal_(self.fc_direct.bias, mean=0, std=std)
        else:
            init.kaiming_normal_(self.fc_in.weight)
            std = self.calculate_std(name='fc_in')
            init.normal_(self.fc_in.bias, mean=0, std=std)

            init.kaiming_normal_(self.fc_out.weight)
            std = self.calculate_std(name='fc_out')
            init.normal_(self.fc_out.bias, mean=0, std=std)

            for i in range(self.num_hidden_layers - 1):
                init.kaiming_normal_(getattr(self, 'fc_hi' + str(i)).weight)
                std = self.calculate_std(name='fc_hi' + str(i))
                init.normal_(getattr(self, 'fc_hi' + str(i)).bias, mean=0, std=std)

    def calculate_std(self, name=None):
        assert name is not None
        fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self, name).weight)
        gain = math.sqrt(2.0)
        std = gain / math.sqrt(fan_in)

        return std

    def forward(self, inputs):
        dim = len(inputs.size())
        if dim == 3:
            batch_size = inputs.size(0)
            seq_len = inputs.size(1)
            inputs = inputs.reshape(batch_size * seq_len, -1)

        if self.num_hidden_layers == 0:
            outputs = self.fc_direct(inputs)
        else:
            outputs = self.relu(self.fc_in(inputs))
            self.drop(outputs)
            for i in range(self.num_hidden_layers - 1):
                outputs = getattr(self, 'relu' + str(i))(getattr(self, 'fc_hi' + str(i))(outputs))
                getattr(self, 'drop' + str(i))(outputs)
            outputs = self.fc_out(outputs)

        if dim == 3:
            outputs = outputs.reshape(batch_size, seq_len, -1)

        return outputs


class HighwayBiLSTM(nn.Module):
    def __init__(self, lstm_input_size, hidden_size, lstm_layers, lstm_dropout, device):
        super(HighwayBiLSTM, self).__init__()
        self.device = device

        # parameters
        self.lstm_input_size = lstm_input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        assert self.lstm_layers > 0

        # modules
        self.projection = FFNN(0, self.lstm_input_size, 1, 2 * self.hidden_size, dropout=0)
        for i in range(self.lstm_layers):
            if i == 0:
                setattr(self, 'bilstm' + str(i), CustomBiLSTM(self.lstm_input_size, self.hidden_size, self.lstm_dropout))
            else:
                setattr(self, 'bilstm' + str(i), CustomBiLSTM(self.hidden_size * 2, self.hidden_size, self.lstm_dropout))
            setattr(self, 'projection' + str(i), FFNN(0, self.hidden_size * 2, 1, self.hidden_size * 2, dropout=0))
            setattr(self, 'sigmoid' + str(i), nn.Sigmoid())
            setattr(self, 'drop' + str(i), nn.Dropout(p=self.lstm_dropout))

    def forward(self, inputs):
        for i in range(self.lstm_layers):
            outputs, _ = getattr(self, 'bilstm' + str(i))(inputs)
            outputs = getattr(self, 'drop' + str(i))(outputs)
            highway_gates = getattr(self, 'sigmoid' + str(i))(getattr(self, 'projection' + str(i))(outputs))
            if i == 0:
                inputs = self.projection(inputs)
            outputs = highway_gates * outputs + (1 - highway_gates) * inputs
            inputs = outputs

        return outputs


class CustomBiLSTM(nn.Module):
    def __init__(self, lstm_input_size, hidden_size, lstm_dropout):
        super(CustomBiLSTM, self).__init__()

        # hyperparameters
        self.lstm_input_size = lstm_input_size
        self.hidden_size = hidden_size
        self.lstm_dropout = lstm_dropout
        self.p_input_size = self.lstm_input_size + self.hidden_size

        # parameters
        self.init_h_fw = nn.Parameter(init.xavier_normal_(torch.zeros(1, self.hidden_size)))
        self.init_c_fw = nn.Parameter(init.xavier_normal_(torch.zeros(1, self.hidden_size)))
        self.init_h_bw = nn.Parameter(init.xavier_normal_(torch.zeros(1, self.hidden_size)))
        self.init_c_bw = nn.Parameter(init.xavier_normal_(torch.zeros(1, self.hidden_size)))

        # modules
        self.fw_drop = nn.Dropout(p=self.lstm_dropout)
        self.bw_drop = nn.Dropout(p=self.lstm_dropout)
        self.fw_projection = FFNN(0, self.p_input_size, 1, 3 * self.hidden_size, dropout=0, out_weights_initializer='orthogonal')
        self.bw_projection = FFNN(0, self.p_input_size, 1, 3 * self.hidden_size, dropout=0, out_weights_initializer='orthogonal')
        self.fw_sigmoid_1 = nn.Sigmoid()
        self.fw_sigmoid_2 = nn.Sigmoid()
        self.bw_sigmoid_1 = nn.Sigmoid()
        self.bw_sigmoid_2 = nn.Sigmoid()
        self.fw_tanh_1 = nn.Tanh()
        self.fw_tanh_2 = nn.Tanh()
        self.bw_tanh_1 = nn.Tanh()
        self.bw_tanh_2 = nn.Tanh()

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)
        h_fw = self.init_h_fw.repeat(batch_size, 1)
        c_fw = self.init_c_fw.repeat(batch_size, 1)
        h_bw = self.init_h_bw.repeat(batch_size, 1)
        c_bw = self.init_c_bw.repeat(batch_size, 1)
        out_fw = []
        out_bw = []
        for i in range(seq_len):
            h_fw, c_fw = self.custom_fw_cell(inputs[i], h_fw, c_fw)
            h_bw, c_bw = self.custom_bw_cell(inputs[seq_len - i - 1], h_bw, c_bw)
            out_fw.append(h_fw.unsqueeze(1))
            out_bw.append(h_bw.unsqueeze(1))
        out_bw.reverse()
        out_fw = torch.cat(out_fw, 1)
        out_bw = torch.cat(out_bw, 1)
        out = torch.cat([out_fw, out_bw], 2)

        return out, None

    def custom_fw_cell(self, inputs, h, c):
        h = self.fw_drop(h)
        concat = self.fw_projection(torch.cat([inputs, h], 1))
        i, j, o = torch.split(concat, split_size_or_sections=self.hidden_size, dim=1)
        i = self.fw_sigmoid_1(i)
        new_c = (1 - i) * c + i * self.fw_tanh_1(j)
        new_h = self.fw_tanh_2(new_c) * self.fw_sigmoid_2(o)

        return new_h, new_c

    def custom_bw_cell(self, inputs, h, c):
        h = self.bw_drop(h)
        concat = self.bw_projection(torch.cat([inputs, h], 1))
        i, j, o = torch.split(concat, split_size_or_sections=self.hidden_size, dim=1)
        i = self.bw_sigmoid_1(i)
        new_c = (1 - i) * c + i * self.bw_tanh_1(j)
        new_h = self.bw_tanh_2(new_c) * self.bw_sigmoid_2(o)

        return new_h, new_c


if __name__ == '__main__':
    pass
