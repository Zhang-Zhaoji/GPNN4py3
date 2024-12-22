import os
import torch
import torch.nn as nn

class LinkFunction(torch.nn.Module):
    def __init__(self, args):
        super(LinkFunction, self).__init__()
        self.args = args
        self.l_definition = args.LinkFunction

        if self.l_definition == 'graphconv':
            self.l_function = self.l_graph_conv
            self.init_graph_conv()
        elif self.l_definition == 'graphconvlstm':
            self.l_function = self.l_graph_conv_lstm
            self.init_graph_conv_lstm()
        else:
            raise NotImplementedError(f"self.l_definition = {self.l_definition} not in graphconv or graphconvlstm")

    def forward(self, edge_features):
        return self.l_function(edge_features)

    # Initialize layers for GraphConv
    def init_graph_conv(self):
        input_size = self.args.edge_feature_size
        hidden_size = self.args.link_hidden_size
        self.layers = nn.ModuleList()

        if self.args.link_relu:
            self.layers.extend([nn.ReLU(), nn.Dropout()])
        for _ in range(self.args.link_hidden_layers-1):
            self.layers.append(nn.Conv2d(input_size, hidden_size, 1))
            self.layers.append(nn.ReLU())
            input_size = hidden_size

        self.layers.append(nn.Conv2d(input_size, 1, 1))

    # GraphConv
    def l_graph_conv(self, edge_features):
        for layer in self.layers:
            edge_features = layer(edge_features)
        return edge_features[:, 0, :, :]

    # Initialize layers for GraphConvLSTM
    def init_graph_conv_lstm(self):
        input_size = self.args.edge_feature_size
        hidden_size = self.args.link_hidden_size
        hidden_layers = self.args.link_hidden_layers

        self.ConvLSTM = ConvLSTM(input_size, hidden_size, hidden_layers)
        self.layers = nn.ModuleList([
            self.ConvLSTM,
            nn.Conv2d(hidden_size, 1, 1),
            nn.Sigmoid()
        ])

    # GraphConvLSTM
    def l_graph_conv_lstm(self, edge_features):
        for layer in self.layers:
            edge_features = layer(edge_features)
        return edge_features[:, 0, :, :]

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # 计算填充大小以保持输入和输出尺寸一致
        self.padding = int((self.kernel_size - 1) / 2)
        self.bias = bias

        # 定义卷积层，输入是输入维度加上隐藏维度，输出是4倍的隐藏维度（对应i, f, o, g）
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        if cur_state == None:
            batch_size = input_tensor.shape[0]
            spatial_size = list(input_tensor.shape[2:])
            cur_state = self.init_hidden([batch_size,self.hidden_dim]+spatial_size)
        h_cur, c_cur = cur_state

        # 沿着通道轴进行拼接
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        # 将输出分割成四个部分，分别对应输入门、遗忘门、输出门和候选单元状态
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新单元状态
        c_next = f * c_cur + i * g
        # 更新隐藏状态
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, shape):
        return (torch.zeros(shape, device=self.conv.weight.device),
                torch.zeros(shape, device=self.conv.weight.device))
    

class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, hidden_layer_num, kernel_size=1):
        super(ConvLSTM, self).__init__()
        if hidden_layer_num < 1:
            raise RuntimeError(f"hidden layer num = {hidden_layer_num} < 1")

        self.hidden_layer_num = hidden_layer_num
        self.learn_modeles = torch.nn.ModuleList()
        self.prev_states = list()
        self.learn_modeles.append(ConvLSTMCell(input_channels, hidden_channels, kernel_size))
        for _ in range(hidden_layer_num-1):
            self.learn_modeles.append(ConvLSTMCell(hidden_channels, hidden_channels, kernel_size))
        self._reset_hidden_states()

    def forward(self, input_, reset=False):
        if reset:
            self._reset_hidden_states()
        else:
            for prev_state in self.prev_states:
                if prev_state:
                    prev_state[0].detach_()
                    prev_state[1].detach_()

        next_layer_input = input_
        for i, layer in enumerate(self.learn_modeles):
            prev_state = layer(next_layer_input, self.prev_states[i])
            next_layer_input = prev_state[0]
            self.prev_states[i] = prev_state

        return next_layer_input
    def _reset_hidden_states(self):
        self.prev_states = [None for _ in range(self.hidden_layer_num)]
    
class MessageFunction(torch.nn.Module):
    def __init__(self, args):
        super(MessageFunction, self).__init__()
        self.m_definition = args.message_def
        self.args = args
        self.edge_feature_size = args.edge_feature_size
        self.node_feature_size = args.node_feature_size
        self.message_size = args.message_size

        if self.m_definition == 'linear':
            self.m_function = self.m_linear
            self.linear_edge = nn.Linear(self.edge_feature_size, self.message_size, bias=True)
            self.linear_node = nn.Linear(self.node_feature_size, self.message_size, bias=True)
        elif self.m_definition == 'linear_edge':
            self.m_function = self.m_linear_edge
            self.linear_edge = nn.Linear(self.edge_feature_size, self.message_size, bias=True)
        elif self.m_definition == 'linear_concat':
            self.message_size = int(args.message_size / 2)
            self.m_function = self.m_linear_concat
            self.linear_edge = nn.Linear(self.edge_feature_size, self.message_size, bias=True)
            self.linear_node = nn.Linear(self.node_feature_size, self.message_size, bias=True)
        elif self.m_definition == 'linear_concat_relu':
            self.message_size = int(args.message_size / 2)
            self.m_function = self.m_linear_concat_relu
            self.linear_edge = nn.Linear(self.edge_feature_size, self.message_size, bias=True)
            self.linear_node = nn.Linear(self.node_feature_size, self.message_size, bias=True)
        else:
            raise NotImplementedError(f"Message function '{self.m_definition}' not implemented")

    def forward(self, h_v, h_w, e_vw):
        return self.m_function(h_v, h_w, e_vw)

    def m_linear(self, h_v, h_w, e_vw):
        message = torch.zeros(e_vw.size()[0], self.message_size, e_vw.size()[2], device=e_vw.device)
        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = self.linear_edge(e_vw[:, :, i_node]) + self.linear_node(h_w[:, :, i_node])
        return message

    def m_linear_edge(self, h_v, h_w, e_vw):
        message = torch.zeros(e_vw.size()[0], self.message_size, e_vw.size()[2], device=e_vw.device)
        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = self.linear_edge(e_vw[:, :, i_node])
        return message

    def m_linear_concat(self, h_v, h_w, e_vw):
        message = torch.zeros(e_vw.size()[0], self.args.message_size, e_vw.size()[2], device=e_vw.device)
        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = torch.cat([self.linear_edge(e_vw[:, :, i_node]), self.linear_node(h_w[:, :, i_node])], 1)
        return message

    def m_linear_concat_relu(self, h_v, h_w, e_vw):
        message = torch.zeros(e_vw.size()[0], self.args.message_size, e_vw.size()[2], device=e_vw.device)
        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = torch.cat([self.linear_edge(e_vw[:, :, i_node]), self.linear_node(h_w[:, :, i_node])], 1)
        return torch.relu(message)
    

class UpdateFunction(torch.nn.Module):
    def __init__(self, args):
        super(UpdateFunction, self).__init__()
        self.u_definition = args.update_def
        self.args = args
        self.node_feature_size = args.node_feature_size
        self.message_size = args.message_size

        if self.u_definition == 'gru':
            self.u_function = self.u_gru
            self.init_gru()
        else:
            raise NotImplementedError(f"Update function '{self.u_definition}' not implemented")

    def forward(self, h_v, m_v):
        return self.u_function(h_v, m_v)

    # GRU: node state as hidden state, message as input
    def u_gru(self, h_v, m_v):
        # Since GRU expects a sequence, we need to add a sequence dimension if it's not present
        #if len(m_v.shape) == len(h_v.shape):
        #    m_v = m_v.unsqueeze(0)  # Add sequence dimension
        output, h = self.gru(m_v, h_v)
        # If a sequence dimension was added, remove it
        if len(m_v.shape) == len(h_v.shape):
            output = output.squeeze(0)
            h = h.squeeze(0)
        return h

    def init_gru(self):
        num_layers = self.args.update_hidden_layers
        bias = True
        dropout = self.args.update_dropout # Assuming dropout is defined as 0 or a float between 0 and 1
        self.gru = nn.GRU(self.message_size, self.node_feature_size, num_layers=num_layers, bias=bias, dropout=dropout)

class ReadoutFunction(torch.nn.Module):
    def __init__(self, args):
        super(ReadoutFunction, self).__init__()
        self.r_definition = args.readout_def
        self.args = args

        if self.r_definition == 'fc':
            self.r_function = self.r_fc
            self.init_fc()
        elif self.r_definition == 'fc_soft_max':
            self.r_function = self.r_fc_soft_max
            self.init_fc_soft_max()
        elif self.r_definition == 'fc_sig':
            self.r_function = self.r_fc_sigmoid
            self.init_fc_sigmoid()
        else:
            raise NotImplementedError(f"Readout function '{self.r_definition}' not implemented")

    def forward(self, h_v):
        return self.r_function(h_v)

    # Fully connected layers with softmax output
    def r_fc_soft_max(self, hidden_state):
        return self.fc_softmax(hidden_state)

    def init_fc_soft_max(self):
        input_size = self.args.readout_input_size
        output_classes = self.args.output_classes
        self.fc_softmax = nn.Sequential(
            nn.Linear(input_size, output_classes),
            nn.Softmax(dim=1)
        )

    # Fully connected layers with sigmoid output
    def r_fc_sigmoid(self, hidden_state):
        return self.fc_sigmoid(hidden_state)

    def init_fc_sigmoid(self):
        input_size = self.args.readout_input_size
        output_classes = self.args.output_classes
        self.fc_sigmoid = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, output_classes),
            nn.Sigmoid()
        )

    # Fully connected layers
    def r_fc(self, hidden_state):
        return self.fc(hidden_state)

    def init_fc(self):
        input_size = self.args.readout_input_size
        output_classes = self.args.output_classes
        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_classes)
        )

class General_GPNN(torch.nn.Module):
    def __init__(self, args):
        super(General_GPNN, self).__init__()
        self.args = args
        if self.args.resize_feature_to_message_size:
            # Resize large features
            self.edge_feature_resize = nn.init.xavier_normal(nn.Linear(self.args.edge_feature_size, args.message_size))
            self.node_feature_resize = nn.init.xavier_normal(nn.Linear(self.args.node_feature_size, args.message_size))
            self.args.edge_feature_size = args.message_size
            self.args.node_feature_size = args.message_size
            args.edge_feature_size = args.message_size
            args.node_feature_size = args.message_size
        
        if args.dataset == 'cad':
            self.subactivity_classes = args.subactivity_classes
            self.affordance_classes = args.affordance_classes

        self.link_fun = LinkFunction(args)
        self.sigmoid = nn.Sigmoid()
        self.message_fun = MessageFunction(args)
        self.update_fun = UpdateFunction(args)
        args.readout_input_size = args.node_feature_size
        args.output_classes = args.hoi_classes
        args.readout_def = 'fc'
        self.readout_fun = ReadoutFunction(args)

        self.propagate_layers = args.propagate_layers

        self._load_link_fun(args)

    def forward(self, edge_features, node_features, adj_mat, node_labels, human_nums, obj_nums, args):
        if self.args.resize_feature_to_message_size:
            edge_features = self.edge_feature_resize(edge_features)
            node_features = self.node_feature_resize(node_features)
        edge_features = edge_features.permute(0, 3, 1, 2)
        node_features = node_features.permute(0, 2, 1)
        hidden_node_states = [[node_features[batch_i, ...].unsqueeze(0).clone() for _ in range(self.propagate_layers+1)] for batch_i in range(node_features.size()[0])]
        hidden_edge_states = [[edge_features[batch_i, ...].unsqueeze(0).clone() for _ in range(self.propagate_layers+1)] for batch_i in range(node_features.size()[0])]

        pred_adj_mat = torch.zeros(adj_mat.size())
        pred_node_labels = torch.zeros(node_labels.size())
        if args.cuda:
            pred_node_labels = pred_node_labels.cuda()
            pred_adj_mat = pred_adj_mat.cuda()

        # Belief propagation

        for batch_idx in range(node_features.size()[0]):
            valid_node_num = human_nums[batch_idx] + obj_nums[batch_idx]

            for passing_round in range(self.propagate_layers):
                pred_adj_mat[batch_idx, :valid_node_num, :valid_node_num] = self.link_fun(hidden_edge_states[batch_idx][passing_round][:, :, :valid_node_num, :valid_node_num])
                sigmoid_pred_adj_mat = self.sigmoid(pred_adj_mat[batch_idx, :, :]).unsqueeze(0)

                # Loop through nodes
                for i_node in range(valid_node_num):
                    h_v = hidden_node_states[batch_idx][passing_round][:, :, i_node]
                    h_w = hidden_node_states[batch_idx][passing_round][:, :, :valid_node_num]
                    e_vw = edge_features[batch_idx, :, i_node, :valid_node_num].unsqueeze(0)
                    m_v = self.message_fun(h_v, h_w, e_vw)

                    # Sum up messages from different nodes according to weights
                    m_v = sigmoid_pred_adj_mat[:, i_node, :valid_node_num].unsqueeze(1).expand_as(m_v) * m_v
                    hidden_edge_states[batch_idx][passing_round+1][:, :, :valid_node_num, i_node] = m_v
                    m_v = torch.sum(m_v, 2)
                    h_v = self.update_fun(h_v[None].contiguous(), m_v[None])

                    # Readout at the final round of message passing
                    if passing_round == self.propagate_layers-1:
                        if args.dataset == 'cad':
                            if i_node == 0:
                                pred_node_labels[:, i_node, :self.subactivity_classes] = self.readout_fun(h_v.squeeze(0))[:,:self.subactivity_classes] # before self.subactivity_classes
                            else:
                                pred_node_labels[:, i_node, :self.affordance_classes] = self.readout_fun(h_v.squeeze(0))[:,self.subactivity_classes:] # after self.subactivity_classes
                        else:
                            pred_node_labels[batch_idx, i_node, :] = self.readout_fun(h_v.squeeze(0))

        return pred_adj_mat, pred_node_labels

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args.model_path):
            os.makedirs(model_args.model_path)
        best_model_file = os.path.join(model_args.model_path, os.pardir, 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])





def main():
    pass


if __name__ == '__main__':
    main()
