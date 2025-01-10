import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GPNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.link_hidden_size = self.args.link_hidden_size
        self.dv = args.node_feature_size  # vertex dimension
        self.de = args.edge_feature_size    # edge dimension
        self.iteration_number = args.propagate_layers # propagate_layers in original code
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_classes = self.args.hoi_classes
        self.subactivity_classes = self.args.subactivity_classes
        self.affordance_classes = self.args.affordance_classes
        self.link_hidden_layers = self.args.link_hidden_layers


        # Link function to compute edge weights
        if self.args.LinkFunction == 'graphconv':
            self.link_Function = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.dv + self.de, 
                      out_channels=1, 
                      kernel_size=(1, 1), 
                      stride=(1, 1)),
            nn.Sigmoid()
        )
            raise("here is dynamic model.")
        elif self.args.LinkFunction == 'graphconvlstm':
            self.convlstm = ConvLSTM(input_channels= 2 * self.dv + self.de,hidden_channels=self.link_hidden_size,hidden_layer_num=self.link_hidden_layers)
            self.link_Function = nn.Sequential(
                ConvLSTMWrapper(self.convlstm),
                nn.Sigmoid()
            )
        else:
            raise(f"{self.args.LinkFunction} is not defined.")
        
        # Message function to compute messages
        if self.args.message_def == 'linear_concat':
            self.message_Function = nn.Linear(2 * self.dv + self.de, self.link_hidden_size)
        elif self.args.message_def == 'linear_concat_relu':
            self.message_Function = nn.Sequential(
            nn.Linear(2 * self.dv + self.de, self.link_hidden_size),
            nn.ReLU()
        )
        else:
            raise NotImplementedError(f"Message function '{self.args.m_definition}' not implemented")
        
        # Update function using GRU
        self.update_function = nn.GRU(input_size = self.link_hidden_size, hidden_size= self.dv)

        # Readout function to map the final hidden state to a specified feature dimension
        if self.args.readout_def == 'fc':
            self.readout_Function = nn.Linear(self.dv, self.output_classes)
        elif self.args.readout_def == 'fc_soft_max':
            self.readout_Function_0 = nn.Sequential(
            nn.Linear(self.dv, self.subactivity_classes),
            nn.Softmax()
        )
            self.readout_Function_1 = nn.Sequential(
            nn.Linear(self.dv, self.affordance_classes),
            nn.Softmax()
        )
        else:
            raise NotImplementedError(f"ReadOut function '{self.args.r_definition}' not implemented")

    def convLSTM(self):
        pass

    def forward(self, edge_features, node_features, adj_mat, node_labels, human_nums, obj_nums, args):
        
        edge_features = edge_features.permute(0, 3, 1, 2) # batch, feature channel, node, node, 
        node_features = node_features.permute(0, 2, 1) # batch, feature channel , node
        batch_size = edge_features.size()[0]
        V = edge_features.size()[2]
        pred_adj_mat = torch.zeros_like(adj_mat)
        pred_node_labels = torch.zeros_like(node_labels)

        device = self.device
        self.link_Function.to(device)
        self.message_Function.to(device)
        self.update_function.to(device)
        self.readout_Function_0.to(device)
        self.readout_Function_1.to(device)
        pred_node_labels = pred_node_labels.to(device)
        pred_adj_mat = pred_adj_mat.to(device)

        Feature = torch.zeros(batch_size, self.dv * 2 + self.de,V,V, device=device)#] * (self.iteration_number+1)
        current_pred_adj_mats = [torch.zeros(1,V,V,device=device)] * (self.iteration_number+1)
            
        edge_feature = edge_features.to(device) # B, feature channel, node, node
        node_feature = node_features.to(device) # B, feature channel, node

        hidden_node_state = [node_feature.clone() for _index in range(self.iteration_number+1)]  # 1, feature channel, node
        hidden_edge_state = [edge_feature.clone() for _index in range(self.iteration_number+1)] # 1, feature channel, node, node

        for passing_round in range(self.iteration_number):
                with torch.no_grad():
                    Feature = Feature.clone()
                    for i_node in range(V):
                        Feature[:,:self.dv,i_node,:] = hidden_node_state[passing_round]
                        Feature[:, self.dv:2 * self.dv,:,i_node] = hidden_node_state[passing_round]
                    Feature[:,2 * self.dv:,:,:] = hidden_edge_state[passing_round] # full edge feature, 1, 2 * dv + de, D, D
                #Feature[passing_round][:,:self.dv,i_node,:] = hidden_node_state[passing_round]
                    #Feature[passing_round][:, self.dv:2 * self.dv,:,i_node] = hidden_node_state[passing_round]
                #Feature[passing_round][:,2 * self.dv:,:,:] = hidden_edge_state[passing_round] # full edge feature, 1, 2 * dv + de, D, D
                # 每次更新feature矩阵。

                new_pred_adj_mat = self.link_Function(Feature).squeeze(1) # update pred_adj_mat, 1*D*D output
                current_pred_adj_mats[passing_round] = new_pred_adj_mat
                # Feature[passing_round] = Feature[passing_round].permute(0,2,3,1)
            
                # Loop through nodes
                for i_node in range(V):
                    h_v = hidden_node_state[passing_round][:,:, i_node] # i node, 1 * dv
                    h_w = hidden_node_state[passing_round] # all node, 1 * dv * V 
                    e_vw = hidden_edge_state[passing_round][:, :, i_node, :] # edge feature, 1 * de * V 
                    h_v_expanded = h_v.unsqueeze(2).expand_as(h_w) # all node, 1 * dv * V 
                    combined_features = torch.cat([h_v_expanded, h_w, e_vw], dim=1).to(device).permute(0,2,1) # full edge feature, 1, V,  2 * dv + de
                    m_v = self.message_Function(combined_features).permute(0,2,1) # full edge feature, 1, V, de
                    hidden_edge_state[passing_round+1][:,:, i_node, :] = m_v # full edge feature, 1, de, D

                    m_v_out = current_pred_adj_mats[passing_round][:, i_node, :].unsqueeze(1).expand_as(m_v) * m_v # full edge feature, 1, de, V
                    m_v_sum = torch.sum(m_v_out, 2) # sum on different vertexs, 1, de, 1

                    _output, h_v_out = self.update_function(m_v_sum.clone(),h_v.clone()) # get new hidden_node_state of 1 * dv
                    h_v_out = h_v_out.squeeze(0)
                    hidden_node_state[passing_round+1][:,:,i_node] = h_v_out.clone()

                    if passing_round == self.iteration_number-1:
                        if i_node == 0:
                            node_label = self.readout_Function0(hidden_node_state[passing_round][:,:,i_node].clone()).squeeze(0)
                            pred_node_labels[:, i_node,:self.subactivity_classes]  = node_label.clone()
                        else:
                            node_label = self.readout_Function1(hidden_node_state[passing_round][:,:,i_node].clone()).squeeze(0)
                            pred_node_labels[:, i_node,: ]  = node_label.clone()
                        pred_adj_mat[:, :, :] = current_pred_adj_mats[passing_round].squeeze(0)
                        
        return pred_adj_mat, pred_node_labels
    
    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args.model_path):
            os.makedirs(model_args.model_path)
        best_model_file = os.path.join(model_args.model_path, os.pardir, 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=self.padding)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.size(0)
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = self._reset_prev_states(state_size)

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), dim=1)
        gates = self.conv(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, dim=1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = remember_gate * prev_cell + in_gate * cell_gate
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

    def _reset_prev_states(self, state_size):
        return (
            torch.zeros(state_size),
            torch.zeros(state_size)
        )


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, hidden_layer_num, kernel_size=1):
        super(ConvLSTM, self).__init__()
        if hidden_layer_num < 1:
            raise ValueError("Hidden layer number must be at least 1.")

        self.hidden_layer_num = hidden_layer_num
        self.cells = nn.Sequential(
            *[ConvLSTMCell(hidden_channels if i > 0 else input_channels, hidden_channels, kernel_size)
              for i in range(hidden_layer_num)]
        )

    def forward(self, input_, prev_states=None):
        if prev_states is None:
            prev_states = [None] * self.hidden_layer_num

        next_layer_input = input_
        next_states = []
        for i, cell in enumerate(self.cells):
            next_layer_input, next_state = cell(next_layer_input, prev_states[i])
            next_states.append(next_state)

        return next_layer_input, next_states
    def _reset_hidden_states(self):
        self.prev_states = [None for _ in range(self.hidden_layer_num)]

class ConvLSTMWrapper(nn.Module):
    def __init__(self, convlstm):
        super(ConvLSTMWrapper, self).__init__()
        self.convlstm = convlstm
        self.convlstm._reset_hidden_states()
        self.hidden_states = None

    def forward(self, input_):
        output, self.hidden_states = self.convlstm(input_, self.hidden_states)
        return output
        