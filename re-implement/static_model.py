import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GPNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dv = args.node_feature_size  # vertex dimension
        self.de = args.edge_feature_size    # edge dimension
        self.iteration_number = args.propagate_layers # propagate_layers in original code
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_classes = self.args.hoi_classes

        # Link function to compute edge weights
        if self.args.LinkFunction == 'graphconv':

            self.link_Function = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.dv + self.de, 
                      out_channels=1, 
                      kernel_size=(1, 1), 
                      stride=(1, 1)),
            nn.Sigmoid()
        )
        elif self.args.LinkFunction == 'graphconvlstm':
            raise("Should import dynamic model package")
        
        # Message function to compute messages
        if self.args.message_def == 'linear_concat':
            self.message_Function = nn.Linear(2 * self.dv + self.de, self.de)
        elif self.args.message_def == 'linear_concat_relu':
            self.message_Function = nn.Sequential(
            nn.Linear(2 * self.dv + self.de, self.de),
            nn.ReLU()
        )
        else:
            raise NotImplementedError(f"Message function '{self.args.m_definition}' not implemented")
        
        # Update function using GRU
        self.update_function = nn.GRU(input_size = self.de, hidden_size= self.dv)

        # Readout function to map the final hidden state to a specified feature dimension
        if self.args.readout_def == 'fc':
            self.readout_Function = nn.Linear(self.dv, self.output_classes)
        elif self.args.readout_def == 'fc_soft_max':
            self.readout_Function = nn.Sequential(
            nn.Linear(self.dv, self.output_classes),
            nn.Softmax()
        )
        else:
            raise NotImplementedError(f"ReadOut function '{self.args.r_definition}' not implemented")


    def forward(self, edge_features, node_features, adj_mat, node_labels, human_nums, obj_nums, args):
        
        edge_features = edge_features.permute(0, 3, 1, 2) # batch, feature channel, node, node, 
        node_features = node_features.permute(0, 2, 1) # batch, feature channel , node

        pred_adj_mat = torch.zeros_like(adj_mat)
        pred_node_labels = torch.zeros_like(node_labels)

        device = self.device
        self.link_Function.to(device)
        self.message_Function.to(device)
        self.update_function.to(device)
        self.readout_Function.to(device)
        pred_node_labels = pred_node_labels.to(device)
        pred_adj_mat = pred_adj_mat.to(device)

        # Feature = [[torch.zeros(1, self.dv * 2 + self.de ,valid_node_num,valid_node_num, device=device)] * (self.iteration_number+1)] * (node_features.size()[0])

        # for every batch:
        for batch_idx in range(node_features.size()[0]):
            valid_node_num = human_nums[batch_idx] + obj_nums[batch_idx] # V
            # Feature = [torch.zeros(1, self.dv * 2 + self.de ,valid_node_num,valid_node_num, device=device)] * (self.iteration_number+1)
            Feature = torch.zeros(1, self.dv * 2 + self.de ,valid_node_num,valid_node_num, device=device)#] * (self.iteration_number+1)
            current_pred_adj_mats = [torch.zeros(1,valid_node_num,valid_node_num,device=device)] * (self.iteration_number+1)
            
            edge_feature = edge_features[batch_idx, :self.de, :valid_node_num,:valid_node_num].unsqueeze(0).to(device) # 1, feature channel, node, node
            node_feature = node_features[batch_idx, :self.dv, :valid_node_num].unsqueeze(0).to(device) # 1, feature channel, node

            hidden_node_state = [node_feature.clone()] * (self.iteration_number+1)  # 1, feature channel, node
            hidden_edge_state = [edge_feature.clone()] * (self.iteration_number+1) # 1, feature channel, node, node

              # 1, node, node, dv * 2 + de
            

            for passing_round in range(self.iteration_number):
                with torch.no_grad():
                    Feature = Feature.clone()
                    for i_node in range(valid_node_num):
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
                for i_node in range(valid_node_num):
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
                        node_label = self.readout_Function(hidden_node_state[passing_round][:,:,i_node].clone()).squeeze(0)
                        pred_node_labels[batch_idx, i_node,: ]  = node_label.clone()
                        pred_adj_mat[batch_idx, :valid_node_num, :valid_node_num] = current_pred_adj_mats[passing_round].squeeze(0)
                        
        return pred_adj_mat, pred_node_labels
    
    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args.model_path):
            os.makedirs(model_args.model_path)
        best_model_file = os.path.join(model_args.model_path, os.pardir, 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])
    
        