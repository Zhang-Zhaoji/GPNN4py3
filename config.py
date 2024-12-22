import os

class Paths(object):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        project_root = 'D:/ai double major/Core/new_GPNN'
        self.project_root = project_root
        self.tmp_root = os.path.join(project_root, 'tmp')

        self.cad_data_root = os.path.join(project_root,'tmp','cad')
        self.hico_data_root = os.path.join(project_root, 'tmp', 'hico')
        self.vcoco_data_root = os.path.join(project_root, 'tmp','vcoco')

def model_args(args,edge_feature_size,node_feature_size,message_size):
    '''
    default model_args function, return a modified args.
    '''
    args.edge_feature_size = edge_feature_size
    args.node_feature_size = node_feature_size
    args.message_size = message_size
    if args.dataset == 'hico':
        args.link_hidden_size = 512
        args.link_hidden_layers = 2
        args.update_hidden_layers = 1
        args.hoi_classes = 117
        args.propagate_layers = 3
        args.link_relu = False
        args.update_bias = True
        args.update_dropout = 0
        args.resize_feature_to_message_size = False
        args.LinkFunction = 'graphconv'
        args.message_def = 'linear_concat_relu'
        args.readout_def = 'fc'
    elif args.dataset == 'cad':
        args.link_hidden_size = 1024
        args.link_hidden_layers = 2
        args. update_hidden_layers = 1
        args.propagate_layers = 3
        args.link_relu = False
        args.update_bias = True
        args.update_dropout = 0
        args.subactivity_classes = 10
        args.affordance_classes = 12
        args.hoi_classes = 10 + 12
        args.resize_feature_to_message_size = False
        args.LinkFunction = 'graphconvlstm'
        args.message_def = 'linear_concat'
        args.readout_def = 'fc_soft_max'
    elif args.dataset == 'vcoco':
        args.link_hidden_size = 256
        args.link_hidden_layers = 3
        args.update_hidden_layers = 1
        args.propagate_layers = 3
        args.link_relu = False
        args.update_bias = True
        args.update_dropout = 0
        args.hoi_classes = 27
        args.resize_feature_to_message_size = False
        args.LinkFunction = 'graphconv'
        args.message_def = 'linear_concat_relu'
        args.readout_def = 'fc'
    return args
        
# hico
# {'model_path': args.resume, 'edge_feature_size': edge_feature_size, 'node_feature_size': node_feature_size, 'message_size': message_size, 'link_hidden_size': 512, 'link_hidden_layers': 2, 'link_relu': False, 'update_hidden_layers': 1, 'update_dropout': False, 'update_bias': True, 'propagate_layers': 3, 'hoi_classes': action_class_num, 'resize_feature_to_message_size': False}

# cad

# {'model_path': args.resume, 'edge_feature_size': edge_feature_size, 'node_feature_size': node_feature_size, 'message_size': edge_feature_size, 'link_hidden_size': 1024, 'link_hidden_layers': 2, 'propagate_layers': 3, 'subactivity_classes': 10, 'affordance_classes': 12}

# vcoco

# {'model_path': args.resume, 'edge_feature_size': edge_feature_size, 'node_feature_size': node_feature_size, 'message_size': message_size, 'link_hidden_size': 256, 'link_hidden_layers': 3, 'link_relu': False, 'update_hidden_layers': 1, 'update_dropout': False, 'update_bias': True, 'propagate_layers': 3, 'hoi_classes': action_class_num, 'resize_feature_to_message_size': False, 'feature_type': 'vcoco_features'}
