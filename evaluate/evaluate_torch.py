import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from scipy.sparse import coo_matrix

import neat.genome


class Net(nn.Module):

    def __init__(self, config, genome: neat.genome.DefaultGenome):
        super(Net, self).__init__()

        # Save useful neat parameters
        self.old_connections = genome.connections
        self.old_layer = genome.layer
        self.old_nodes = genome.nodes
        self.num_cnn_layer = config.genome_config.num_cnn_layer
        self.dense_afer_cnn = config.genome_config.dense_after_cnn
        self.num_gnn_layer = config.genome_config.num_gnn_layer
        self.dense_after_gnn = config.genome_config.dense_after_gnn
        self.num_inputs = config.genome_config.num_inputs
        self.num_outputs = config.genome_config.num_outputs
        self.input_size = config.genome_config.input_size

        self.maxpooling_mask = genome.maxpooling_mask
        self.nodes_every_layers = genome.nodes_every_layers

        self.cnn_layers = self._make_cnn_layers
        self.fc_layers_after_cnn = self._make_fc_layers_after_cnn()
        self.gnn_layer = self._make_gnn_layers()
        self.fc_layers_after_gnn = self._make_fc_layers_after_gnn()

        self.set_parameters(genome)

    @property
    def _make_cnn_layers(self):

        layers = []

        for i in range(self.num_cnn_layer):
            if i == 0:
                layers.append(nn.Conv2d(in_channels=self.num_inputs, out_channels=self.nodes_every_layers[0],
                                        kernel_size=3, stride=1, padding=1, bias=True))
            else:
                layers.append(nn.Conv2d(in_channels=self.nodes_every_layers[i-1], out_channels=self.nodes_every_layers[i],
                                        kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=self.nodes_every_layers[i]))
            layers.append(nn.ReLU(inplace=True))
            if self.maxpooling_mask[i]:
                layers.append(nn.MaxPool2d(kernel_size=2))

        return nn.Sequential(*layers)


    def _make_fc_layers_after_cnn(self): # @默认了cnn最后一层形状是1
        layers = []

        for i in range(self.num_cnn_layer, self.num_cnn_layer+self.dense_afer_cnn):
            layers.append(nn.Linear(in_features=self.nodes_every_layers[i-1],
                                    out_features=self.nodes_every_layers[i], bias=True))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_gnn_layers(self):
        layers = []
        for i in range(self.num_cnn_layer + self.dense_afer_cnn, self.num_cnn_layer +
                                                                                  self.dense_afer_cnn + self.num_gnn_layer):
            layers.append(GCNConv(self.nodes_every_layers[i-1], self.nodes_every_layers[i], bias=True))
        self.gcn_layers = layers
        return nn.Sequential(*layers)

    def _make_fc_layers_after_gnn(self):
        layers = []
        for i in range(self.num_cnn_layer+self.dense_afer_cnn+self.num_gnn_layer, len(self.nodes_every_layers)):
            if i == len(self.nodes_every_layers)-1:
                layers.append(nn.Linear(in_features=self.nodes_every_layers[i-1],
                                        out_features=self.num_outputs, bias=True))
            else:
                layers.append(nn.Linear(in_features=self.nodes_every_layers[i-1],
                                        out_features=self.nodes_every_layers[i], bias=True))
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)


    def forward(self, x, gso):
        num_agent = x.shape[1]
        num_batch = x.shape[0]

        # CNN layers + fc layer
        extractFeatureMap = torch.zeros(num_batch, num_agent, self.nodes_every_layers[self.num_cnn_layer+self.dense_afer_cnn]) # (64,10,128)
        if torch.cuda.is_available():
            extractFeatureMap = extractFeatureMap.cuda()
        for id_agent in range(num_agent):
            input_currentAgent = x[:, id_agent]
            cnn_output = self.cnn_layers(input_currentAgent)
            cnn_output_flatten = cnn_output.view(cnn_output.size(0), -1)
            fc_output_currentAgent = self.fc_layers_after_cnn(cnn_output_flatten)
            extractFeatureMap[:, id_agent, :] = fc_output_currentAgent

        # GNN layers
        feature_gcn = torch.zeros(num_batch, num_agent, self.nodes_every_layers[self.num_cnn_layer
                                                                               +self.dense_afer_cnn+self.num_gnn_layer-1]) # (64,10,128)
        if torch.cuda.is_available():
            feature_gcn = feature_gcn.cuda()
        for batch in range(num_batch):
            adj_currentBatch = coo_matrix(gso[batch])
            edge_index_currentBatch = [adj_currentBatch.row, adj_currentBatch.col]
            edge_index_currentBatch = torch.tensor(np.array(edge_index_currentBatch), dtype=torch.int64)
            edge_weight_currentBatch = [adj_currentBatch.data]
            edge_weight_currentBatch = torch.Tensor(np.array(edge_weight_currentBatch)).squeeze()
            feature_currentBatch = extractFeatureMap[batch]
            if torch.cuda.is_available():
                edge_index_currentBatch = edge_index_currentBatch.cuda()
                edge_weight_currentBatch = edge_weight_currentBatch.cuda()
            for index, gcn in enumerate(self.gcn_layers):
                if index == 0:
                    gcn_output = gcn(feature_currentBatch, edge_index_currentBatch, edge_weight_currentBatch)
                    gcn_output = torch.relu(gcn_output)
                else:
                    gcn_output = gcn(gcn_output,edge_index_currentBatch,edge_weight_currentBatch)
                    gcn_output = torch.relu(gcn_output)
            feature_gcn[batch] = gcn_output

        # fc layers
        actions = []
        for id_agent in range(num_agent):
            action_currentAgent = self.fc_layers_after_gnn(feature_gcn[:, id_agent])
            actions.append(action_currentAgent)

        return actions

    def set_parameters(self, genome: neat.genome.DefaultGenome):

        layer = list()  # make sure change layer can affect parameters in cnn
        for module in self.children():
            for block in module:
                if isinstance(block, nn.Conv2d):
                    layer.append(block)
                elif isinstance(block, nn.Linear):
                    layer.append(block)
                elif isinstance(block, GCNConv):
                    layer.append(block)


        nodes = {}

        # add the input node to nodes dict
        order = 0
        for i in range(-self.num_inputs, 0):
            position = [-1, order]  # -1 means input node
            nodes.update({i: position})
            order += 1

        # add every layers to nodes dict
        for i in range(len(self.nodes_every_layers)):
            l = list(genome.layer[i][1])
            l.sort()
            order = 0
            for j in range(len(l)):
                # add node (id, [layer, order in layer]
                position = [i, j]
                nodes.update({l[j]: position})

                # add conv kernel and bias to pytorch module
                if i < self.num_cnn_layer:
                    a = np.array(self.old_nodes[l[j]].kernel)
                    if i == 0:
                        layer[i].weight.data[j] = torch.FloatTensor(a.reshape(3, 3, 3)) # @
                    else:
                        layer[i].weight.data[j] = torch.FloatTensor(a.reshape(self.nodes_every_layers[i-1], 3, 3))
                    b = self.old_nodes[l[j]].bias
                    layer[i].bias.data[j] = torch.FloatTensor([b])
                else:
                    b = self.old_nodes[l[j]].bias
                    layer[i].bias.data[j] = torch.FloatTensor([b])

        for in_node, out_node in genome.connections:

            in_layer = nodes[in_node][0]  # out_node layer number
            out_layer = nodes[out_node][0]   # in_node layer number

            if in_layer == 5:
                pass

            in_num = nodes[in_node][1]
            out_num = nodes[out_node][1]

            if hasattr(layer[out_layer], 'lin'):
                (layer[out_layer].lin.weight.data[out_num])[in_num] = \
                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])
            else:
                (layer[out_layer].weight.data[out_num])[in_num] = \
                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])