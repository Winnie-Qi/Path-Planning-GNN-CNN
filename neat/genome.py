"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function

from itertools import count
from random import choice, random, shuffle, randint

import sys

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle
from neat.six_util import iteritems, iterkeys


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('conn_add_num', int),
                        ConfigParameter('conn_delete_num', int),
                        ConfigParameter('node_add_num', int),
                        ConfigParameter('node_delete_num', int),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected'),
                        ConfigParameter('num_cnn_layer', int),
                        ConfigParameter('dense_after_cnn', int),
                        ConfigParameter('num_gnn_layer', int),
                        ConfigParameter('dense_after_gnn', int),
                        ConfigParameter('kernel_size', int),
                        ConfigParameter('input_size', int),
                        ConfigParameter('full_connect_input', bool)]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1','yes','true','on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0','no','false','off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write('initial_connection      = {0} {1}\n'.format(self.initial_connection,
                                                                 self.connection_fraction))
        else:
            f.write('initial_connection      = {0}\n'.format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if not 'initial_connection' in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key, config):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}
        self.layer = []
        self.maxpooling_mask = []

        # Create layer: cnn layer + dense layer + gnn layer + dense layer
        for i in range(config.num_cnn_layer):
            self.layer.append(['cnn', set()])
        for i in range(len(self.layer), len(self.layer) + config.dense_after_cnn):
            self.layer.append(['fc', set()])
        for i in range(len(self.layer), len(self.layer) + config.num_gnn_layer):
            self.layer.append(['gnn', set()])
        for i in range(len(self.layer), len(self.layer) + config.dense_after_gnn):
            self.layer.append(['fc', set()])

        # Compute node number in every layer.

        # cnn part -In the law of [32, 32, 64, 64...], depending on the number of layers
        self.nodes_every_layers = [2 ** (5 + i // 2) for i in range(config.num_cnn_layer * 2)]
        if len(self.nodes_every_layers) > config.num_cnn_layer:
            self.nodes_every_layers = self.nodes_every_layers[:config.num_cnn_layer]
        # Compute output size of cnn
        convW = [config.input_size]
        convH = [config.input_size]
        nFilterTaps = [int(pow(config.kernel_size,0.5))] * config.num_cnn_layer
        nPaddingSzie = [1] * config.num_cnn_layer
        for i in range(config.num_cnn_layer):
            W_tmp = int((convW[i] - nFilterTaps[i] + 2 * nPaddingSzie[i])) + 1
            H_tmp = int((convH[i] - nFilterTaps[i] + 2 * nPaddingSzie[i])) + 1
            if i % 2 == 0:
                W_tmp = int((W_tmp - 2) / 2) + 1
                H_tmp = int((H_tmp - 2) / 2) + 1
            convW.append(W_tmp)
            convH.append(H_tmp)
        self.size_output_cnn = convW[-1] * convH[-1]

        # dense part after cnn -In the law of [128, 128, 64, ...], depending on the number of layers
        if config.dense_after_cnn > 0:
            for i in range(0, config.dense_after_gnn):
                if not i % 2:
                    self.nodes_every_layers.append(self.nodes_every_layers[-1])
                else:
                    self.nodes_every_layers.append(self.nodes_every_layers[-1] // 2)

        # gnn part
        self.nodes_every_layers = self.nodes_every_layers + [self.nodes_every_layers[-1]]*config.num_gnn_layer

        # dense part after gnn
        if config.dense_after_gnn == 1:
            self.nodes_every_layers.append(config.num_outputs)
        else:
            for i in range(0, config.dense_after_gnn):
                if not i % 2:
                    self.nodes_every_layers.append(self.nodes_every_layers[-1])
                else:
                    self.nodes_every_layers.append(self.nodes_every_layers[-1] // 2)

        # Generate maxpooling mask.
        self.maxpooling_mask = [0] * (config.num_cnn_layer + config.dense_after_cnn +
                                        config.num_gnn_layer + config.dense_after_gnn)
        for i in range(config.num_cnn_layer):
            if i%2 == 0:
                self.maxpooling_mask[i] = 1

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key, len(self.layer)-1, 'fc', '_')
        # Add output layer nodes
        self.layer[-1][1] = set(config.output_keys)

        # Create node genes for the cnn nodes
        config.node_indexer = None # reset node_indexer
        for i in range(config.num_cnn_layer):
            for j in range(self.nodes_every_layers[i]):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                if i == 0:
                    node = self.create_node(config, node_key, i, 'cnn', [3,self.nodes_every_layers[i]])
                else:
                    node = self.create_node(config, node_key, i, 'cnn', self.nodes_every_layers[i-1:i+1])
                self.nodes[node_key] = node
                self.layer[i][1].add(node_key)

        # Create node genes for the fc nodes after cnn layer and for the gnn nodes
        for i in range(config.num_cnn_layer, config.num_cnn_layer + config.dense_after_cnn + config.num_gnn_layer):
            for j in range(self.nodes_every_layers[i]):
                node_key = config.get_new_node_key(self.nodes)
                node = self.create_node(config, node_key, i, 'fc', '_')
                self.nodes[node_key] = node
                self.layer[i][1].add(node_key)

        # Create connection genes for fc nodes
        if config.initial_connection == 'full':
            self.connect_full(config)
        elif config.initial_connection == 'partial':
            self.connect_partial(config)
        else:
            print("Only full and partial connection allowed in CNN!")

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """

        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness < genome2.fitness: # make sure genome1.fitness > genome2.fitness
            genome1, genome2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(genome1.connections):
            cg2 = genome2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        for key, ng1 in iteritems(genome1.nodes):
            ng2 = genome2.nodes.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

        # Add layer according to nodes in new genome
        for node in iteritems(self.nodes):
            self.layer[node[1].layer][1].add(node[1].key)

        # Compute node num in every layer
        self.nodes_every_layers = [0] * len(self.layer)
        for i in range(len(self.layer)):
            self.nodes_every_layers[i] = len(self.layer[i][1])

    def mutate(self, config): 
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1,(config.node_add_prob + config.node_delete_prob +
                         config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob/div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob)/div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob)/div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob)/div):
                self.mutate_delete_connection(config)
        else:
            # if random() < config.node_add_prob:
            #     self.mutate_add_node(config)
            #
            # if random() < config.node_delete_prob:
            #     self.mutate_delete_node(config)
            #
            # if random() < config.conn_add_prob:
            #     self.mutate_add_connection(config)
            #
            # if random() < config.conn_delete_prob:
            #     self.mutate_delete_connection(config)

            pass

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    # Added by Andrew @20181107
    # Add a node to the network, if the added node in the first layer then judge if should add a full connection
    # to all the input. Then add one connection to the former layer and one to the after layer.
    # Note: The node cannot be added to the last layer!
    # TODO: Add connection according to the conncetion type parameter in the config file
    def mutate_add_node(self, config):
        num = 0
        for i in range(config.node_add_num):
            num += 1
            # Choose the layer to add node (not the last layer)
            layer_num = randint(0, len(self.nodes_every_layers)-2)
            node_type = self.layer[layer_num][0]

            # Revise the nodes_every_layers list
            self.nodes_every_layers[layer_num] += 1

            new_node_id = config.get_new_node_key(self.nodes)
            if node_type == 'cnn':
                ng = self.create_node(config, new_node_id, layer_num, 'cnn', self.nodes_every_layers[layer_num-1])
            else:
                ng = self.create_node(config, new_node_id, layer_num, 'fc', '_')

            self.layer[layer_num][1].add(new_node_id)
            self.nodes[new_node_id] = ng

            # if the added node in first layer
            connections = []
            '''
            if layer_num == 0: #TODO: Add connections to the following layers
                # Add full connection between input and the first layer
                if config.full_connect_input:
                    for input_id in config.input_keys:
                        connections.append((input_id, new_node_id))

                # Add one connction between input and the first layer
                else:
                    input_id = choice(config.input_keys)
                    connections.append((input_id, new_node_id))
            '''


            if layer_num == 0: # Add connection to input, if the added node in first layer
                for input_id in config.input_keys:
                    connections.append((input_id, new_node_id))
            elif layer_num == len(self.nodes_every_layers)-2: # Add connection to output, if the added node in last layer
                for output_id in config.output_keys:
                    connections.append((new_node_id, output_id))
            elif layer_num > config.num_cnn_layer-1:
                for j in list(self.layer[layer_num - 1][1]):
                    connections.append((j, new_node_id))
                for j in list(self.layer[layer_num + 1][1]):
                    connections.append((new_node_id, j))

            # 如果增加的是卷积层，要改变下一层卷积核的层数
            # 如果增加的是卷积层中的最后一层，则不用
            #connection = self.create_connection(config, node_id, new_node_id)
            #self.connections[connection.key] = connection

            # Add to support dense connection. by Andrew 2019.3.18
            # Add connections to the layer before
            # if layer_num <= config.num_cnn_layer: # if the layer of the added node is in cnn layer or in the first fc layer
            #     for i in (range(config.num_dense_layer)):
            #         if layer_num-i-1 >= 0:
            #             for j in list(self.layer[layer_num-i-1][1]):
            #                 connections.append((j, new_node_id))
            # else:
            #     for j in list(self.layer[layer_num - 1][1]):
            #         connections.append((j, new_node_id))
            #
            # # Add connections to the layer after
            # if layer_num < config.num_cnn_layer: # if the layer of the added node is in cnn layer
            #     for i in (range(config.num_dense_layer)):
            #         if layer_num+i+1 <= config.num_cnn_layer: # connect to following cnn layer or the first fc layer
            #             for j in list(self.layer[layer_num+i+1][1]):
            #                 connections.append((new_node_id, j))
            # else: # the added node is in fc layers
            #     for j in list(self.layer[layer_num + 1][1]):
            #         connections.append((new_node_id, j))


            if config.initial_connection == 'full':
                for node1, node2 in connections:
                    connection = self.create_connection(config, node1, node2)
                    self.connections[connection.key] = connection
            elif config.initial_connection == 'partial':
                assert 0 <= config.connection_fraction <= 1
                shuffle(connections)
                num_to_add = int(round(len(connections) * config.connection_fraction))
                for input_id, output_id in connections[:num_to_add]:
                    connection = self.create_connection(config, input_id, output_id)
                    self.connections[connection.key] = connection
            else:
                print("Only full and partial connection allowed in CNN!")
            '''
            out_node_layer_distance = randint(left, right)
            out_node = choice(list(self.layer[layer_num + out_node_layer_distance][1]))
            connection = self.create_connection(config, new_node_id, out_node)
            self.connections[connection.key] = connection
            '''

        print("{0} nodes added!".format(num))

    """
    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id, -198043)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)
    """
    # Not used
    def add_connection(self, config, input_key, output_key, weight, enabled):

        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)

        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    """
    def mutate_add_connection(self, config):
        """"""
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """"""
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    """
    # Added by Andrew @20181107
    # Add a connection to the network.
    # TODO: Add connection with the probability according to its connections already has.
    # TODO: Add connection according to the conncetion type parameter in the config file
    def mutate_add_connection(self, config):
        num = 0
        for i in range(config.conn_add_num):

            # Choose the outnode layer
            layer_num = randint(0, config.num_layer - 1)

            # If choose out_node form the first layer, the input_node should choose from input of the network.
            if layer_num == 0:
                out_node = choice(list(self.layer[layer_num][1]))
                in_node = choice(config.input_keys)
            else:
                out_node = choice(list(self.layer[layer_num][1]))
                #in_node = choice(list(self.layer[layer_num - 1][1]))
                # Changed to support dense connection. by Andrew 2019.3.18
                left = 1
                right = layer_num if layer_num < config.num_dense_layer else config.num_dense_layer
                in_node_layer_distance = randint(left, right)
                in_node = choice(list(self.layer[layer_num - in_node_layer_distance][1]))

            # Don't duplicate connections.
            key = (in_node, out_node)
            if key in self.connections:
                # TODO: Should this be using mutation to/from rates? Hairy to configure...
                if config.check_structural_mutation_surer():
                    self.connections[key].enabled = True
                continue

            # Don't allow connections between two output nodes
            if in_node in config.output_keys and out_node in config.output_keys:
                continue

            # No need to check for connections between input nodes:
            # they cannot be the output end of a connection (see above).

            # For feed-forward networks, avoid creating cycles.
            if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), key):
                continue

            cg = self.create_connection(config, in_node, out_node)
            self.connections[cg.key] = cg
            num += 1
        print("{0} connections added!".format(num))

    def mutate_delete_node(self, config):
        num = 0
        for i in range(config.node_delete_num):
            # Do nothing if there are no non-output nodes.
            available_nodes = [k for k in iterkeys(self.nodes) if k not in config.output_keys]
            if not available_nodes:
                continue

            del_key = choice(available_nodes)

            # Cannot delete node in the first fc layer
            #if self.nodes[del_key].layer == config.num_cnn_layer:
            #    return -1

            # Cannot delete node in the last (output) layer
            if self.nodes[del_key].layer == config.num_layer:
                continue

            # If there is only one node
            if len(self.layer[self.nodes[del_key].layer][1]) <= 1:
                continue

            connections_to_delete = set()
            for k, v in iteritems(self.connections):
                if del_key in v.key:
                    connections_to_delete.add(v.key)

            for key in connections_to_delete:
                del self.connections[key]

            self.layer[self.nodes[del_key].layer][1].remove(del_key)

            # Revise the nodes_every_layers list
            self.nodes_every_layers[self.nodes[del_key].layer] -= 1

            del self.nodes[del_key]

            num += 1
        print("{0} nodes deleted!".format(num))

    def mutate_delete_connection(self, config):
        num = 0
        for i in range(config.conn_delete_num):
            if self.connections:
                key = choice(list(self.connections.keys()))
                #TODO: add judgement to avoid del the last connection between two layers
                del self.connections[key]
                num += 1
        print("{0} connections deleted!".format(num))

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)

        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)

        s += "\nLayers:"
        for i in range(len(self.layer)):
            s += "\n\t" + self.layer[i][0] + ": "
            l = list(self.layer[i][1])
            l.sort()
            for node in l:
                s += " {0}".format(node)
        return s

    @staticmethod
    def create_node(config, node_id, layer, node_type, num_nodes):
        node = config.node_gene_type(node_id, layer)
        node.init_attributes(config, node_type, num_nodes)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id,num_nodes):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config,0,num_nodes)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in iterkeys(self.nodes) if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.output_keys]
        output = [i for i in iterkeys(self.nodes) if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections

    def compute_full_connections_with_layer(self, config, i):
        """
        Compute connections for a fully-connected cnn genome--each node in one
        layer connected to all nodes in the next layer
        """
        connections = []

        if self.layer[i-1][0] == 'cnn': # previous layer is cnn
            if self.size_output_cnn == 1:
                for node_i in self.layer[i-1][1]:
                    for node_j in self.layer[i][1]:
                        connections.append((node_i, node_j))
            else:
                for node_i in self.layer[i-1][1]:
                    for node_j in self.layer[i][1]:
                        for n in range(self.size_output_cnn):
                            connections.append((node_i, n, node_j))
        else:
            for node_i in self.layer[i-1][1]:
                for node_j in self.layer[i][1]:
                    connections.append((node_i, node_j))

        '''
        # Original none dense connention
        for i in range(len(self.layer) - 1):
             for node1 in self.layer[i][1]:
                    for node2 in self.layer[i+1][1]:
                        connections.append((node1, node2))
        '''

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections

    def connect_full(self, config):
        """
        Create a fully-connected cnn genome
        """
        fc_layer = [i for i in range(config.num_cnn_layer,len(self.layer))]
        for i in fc_layer:
            for node1, node2 in self.compute_full_connections_with_layer(config, i):
                connection = self.create_connection(config, node1, node2, self.nodes_every_layers[i-1:i+1])
                self.connections[connection.key] = connection

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections_with_layer(config)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
