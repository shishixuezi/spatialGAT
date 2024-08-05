import logging
import pandas as pd
import torch
from torch_geometric.data import Data
from utils import mesh_pair_distance


class ODGraph:
    def __init__(self):
        self.digits = {'city': 5, 'mesh': 9, 'pref': 2, 'region': 3}
        self.name = None
        # For node IDs and mapped indices
        self.nodes = {}
        self.xs = {}
        self.edges = {}
        self.graph = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_graph_info(self):
        """
        Log information about the generated graph.
        """
        if self.graph is None:
            self.logger.error("Graph has not been generated yet. Call generate_graph() first.")
            return
        self.logger.info('============== Complete Graph Info ===============')
        self.logger.info(f'#Nodes: {self.graph.num_nodes}')
        self.logger.info(f'#Edges: {self.graph.num_edges}')
        self.logger.info(f'#Node features: {self.graph.num_node_features}')
        self.logger.info(f'Has self loops: {self.graph.has_self_loops()}')

    def add_node_list(self, nodes):
        unique_id = nodes.unique()
        assert len(set(pd.Series(unique_id).str.len())) == 1
        inv_map = {v: k for k, v in self.digits.items()}
        name = inv_map[len(unique_id[0])]
        self.nodes[name] = pd.DataFrame(data={'rawId': unique_id,
                                              'mappedId': pd.RangeIndex(len(unique_id))})
        self.nodes[name] = dict(zip(self.nodes[name]['rawId'], self.nodes[name]['mappedId']))

    def mapping_nodes_to_raw_id(self, mapped_id, name='mesh'):
        a = self.nodes[name]
        inv_map = {v: k for k, v in a.items()}
        return mapped_id.map(inv_map)


class HomogeneousODGraph(ODGraph):
    def __init__(self, xs, od):
        super().__init__()
        self.xs['mesh'] = xs['mesh']
        od['distance'] = 1 / (mesh_pair_distance(od['origin_key'], od['dest_key']) + 0.001)
        self.edges['od'] = od
        self.graph = Data()
        self.generate_graph()
        super().log_graph_info()

    def generate_graph(self):
        self.add_node_list(self.xs['mesh']['KEY_CODE'])
        self.xs['mesh'].drop(['KEY_CODE'], axis=1, inplace=True)
        self.graph.x = torch.tensor(self.xs['mesh'].values, dtype=torch.float)
        self.graph.edge_index = torch.stack([
            torch.tensor(self.edges['od']['origin_key'].map(self.nodes['mesh']).values),
            torch.tensor(self.edges['od']['dest_key'].map(self.nodes['mesh']).values)], dim=0)
        self.graph.edge_weight = torch.tensor(self.edges['od']['distance'].values, dtype=torch.float)
        self.graph.edge_label = torch.tensor(self.edges['od']['num'].values, dtype=torch.float)

    def convert_edge_index_with_edge_value_to_df(self, edge_index, edge_value=None):
        assert edge_index.shape[0] == 2
        df = pd.concat([
            self.mapping_nodes_to_raw_id(pd.Series(edge_index[0].cpu().numpy())),
            self.mapping_nodes_to_raw_id(pd.Series(edge_index[1].cpu().numpy()))], axis=1)
        if edge_value is not None:
            if not isinstance(edge_value, list):
                edge_value = [edge_value]
        for v in edge_value:
            assert edge_index.shape[1] == v.shape[0]
            df = pd.concat([df, pd.Series(v.squeeze().cpu().numpy())], axis=1)
        return df
