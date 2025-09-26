import os
import yaml
import pgl
import time
import copy
import numpy as np
import pickle
import os.path as osp
from pgl.utils.logger import log
from pgl.graph import Graph
from pgl import graph_kernel
from pgl.sampling.custom import subgraph
from tqdm import tqdm


class CitationNetwork(object):
    """Citation Network Dataset Loader"""
    def __init__(self, config):
        self.data_dir = config.data_dir
        
        if os.path.isabs(self.data_dir):
            # If data_dir is absolute path, use it directly
            self.processed_dir = os.path.join(self.data_dir, "processed")
        else:
            # If data_dir is relative, resolve it properly
            # Get the current script's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to r_unimp directory, then to data_dir
            r_unimp_dir = os.path.dirname(current_dir)
            self.processed_dir = os.path.join(r_unimp_dir, self.data_dir, "processed")
            
        self.seed = config.seed
        self.m2v_dim = config.m2v_dim  # Get m2v dimension from config

        print("Self.processed_dir:", self.processed_dir)
        print("Checking if processed_dir exists:", os.path.exists(self.processed_dir))
        
        # Load metadata with proper path
        metadata_path = os.path.join(self.processed_dir, 'metadata.pkl')
        print("Metadata path:", metadata_path)
        print("Metadata file exists:", os.path.exists(metadata_path))
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        self.num_features = self.metadata['feature_dim']
        self.num_classes = self.metadata['num_classes']
        self.num_papers = self.metadata['num_papers']
        self.num_authors = self.metadata['num_authors']
        self.total_nodes = self.metadata.get('total_nodes', self.num_papers + self.num_authors)
        
    def prepare_data(self):
        """Prepare graph data for training"""
        log.info("Preparing Citation Network data...")
        
        # Load features - use the correct filename that our parser creates
        paper_features = np.load(os.path.join(self.processed_dir, 'paper_features.npy'))
        
        # Create features for all nodes (papers + authors)
        # Papers get their computed features, authors get random features
        author_features = np.random.normal(0, 0.05, (self.num_authors, self.num_features)).astype(np.float32)
        self.x = np.vstack([paper_features, author_features])
        
        # Load labels (only for papers)
        paper_labels = np.load(os.path.join(self.processed_dir, 'paper_labels.npy'))
        # Extend labels for all nodes (papers + authors), set author labels to 0
        self.y = np.zeros(self.total_nodes, dtype=np.int64)
        self.y[:self.num_papers] = paper_labels
        
        # Load data splits - use paper indices
        paper_indices = np.load(os.path.join(self.processed_dir, 'paper_indices.npy'))
        
        # Create train/val/test splits
        np.random.seed(self.seed)
        indices = np.random.permutation(len(paper_indices))
        
        train_ratio, val_ratio = 0.7, 0.15
        train_size = int(train_ratio * len(paper_indices))
        val_size = int(val_ratio * len(paper_indices))
        
        self.train_idx = indices[:train_size]
        self.val_idx = indices[train_size:train_size + val_size]
        self.test_idx = indices[train_size + val_size:]
        
        # Load graphs
        self.graph = []
        
        # Citation graph
        citation_path = os.path.join(self.processed_dir, 'paper_citation_graph')
        if os.path.exists(citation_path):
            citation_graph = Graph.load(citation_path, mmap_mode='r+')
            # Extend to include all nodes
            citation_graph_extended = self._extend_graph_to_all_nodes(citation_graph, 0)
            self.graph.append(citation_graph_extended)
        
        # Author-paper graphs
        ap_src_path = os.path.join(self.processed_dir, 'author_paper_graph_src')
        if os.path.exists(ap_src_path):
            ap_graph = Graph.load(ap_src_path, mmap_mode='r+')
            self.graph.append(ap_graph)
            
        ap_dst_path = os.path.join(self.processed_dir, 'author_paper_graph_dst')
        if os.path.exists(ap_dst_path):
            ap_graph = Graph.load(ap_dst_path, mmap_mode='r+')
            self.graph.append(ap_graph)

        # If no graphs loaded, create dummy graphs
        if len(self.graph) == 0:
            log.warning("No graphs found, creating dummy graphs")
            self._create_dummy_graphs()
            
        log.info(f"Loaded {len(self.graph)} graphs")
        
        # Create metapath2vec-like embeddings with correct dimension
        self.id_x = np.random.normal(0, 0.1, (self.total_nodes, self.m2v_dim)).astype(np.float32)
        
        # Initialize training data references
        self.train_idx_label = None
        self.train_idx_data = None
        
    def _extend_graph_to_all_nodes(self, graph, edge_type):
        """Extend graph to include all nodes (papers + authors)"""
        # Ensure the graph has the right number of nodes
        edges = graph.edges
        edge_types = np.full(len(edges), edge_type, dtype=np.int32)
        
        extended_graph = Graph(
            edges=edges,
            num_nodes=self.total_nodes,
            edge_feat={'edge_type': edge_types}
        )
        return extended_graph
        
    def _create_dummy_graphs(self):
        """Create dummy graphs for testing"""
        # Create a simple chain graph
        edges = []
        for i in range(min(1000, self.total_nodes - 1)):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Bidirectional
            
        edges = np.array(edges, dtype=np.int64)
        edge_types = np.zeros(len(edges), dtype=np.int32)
        
        dummy_graph = Graph(
            edges=edges,
            num_nodes=self.total_nodes,
            edge_feat={'edge_type': edge_types}
        )
        
        self.graph = [dummy_graph]
    
    @property
    def train_examples(self):
        """Get training examples"""
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        count_line = 0
        
        np.random.shuffle(self.train_idx)
        
        if self.train_idx_label is not None:
            del self.train_idx_label
        if self.train_idx_data is not None:
            del self.train_idx_data
            
        self.train_idx_label = set(self.train_idx)
        self.train_idx_data = self.train_idx
       
        for idx in self.train_idx_data:
            count_line += 1
            if count_line % trainer_num == trainer_id:
                yield idx
                
    @property        
    def eval_examples(self):
        """Get validation examples"""
        if self.train_idx_label is not None:
            del self.train_idx_label
        if self.train_idx_data is not None:
            del self.train_idx_data
        self.train_idx_label = set(self.train_idx)
        
        for idx in self.val_idx:
            yield idx
                
    @property
    def test_examples(self):
        """Get test examples"""
        self.train_idx_label = set(self.train_idx)
        for idx in self.test_idx:
            yield idx


def add_self_loop(graph, sub_nodes=None):
    """Add self loop for subgraph"""
    assert not graph.is_tensor(), "You must call Graph.numpy() first."
    
    if sub_nodes is not None:
        self_loop_edges = np.zeros((sub_nodes.shape[0], 2))
        self_loop_edges[:, 0] = self_loop_edges[:, 1] = sub_nodes
    else:
        self_loop_edges = np.zeros((graph.num_nodes, 2))
        self_loop_edges[:, 0] = self_loop_edges[:, 1] = np.arange(graph.num_nodes)
    edges = np.vstack((graph.edges, self_loop_edges))
    edges = np.unique(edges, axis=0)
    new_g = Graph(
        edges=edges,
        num_nodes=graph.num_nodes,
        )
    return new_g


def traverse(item):
    """Traverse nested list/array"""
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes):
    """Flatten node and edge lists"""
    nodes = list(set(traverse(nodes)))
    return nodes


def neighbor_sample(graph, nodes, samples):
    """Sample neighbors for nodes"""
    graph_list = []
    samples_list = [[25, 10, 5], [15, 10, 5]]  # Adjusted for 3 edge types
    
    for idi, max_deg in enumerate(samples):
        start_nodes = copy.deepcopy(nodes)
        edges = []
        edge_ids = []
        edge_feats = []
        neigh_nodes = [start_nodes]
        
        if max_deg == -1:
            pred_nodes, pred_eids = graph[0].predecessor(start_nodes, return_eids=True)
        else:
            for idj, g_t in enumerate(graph[:min(len(samples_list[idi]), len(graph))]):
                pred_nodes, pred_eids = g_t.sample_predecessor(
                    start_nodes, max_degree=samples_list[idi][idj], return_eids=True)
                neigh_nodes.append(pred_nodes)
                
                for dst_node, src_nodes, src_eids in zip(start_nodes, pred_nodes, pred_eids):
                    for src_node, src_eid in zip(src_nodes, src_eids):
                        edges.append((src_node, dst_node))
                        edge_ids.append(src_eid)
                        edge_feats.append(g_t.edge_feat['edge_type'][src_eid])
                        
        neigh_nodes = flat_node_and_edge(neigh_nodes)
        
        from_reindex = {x: i for i, x in enumerate(neigh_nodes)}
        sub_node_index = graph_kernel.map_nodes(nodes, from_reindex)
        
        sg = subgraph(graph[0],
                      eid=edge_ids,
                      nodes=neigh_nodes,
                      edges=edges,
                      with_node_feat=False,
                      with_edge_feat=False)
        edge_feats = np.array(edge_feats, dtype='int32')
        sg._edge_feat['edge_type'] = edge_feats
        
        graph_list.append((sg, neigh_nodes, sub_node_index))
        nodes = neigh_nodes
        
    graph_list = graph_list[::-1] 
    return graph_list, from_reindex


class BaseDataGenerator(object):
    """Base data generator class"""
    def __init__(self, buf_size, batch_size, num_workers, shuffle=True):
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
    def generator(self):
        """Generate batches"""
        batch = []
        for example in self.line_examples:
            batch.append(example)
            if len(batch) >= self.batch_size:
                yield self.batch_fn(batch)
                batch = []
        
        if len(batch) > 0:
            yield self.batch_fn(batch)


class DataGenerator(BaseDataGenerator):
    """Data generator for citation network"""
    def __init__(self, dataset, samples, batch_size, num_workers, data_type):
        super(DataGenerator, self).__init__(
            buf_size=10, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=True if data_type=='train' else False
        )
        
        self.dataset = dataset
        self.samples = samples
        
        if data_type == 'train':
            self.line_examples = self.dataset.train_examples
        elif data_type == 'eval':
            self.line_examples = self.dataset.eval_examples
        else: 
            self.line_examples = self.dataset.test_examples

    def batch_fn(self, batch_nodes):
        """Create batch data"""
        graph_list, from_reindex = neighbor_sample(
            self.dataset.graph, batch_nodes, self.samples)
        
        neigh_nodes = graph_list[0][1]
        neigh_nodes = np.array(neigh_nodes, dtype='int32')
        y = self.dataset.y[batch_nodes]
        
        # Get labeled nodes for label propagation
        label_idx = list((set(neigh_nodes) - set(batch_nodes)) & self.dataset.train_idx_label)
        if len(label_idx) == 0:
            # If no labeled neighbors, use some random training nodes
            available_train = list(self.dataset.train_idx_label)
            label_idx = available_train[:min(10, len(available_train))]
            
        sub_label_index = graph_kernel.map_nodes(label_idx, from_reindex)
        sub_label_y = self.dataset.y[label_idx]
        
        # Create dummy position encoding (since we don't have year info)
        pos = np.random.randint(0, 50, len(neigh_nodes))
        
        return graph_list, neigh_nodes, y, sub_label_y, sub_label_index, pos
    
    def post_fn(self, batch):
        """Post-process batch data"""
        graph_list, neigh_nodes, y, sub_label_y, sub_label_index, pos = batch
        
        print(f"DEBUG post_fn - neigh_nodes shape: {neigh_nodes.shape}")
        print(f"DEBUG post_fn - self.dataset.x shape: {self.dataset.x.shape}")
        print(f"DEBUG post_fn - self.dataset.id_x shape: {self.dataset.id_x.shape}")
        
        x = self.dataset.x[neigh_nodes]
        id_x = self.dataset.id_x[neigh_nodes]
        
        print(f"DEBUG post_fn - x shape after indexing: {x.shape}")
        print(f"DEBUG post_fn - id_x shape after indexing: {id_x.shape}")
        
        # Simple position encoding
        pos_encoding = np.random.normal(0, 0.1, (len(neigh_nodes), x.shape[1])).astype(np.float32)
        x = x + pos_encoding * 0.1  # Small positional perturbation
        
        return graph_list, x, id_x, y, sub_label_y, sub_label_index


if __name__ == "__main__":
    from easydict import EasyDict as edict
    
    config = edict({
        'data_dir': '.',
        'seed': 42,
        'm2v_dim': 512
    })
    
    dataset = CitationNetwork(config)
    dataset.prepare_data()
    
    print(f"Dataset loaded: {dataset.num_papers} papers, {dataset.num_authors} authors")
    print(f"Features shape: {dataset.x.shape}")
    print(f"M2V features shape: {dataset.id_x.shape}")
    print(f"Classes: {dataset.num_classes}")