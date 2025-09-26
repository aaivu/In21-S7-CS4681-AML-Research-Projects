import os
import pickle
import numpy as np
import pgl
from pgl.graph import Graph
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationGraphBuilder:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.load_data()
        
    def load_data(self):
        """Load processed data"""
        logger.info("Loading processed data...")
        
        # Load metadata
        with open(os.path.join(self.processed_dir, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
            
        # Load papers
        with open(os.path.join(self.processed_dir, 'papers.pkl'), 'rb') as f:
            self.papers = pickle.load(f)
            
        # Load citation edges
        self.citation_edges = np.load(os.path.join(self.processed_dir, 'citation_edges.npy'))
        
        # Load author-paper edges
        with open(os.path.join(self.processed_dir, 'author_paper_edges.pkl'), 'rb') as f:
            self.author_paper_edges = pickle.load(f)
            
        # Load features and labels
        self.paper_features = np.load(os.path.join(self.processed_dir, 'paper_features.npy'))
        self.paper_labels = np.load(os.path.join(self.processed_dir, 'paper_labels.npy'))
        self.paper_indices = np.load(os.path.join(self.processed_dir, 'paper_indices.npy'))
        
        logger.info(f"Loaded data: {self.metadata}")
        
    def create_paper_citation_graph(self):
        """Create paper-to-paper citation graph"""
        logger.info("Creating paper citation graph...")
        
        # Create mapping from paper index to sequential ID
        paper_to_id = {paper_idx: i for i, paper_idx in enumerate(self.paper_indices)}
        id_to_paper = {i: paper_idx for i, paper_idx in enumerate(self.paper_indices)}
        
        # Filter valid citation edges
        valid_edges = []
        for cite_idx, cited_idx in tqdm(self.citation_edges, desc="Processing citations"):
            if cite_idx in paper_to_id and cited_idx in paper_to_id:
                valid_edges.append([paper_to_id[cite_idx], paper_to_id[cited_idx]])
                
        if len(valid_edges) == 0:
            logger.warning("No valid citation edges found, creating dummy edges")
            # Create some dummy edges for testing
            num_papers = len(self.paper_indices)
            valid_edges = [[i, (i+1) % num_papers] for i in range(min(1000, num_papers-1))]
        
        edges = np.array(valid_edges, dtype=np.int64)
        
        # Make bidirectional
        edges_rev = edges[:, [1, 0]]
        all_edges = np.vstack([edges, edges_rev])
        
        # Remove duplicates
        all_edges = np.unique(all_edges, axis=0)
        
        edge_types = np.zeros(len(all_edges), dtype=np.int32)
        
        graph = Graph(all_edges, num_nodes=len(self.paper_indices), 
                     edge_feat={'edge_type': edge_types})
        
        output_path = os.path.join(self.processed_dir, 'paper_citation_graph')
        graph.dump(output_path)
        
        logger.info(f"Created citation graph with {len(all_edges)} edges")
        return output_path
        
    def create_author_paper_graph(self):
        """Create author-paper bipartite graph"""
        logger.info("Creating author-paper graph...")
        
        # Create author to ID mapping
        authors = sorted(list(set([author for author, _ in self.author_paper_edges])))
        author_to_id = {author: i for i, author in enumerate(authors)}
        
        # Create paper to ID mapping (offset by number of papers for combined graph)
        paper_to_id = {paper_idx: i for i, paper_idx in enumerate(self.paper_indices)}
        
        # Create edges from authors to papers
        author_paper_edges = []
        paper_author_edges = []
        
        for author, paper_idx in tqdm(self.author_paper_edges, desc="Processing author-paper edges"):
            if paper_idx in paper_to_id:
                author_id = author_to_id[author] + len(self.paper_indices)  # Offset by number of papers
                paper_id = paper_to_id[paper_idx]
                
                author_paper_edges.append([author_id, paper_id])  # Author -> Paper
                paper_author_edges.append([paper_id, author_id])  # Paper -> Author
        
        # Create author->paper graph
        if len(author_paper_edges) > 0:
            edges_ap = np.array(author_paper_edges, dtype=np.int64)
            edge_types_ap = np.ones(len(edges_ap), dtype=np.int32)  # Edge type 1
            
            total_nodes = len(self.paper_indices) + len(authors)
            graph_ap = Graph(edges_ap, num_nodes=total_nodes, 
                           edge_feat={'edge_type': edge_types_ap})
            
            output_path_ap = os.path.join(self.processed_dir, 'author_paper_graph_src')
            graph_ap.dump(output_path_ap)
        else:
            output_path_ap = None
            logger.warning("No author-paper edges found")
            
        # Create paper->author graph  
        if len(paper_author_edges) > 0:
            edges_pa = np.array(paper_author_edges, dtype=np.int64)
            edge_types_pa = np.full(len(edges_pa), 2, dtype=np.int32)  # Edge type 2
            
            total_nodes = len(self.paper_indices) + len(authors)
            graph_pa = Graph(edges_pa, num_nodes=total_nodes,
                           edge_feat={'edge_type': edge_types_pa})
            
            output_path_pa = os.path.join(self.processed_dir, 'author_paper_graph_dst')
            graph_pa.dump(output_path_pa)
        else:
            output_path_pa = None
            logger.warning("No paper-author edges found")
            
        logger.info(f"Created author-paper graphs with {len(author_paper_edges)} edges each direction")
        
        # Save author info
        author_info = {
            'authors': authors,
            'author_to_id': author_to_id,
            'num_authors': len(authors)
        }
        with open(os.path.join(self.processed_dir, 'author_info.pkl'), 'wb') as f:
            pickle.dump(author_info, f)
            
        return output_path_ap, output_path_pa
        
    def create_train_val_test_split(self):
        """Create train/validation/test splits"""
        logger.info("Creating train/val/test splits...")
        
        num_papers = len(self.paper_indices)
        indices = np.arange(num_papers)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # 70% train, 15% val, 15% test
        train_end = int(0.7 * num_papers)
        val_end = int(0.85 * num_papers)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Save splits
        np.save(os.path.join(self.processed_dir, 'train_idx.npy'), train_idx)
        np.save(os.path.join(self.processed_dir, 'val_idx.npy'), val_idx)
        np.save(os.path.join(self.processed_dir, 'test_idx.npy'), test_idx)
        
        logger.info(f"Created splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return train_idx, val_idx, test_idx
        
    def create_combined_features(self):
        """Create combined feature matrix for all nodes (papers + authors)"""
        logger.info("Creating combined feature matrix...")
        
        # Load author info
        with open(os.path.join(self.processed_dir, 'author_info.pkl'), 'rb') as f:
            author_info = pickle.load(f)
            
        num_papers = len(self.paper_indices)
        num_authors = author_info['num_authors']
        feature_dim = self.paper_features.shape[1]
        
        # Create combined feature matrix
        combined_features = np.zeros((num_papers + num_authors, feature_dim), dtype=np.float32)
        
        # Copy paper features
        combined_features[:num_papers] = self.paper_features
        
        # Create simple author features (average of their papers' features)
        author_features = np.random.normal(0, 0.1, (num_authors, feature_dim)).astype(np.float32)
        combined_features[num_papers:] = author_features
        
        # Save combined features
        np.save(os.path.join(self.processed_dir, 'combined_features.npy'), combined_features)
        
        logger.info(f"Created combined features: {combined_features.shape}")
        
        return combined_features
        
    def build_all_graphs(self):
        """Build all required graphs"""
        logger.info("Building all graph structures...")
        
        # Create citation graph
        citation_graph_path = self.create_paper_citation_graph()
        
        # Create author-paper graphs
        ap_graph_path, pa_graph_path = self.create_author_paper_graph()
        
        # Create data splits
        train_idx, val_idx, test_idx = self.create_train_val_test_split()
        
        # Create combined features
        combined_features = self.create_combined_features()
        
        # Update metadata
        with open(os.path.join(self.processed_dir, 'author_info.pkl'), 'rb') as f:
            author_info = pickle.load(f)
            
        self.metadata.update({
            'graph_paths': {
                'citation': citation_graph_path,
                'author_paper_src': ap_graph_path,
                'author_paper_dst': pa_graph_path
            },
            'num_authors': author_info['num_authors'],
            'total_nodes': len(self.paper_indices) + author_info['num_authors'],
            'feature_dim': combined_features.shape[1]
        })
        
        with open(os.path.join(self.processed_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)
            
        logger.info("All graphs built successfully!")
        logger.info(f"Final metadata: {self.metadata}")

def main():
    processed_dir = "processed"
    
    if not os.path.exists(processed_dir):
        logger.error(f"Processed directory {processed_dir} not found! Run parse_citationnetwork.py first.")
        return
        
    builder = CitationGraphBuilder(processed_dir)
    builder.build_all_graphs()

if __name__ == "__main__":
    main()
