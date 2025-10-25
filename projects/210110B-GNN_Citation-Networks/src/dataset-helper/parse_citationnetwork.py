import os
import re
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationNetworkParser:
    def __init__(self, data_file):
        self.data_file = data_file
        self.papers = {}
        self.authors = set()
        self.venues = set()
        self.citations = []
        self.author_paper_edges = []
        self.venue_to_label = {}
        
    def parse_dataset(self):
        """Parse the outputacm.txt file"""
        logger.info("Starting to parse dataset...")
        
        current_paper = {}
        paper_id = 0
        
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(tqdm(f, desc="Parsing lines")):
                line = line.strip()
                
                if line.startswith('#*'):  # Title
                    if current_paper:  # Save previous paper
                        self._save_paper(current_paper, paper_id)
                        paper_id += 1
                        current_paper = {}
                    
                    current_paper['title'] = line[2:].strip()
                    current_paper['index'] = None
                    current_paper['authors'] = []
                    current_paper['year'] = None
                    current_paper['venue'] = ''
                    current_paper['references'] = []
                    
                elif line.startswith('#@'):  # Authors
                    authors = line[2:].strip().split(',')
                    current_paper['authors'] = [author.strip() for author in authors]
                    for author in current_paper['authors']:
                        self.authors.add(author)
                        
                elif line.startswith('#t'):  # Year
                    try:
                        current_paper['year'] = int(line[2:].strip())
                    except ValueError:
                        current_paper['year'] = 2000  # Default year
                        
                elif line.startswith('#c'):  # Venue
                    venue = line[2:].strip()
                    current_paper['venue'] = venue
                    if venue:
                        self.venues.add(venue)
                        
                elif line.startswith('#index'):  # Paper index
                    try:
                        current_paper['index'] = int(line[6:])
                    except ValueError:
                        current_paper['index'] = paper_id
                        
                elif line.startswith('#!'):  # References
                    # Extract referenced paper indices
                    ref_text = line[2:].strip()
                    # Look for index numbers in the reference text
                    indices = re.findall(r'#index(\d+)', ref_text)
                    for idx in indices:
                        try:
                            current_paper['references'].append(int(idx))
                        except ValueError:
                            pass
        
        # Save the last paper
        if current_paper:
            self._save_paper(current_paper, paper_id)
            
        logger.info(f"Parsed {len(self.papers)} papers")
        logger.info(f"Found {len(self.authors)} unique authors")
        logger.info(f"Found {len(self.venues)} unique venues")
        
    def _save_paper(self, paper_data, paper_id):
        """Save paper data and create edges"""
        if 'title' not in paper_data:
            return
            
        # Use the index from file or assign sequential ID
        idx = paper_data.get('index', paper_id)
        
        self.papers[idx] = {
            'title': paper_data['title'],
            'authors': paper_data['authors'],
            'year': paper_data.get('year', 2000),
            'venue': paper_data.get('venue', ''),
            'references': paper_data.get('references', [])
        }
        
        # Create author-paper edges
        for author in paper_data['authors']:
            self.author_paper_edges.append((author, idx))
            
        # Create citation edges
        for ref_idx in paper_data['references']:
            self.citations.append((idx, ref_idx))  # (citing_paper, cited_paper)
    
    def create_venue_labels(self):
        """Create venue-based labels for classification"""
        logger.info("Creating venue-based labels...")
        
        # Group venues and create labels
        venue_list = list(self.venues)
        venue_list.sort()  # Sort for consistency
        
        # Create mapping from venue to label
        for i, venue in enumerate(venue_list):
            self.venue_to_label[venue] = i
            
        logger.info(f"Created {len(self.venue_to_label)} venue labels")
        
    def create_features(self):
        """Create simple features for papers"""
        logger.info("Creating paper features...")
        
        # Create simple features based on title length, author count, year
        paper_indices = list(self.papers.keys())
        paper_indices.sort()
        
        features = []
        labels = []
        
        # Limit number of classes to make training manageable
        max_classes = 100
        
        for idx in paper_indices:
            paper = self.papers[idx]
            
            # Simple features: [title_length, num_authors, normalized_year]
            title_len = len(paper['title'].split())
            num_authors = len(paper['authors'])
            norm_year = (paper['year'] - 1900) / 100.0  # Normalize year
            
            # Create a simple 768-dimensional feature vector
            feat = np.zeros(768)
            feat[0] = title_len / 20.0  # Normalize title length
            feat[1] = num_authors / 10.0  # Normalize author count
            feat[2] = norm_year
            
            # Add some random features for diversity
            feat[3:] = np.random.normal(0, 0.1, 765)
            
            features.append(feat)
            
            # Create label based on venue (limited to max_classes)
            venue = paper['venue']
            if venue in self.venue_to_label:
                label = self.venue_to_label[venue] % max_classes
                labels.append(label)
            else:
                labels.append(0)  # Default label
                
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64), paper_indices
        
    def save_processed_data(self, output_dir):
        """Save all processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Saving processed data...")
        
        # Save paper information
        with open(os.path.join(output_dir, 'papers.pkl'), 'wb') as f:
            pickle.dump(self.papers, f)
            
        # Save citation edges
        citation_edges = np.array(self.citations, dtype=np.int64)
        np.save(os.path.join(output_dir, 'citation_edges.npy'), citation_edges)
        
        # Save author-paper edges
        with open(os.path.join(output_dir, 'author_paper_edges.pkl'), 'wb') as f:
            pickle.dump(self.author_paper_edges, f)
            
        # Save venue labels
        with open(os.path.join(output_dir, 'venue_labels.pkl'), 'wb') as f:
            pickle.dump(self.venue_to_label, f)
            
        # Create and save features
        features, labels, paper_indices = self.create_features()
        np.save(os.path.join(output_dir, 'paper_features.npy'), features)
        np.save(os.path.join(output_dir, 'paper_labels.npy'), labels)
        np.save(os.path.join(output_dir, 'paper_indices.npy'), np.array(paper_indices))
        
        # Save metadata
        metadata = {
            'num_papers': len(self.papers),
            'num_authors': len(self.authors),
            'num_venues': len(self.venues),
            'num_citations': len(self.citations),
            'feature_dim': 768,
            'num_classes': min(100, len(self.venue_to_label)),  # Limit to 100 classes
            'total_nodes': len(self.papers) + len(self.authors)
        }
        
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved all data to {output_dir}")
        logger.info(f"Dataset statistics: {metadata}")

def main():
    data_file = "outputacm.txt"
    output_dir = "processed"
    
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found!")
        return
        
    parser = CitationNetworkParser(data_file)
    parser.parse_dataset()
    parser.create_venue_labels()
    parser.save_processed_data(output_dir)

if __name__ == "__main__":
    main()
