#!/usr/bin/env python3
"""
Semantic retrieval method implementation
Use Sentence Transformers for semantic similarity retrieval, support Elasticsearch integration
"""

import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from elasticsearch import Elasticsearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set Elasticsearch log level to WARNING, avoid too many debug information
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)

class SemanticRetriever:
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", 
                 index_name: str = "financial_corpus",
                 host: str = "localhost", 
                 port: int = 9200,
                 use_elasticsearch: bool = True,
                 device: str = None):
        """
        Initialize semantic retriever
        
        Args:
            model_name: Sentence Transformer model name
            index_name: Elasticsearch index name
            host: Elasticsearch host address
            port: Elasticsearch port
            use_elasticsearch: Whether to use Elasticsearch
            device: Device to use ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc.). 
                   If None, auto-detect (prefer GPU if available)
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                # Try to find an available GPU (prefer GPU 1, 2, 3 over 0)
                for gpu_id in [1, 2, 3, 0]:
                    try:
                        # Check if GPU has free memory
                        if torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id) > 1024 * 1024 * 1024:  # At least 1GB free
                            device = f'cuda:{gpu_id}'
                            print(f"Auto-selected GPU {gpu_id} (has free memory)")
                            break
                    except:
                        continue
                if device is None:
                    device = 'cpu'
                    print("No GPU with sufficient memory found, using CPU")
            else:
                device = 'cpu'
                print("CUDA not available, using CPU")
        else:
            print(f"Using specified device: {device}")
        
        self.device = device
        
        print(f"Loading semantic retrieval model: {model_name} on {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            print(f"Semantic retrieval model loaded successfully, dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Failed to load semantic retrieval model: {e}")
            # Fallback to CPU if GPU fails
            if device != 'cpu':
                print("Falling back to CPU...")
                try:
                    self.model = SentenceTransformer(model_name, device='cpu')
                    self.device = 'cpu'
                    print(f"Model loaded on CPU, dimension: {self.model.get_sentence_embedding_dimension()}")
                except Exception as e2:
                    print(f"Failed to load model on CPU: {e2}")
                    raise
            else:
                raise
        
        self.use_elasticsearch = use_elasticsearch
        
        if use_elasticsearch:
            self.index_name = index_name
            try:
                self.es = Elasticsearch([{'host': host, 'port': port}])
                
                # Check Elasticsearch connection
                if not self.es.ping():
                    raise ConnectionError(f"Cannot connect to Elasticsearch {host}:{port}")
                
                print(f"Successfully connected to Elasticsearch: {host}:{port}")
                
                # Check if index exists
                if not self.es.indices.exists(index=index_name):
                    print(f"Warning: Index {index_name} does not exist")
                    print("Creating index with vector field...")
                    self._create_index()
                else:
                    # Check index mapping
                    try:
                        mapping = self.es.indices.get_mapping(index=index_name)
                        properties = mapping[index_name]['mappings']['properties']
                        print(f"Index {index_name} exists, contains fields: {list(properties.keys())}")
                        
                        if 'text_vector' in properties:
                            print("✓ Detected vector field, support semantic search")
                        else:
                            print("⚠ Detected no vector field, rebuilding index...")
                            # Delete old index and rebuild
                            self.es.indices.delete(index=index_name)
                            self._create_index()
                            
                    except Exception as e:
                        print(f"Error checking index mapping: {e}")
                        print("Rebuilding index...")
                        try:
                            self.es.indices.delete(index=index_name)
                        except:
                            pass
                        self._create_index()
                
                logger.info(f"Semantic retriever initialized, using Elasticsearch index: {index_name}")
                
            except Exception as e:
                print(f"Elasticsearch connection failed: {e}")
                print("Switching to local mode")
                self.use_elasticsearch = False
                self.es = None
        else:
            # Traditional mode: load local corpus
            self.corpus = []
            self.corpus_ids = []
            self.corpus_embeddings = None
            print("Using local corpus mode")
    
    def _create_index(self):
        """Create Elasticsearch index with vector field"""
        try:
            # Get model vector dimension
            vector_dims = self.model.get_sentence_embedding_dimension()
            
            # Define index mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "search_analyzer": "standard"
                        },
                        "text_vector": {
                            "type": "dense_vector",
                            "dims": vector_dims
                        },
                        "file_path": {
                            "type": "keyword"
                        },
                        "created_at": {
                            "type": "date"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
            
            self.es.indices.create(index=self.index_name, body=mapping)
            print(f"✓ Index {self.index_name} created successfully, contains vector field (dimension: {vector_dims})")
            
        except Exception as e:
            print(f"Failed to create index: {e}")
            # If creation fails, try to create a simple text index
            try:
                simple_mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "file_path": {"type": "keyword"}
                        }
                    }
                }
                self.es.indices.create(index=self.index_name, body=simple_mapping)
                print(f"⚠ Created simple index {self.index_name} (no vector field)")
            except Exception as e2:
                print(f"Failed to create simple index: {e2}")
                raise
     
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> bool:
        """
        Add documents to Elasticsearch index
        
        Args:
            documents: Document list, each document should contain text, file_path etc.
            batch_size: Batch size for encoding documents (default: 32)
            
        Returns:
            bool: Whether successful
        """
        if not self.use_elasticsearch:
            logger.warning("Not using Elasticsearch mode, cannot add documents")
            return False
            
        try:
            print(f"Adding {len(documents)} documents to index {self.index_name}...")
            print(f"Using batch size: {batch_size}, device: {self.device}")
            
            # Process documents in batches for encoding
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(documents), batch_size):
                batch_docs = documents[batch_idx:batch_idx + batch_size]
                batch_num = batch_idx // batch_size + 1
                
                # Prepare batch texts
                batch_texts = []
                batch_doc_data = []
                
                for i, doc in enumerate(batch_docs):
                    doc_idx = batch_idx + i
                    text_content = doc.get('text', doc.get('content', ''))
                    batch_texts.append(text_content)
                    
                    # Extract original _id from document (critical for doc_id matching)
                    doc_id = doc.get('_id', doc.get('id', f"doc_{doc_idx}"))
                    
                    doc_data = {
                        'text': text_content,
                        'file_path': doc.get('file_path', f'doc_{doc_idx}'),
                        'created_at': doc.get('created_at', '2024-01-01T00:00:00'),
                        'original_id': doc_id  # Store original ID for reference
                    }
                    batch_doc_data.append((doc_idx, doc_data, doc_id))
                
                # Batch encode all texts at once (more efficient)
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=False,  # Return numpy array directly
                        show_progress_bar=False,
                        batch_size=min(batch_size, len(batch_texts)),
                        device=self.device
                    )
                    
                    # Add documents to Elasticsearch
                    for (doc_idx, doc_data, doc_id), embedding in zip(batch_doc_data, batch_embeddings):
                        doc_data['text_vector'] = embedding.tolist()
                        
                        # Use original _id from JSONL file as Elasticsearch document ID
                        # This ensures doc_ids returned by retrieval match the original corpus
                        self.es.index(
                            index=self.index_name,
                            id=str(doc_id),  # Use original _id from JSONL
                            body=doc_data
                        )
                    
                    print(f"Processed batch {batch_num}/{total_batches} ({batch_idx + len(batch_docs)}/{len(documents)} documents)")
                    
                except Exception as e:
                    print(f"Error processing batch {batch_num}: {e}")
                    # Fallback: process one by one
                    print("Falling back to individual document processing...")
                    for doc_idx, doc in enumerate(batch_docs):
                        try:
                            text_content = doc.get('text', doc.get('content', ''))
                            if text_content:
                                embedding = self.model.encode(
                                    text_content,
                                    convert_to_tensor=False,
                                    show_progress_bar=False,
                                    device=self.device
                                )
                                
                                # Extract original _id from document (critical for doc_id matching)
                                doc_id = doc.get('_id', doc.get('id', f"doc_{batch_idx + doc_idx}"))
                                
                                doc_data = {
                                    'text': text_content,
                                    'text_vector': embedding.tolist(),
                                    'file_path': doc.get('file_path', f'doc_{batch_idx + doc_idx}'),
                                    'created_at': doc.get('created_at', '2024-01-01T00:00:00'),
                                    'original_id': doc_id  # Store original ID for reference
                                }
                                
                                # Use original _id from JSONL file as Elasticsearch document ID
                                # This ensures doc_ids returned by retrieval match the original corpus
                                self.es.index(
                                    index=self.index_name,
                                    id=str(doc_id),  # Use original _id from JSONL
                                    body=doc_data
                                )
                        except Exception as e2:
                            print(f"Error processing document {batch_idx + doc_idx}: {e2}")
                            continue
            
            # Refresh index to ensure documents are searchable
            self.es.indices.refresh(index=self.index_name)
            print(f"✓ Successfully added {len(documents)} documents to index {self.index_name}")
            return True
            
        except Exception as e:
            print(f"Failed to add documents: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def load_corpus(self, corpus_dir: str):
        """
        Load corpus (only when not using Elasticsearch)
        
        Args:
            corpus_dir: Corpus directory path
        """
        if self.use_elasticsearch:
            logger.warning("Using Elasticsearch mode does not need to load local corpus")
            return
            
        print(f"Loading corpus: {corpus_dir}")
        
        # Support multiple file formats
        supported_extensions = ['.txt', '.json', '.md']
        
        for root, dirs, files in os.walk(corpus_dir):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        if file.endswith('.json'):
                            # Process JSON file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        if 'content' in item:
                                            self.corpus.append(item['content'])
                                            self.corpus_ids.append(f"{file}:{item.get('id', len(self.corpus))}")
                                elif isinstance(data, dict) and 'content' in data:
                                    self.corpus.append(data['content'])
                                    self.corpus_ids.append(f"{file}:{data.get('id', '0')}")
                        else:
                            # Process text file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Split by paragraphs
                                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                                for i, para in enumerate(paragraphs):
                                    if len(para) > 50:  # Filter too short paragraphs
                                        self.corpus.append(para)
                                        self.corpus_ids.append(f"{file}:{i}")
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
        
        print(f"Corpus loaded, total {len(self.corpus)} document fragments")
        
        if self.corpus:
            self._build_embeddings()
    
    def _build_embeddings(self):
        """Build document embeddings (only when not using Elasticsearch)"""
        if self.use_elasticsearch:
            return
            
        print("Building document embeddings...")
        
        # Batch calculate embeddings to improve efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(self.corpus), batch_size):
            batch = self.corpus[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
        
        # Merge all batches of embeddings
        self.corpus_embeddings = torch.cat(embeddings, dim=0)
        print("Document embeddings built!")
    
    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[str, float, str]]:
        """
        Semantic retrieval related documents
        
        Args:
            query: Query text
            top_k: Return the top-k results
            similarity_threshold: Similarity threshold, results below this value will be filtered
            
        Returns:
            Retrieval results list, each element contains (document content, similarity score, document ID)
        """
        if self.use_elasticsearch:
            return self._retrieve_elasticsearch(query, top_k, similarity_threshold)
        else:
            return self._retrieve_local(query, top_k, similarity_threshold)
    
    def _retrieve_elasticsearch(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[str, float, str]]:
        """Use Elasticsearch for semantic retrieval"""
        try:
            # First check if the index has a vector field
            mapping = self.es.indices.get_mapping(index=self.index_name)
            has_vector_field = 'text_vector' in mapping[self.index_name]['mappings']['properties']
            
            if has_vector_field:
                # Use vector search
                query_embedding = self.model.encode(query, convert_to_tensor=True)
                query_vector = query_embedding.cpu().numpy().tolist()
                
                # Verify vector dimension matches index
                vector_dims = len(query_vector)
                mapping_dims = mapping[self.index_name]['mappings']['properties'].get('text_vector', {}).get('dims', 0)
                
                if mapping_dims > 0 and vector_dims != mapping_dims:
                    logger.warning(f"Query vector dimension ({vector_dims}) does not match index dimension ({mapping_dims}), falling back to text search")
                    return self._fallback_text_search(query, top_k)
                
                # Use script_score query for cosine similarity
                search_body = {
                    "query": {
                        "script_score": {
                            "query": {
                                "bool": {
                                    "must": {"match_all": {}},
                                    "filter": {
                                        "exists": {"field": "text_vector"}
                                    }
                                }
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    },
                    "size": top_k,
                    "_source": ["text"]
                }
            else:
                # If there is no vector field, use text search
                logger.info("Index has no vector field, using text search")
                return self._fallback_text_search(query, top_k)
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                content = hit['_source']['text']
                
                # Apply similarity threshold
                if score >= similarity_threshold:
                    results.append((content, score, doc_id))
            
            logger.info(f"Elasticsearch semantic retrieval found {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Elasticsearch semantic retrieval error: {e}")
            
            # Check if it's a dimension mismatch error
            if "runtime error" in error_msg.lower() or "dimension" in error_msg.lower():
                logger.warning("Possible vector dimension mismatch detected. Checking dimensions...")
                try:
                    query_dims = len(query_vector) if 'query_vector' in locals() else 0
                    mapping_dims = mapping[self.index_name]['mappings']['properties'].get('text_vector', {}).get('dims', 0)
                    model_dims = self.model.get_sentence_embedding_dimension()
                    logger.warning(f"Query vector dimension: {query_dims}, Index dimension: {mapping_dims}, Model dimension: {model_dims}")
                    if query_dims != mapping_dims:
                        logger.error(f"Dimension mismatch! Query vector ({query_dims}D) does not match index ({mapping_dims}D). "
                                   f"Please use the same model that was used to create the index, or recreate the index with the current model.")
                except:
                    pass
            
            # If vector search fails, fallback to text search
            logger.info("Fallback to text search...")
            return self._fallback_text_search(query, top_k)
    
    def _fallback_text_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Fallback to text search"""
        try:
            logger.info(f"Executing text search, query: {query[:100]}...")
            
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "type": "best_fields",
                        "fields": ["text"],
                        "tie_breaker": 0.5
                    }
                },
                "size": top_k,
                "_source": ["text"]
            }
            
            logger.info(f"Text search query body: {search_body}")
            
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                content = hit['_source']['text']
                results.append((content, score, doc_id))
            
            logger.info(f"Fallback text search successfully found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Fallback text search failed: {e}")
            logger.error(f"Detailed error information: {str(e)}")
            import traceback
            logger.error(f"Error stack: {traceback.format_exc()}")
            return []
    
    def _retrieve_local(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[str, float, str]]:
        """Use local corpus for semantic retrieval"""
        if self.corpus_embeddings is None:
            raise ValueError("Document embeddings not built, please load corpus first")
        
        # Calculate query embedding vector
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Get top-k results
        top_results = torch.topk(cos_scores, min(top_k, len(cos_scores)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            score_value = score.item()
            if score_value >= similarity_threshold:
                results.append((
                    self.corpus[idx],
                    score_value,
                    self.corpus_ids[idx]
                ))
        
        return results
    
    def retrieve_with_reranking(self, query: str, top_k: int = 5, rerank_top_k: int = 20) -> List[Tuple[str, float, str]]:
        """
        Semantic retrieval with reranking
        
        Args:
            query: Query text
            top_k: Return the top-k results
            rerank_top_k: Consider the top-k results when reranking
            
        Returns:
            Retrieval results list
        """
        # First get more candidate results
        candidates = self.retrieve(query, top_k=rerank_top_k, similarity_threshold=0.1)
        
        if not candidates:
            return []
        
        # Use more precise similarity calculation for reranking
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        reranked_results = []
        for content, _, doc_id in candidates:
            # Recalculate similarity
            content_embedding = self.model.encode(content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, content_embedding)[0][0].item()
            reranked_results.append((content, similarity, doc_id))
        
        # Sort by similarity
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k]
    
    def add_document(self, content: str, doc_id: str = None):
        """
        Add new document
        
        Args:
            content: Document content
            doc_id: Document ID
        """
        if self.use_elasticsearch:
            self._add_document_elasticsearch(content, doc_id)
        else:
            self._add_document_local(content, doc_id)
    
    def _add_document_elasticsearch(self, content: str, doc_id: str = None):
        """Add document to Elasticsearch index"""
        try:
            # Calculate document embedding vector
            content_embedding = self.model.encode(content, convert_to_tensor=True)
            content_vector = content_embedding.cpu().numpy().tolist()
            
            document = {
                "text": content,
                "text_vector": content_vector
            }
            
            if doc_id:
                # Use specified ID
                self.es.index(
                    index=self.index_name,
                    id=doc_id,
                    body=document
                )
            else:
                # Automatically generate ID
                self.es.index(
                    index=self.index_name,
                    body=document
                )
            
            # Refresh index
            self.es.indices.refresh(index=self.index_name)
            logger.info(f"Successfully added document to Elasticsearch index")
            
        except Exception as e:
            logger.error(f"Error adding document to Elasticsearch: {e}")
    
    def _add_document_local(self, content: str, doc_id: str = None):
        """Add document to local corpus"""
        if doc_id is None:
            doc_id = f"manual:{len(self.corpus)}"
        
        self.corpus.append(content)
        self.corpus_ids.append(doc_id)
        
        # Calculate new document embedding vector
        new_embedding = self.model.encode(content, convert_to_tensor=True)
        
        if self.corpus_embeddings is None:
            self.corpus_embeddings = new_embedding.unsqueeze(0)
        else:
            self.corpus_embeddings = torch.cat([self.corpus_embeddings, new_embedding.unsqueeze(0)], dim=0)
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get corpus statistics
        
        Returns:
            Corpus statistics
        """
        if self.use_elasticsearch:
            return self._get_elasticsearch_stats()
        else:
            return self._get_local_stats()
    
    def _get_elasticsearch_stats(self) -> Dict[str, Any]:
        """Get Elasticsearch index statistics"""
        try:
            # Get index statistics
            stats = self.es.indices.stats(index=self.index_name)
            index_stats = stats['indices'][self.index_name]
            
            # Get total number of documents
            count_response = self.es.count(index=self.index_name)
            total_docs = count_response['count']
            
            return {
                "index_name": self.index_name,
                "total_documents": total_docs,
                "index_size_bytes": index_stats['total']['store']['size_in_bytes'],
                "index_size_mb": round(index_stats['total']['store']['size_in_bytes'] / (1024 * 1024), 2),
                "segments_count": index_stats['total']['segments']['count'],
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "retrieval_method": "Elasticsearch + Sentence Transformers"
            }
            
        except Exception as e:
            logger.error(f"Error getting Elasticsearch statistics: {e}")
            return {"error": str(e)}
    
    def _get_local_stats(self) -> Dict[str, Any]:
        """Get local corpus statistics"""
        return {
            "total_documents": len(self.corpus),
            "total_characters": sum(len(doc) for doc in self.corpus),
            "avg_document_length": np.mean([len(doc) for doc in self.corpus]) if self.corpus else 0,
            "embedding_dimension": self.corpus_embeddings.shape[1] if self.corpus_embeddings is not None else 0,
            "corpus_ids": self.corpus_ids[:10],  # Only show the first 10 IDs
            "retrieval_method": "Local + Sentence Transformers"
        }
    
    def find_similar_documents(self, document: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Find the most similar documents to the given document
        
        Args:
            document: Target document
            top_k: Return the top-k results
            
        Returns:
            Similar documents list
        """
        return self.retrieve(document, top_k=top_k)

