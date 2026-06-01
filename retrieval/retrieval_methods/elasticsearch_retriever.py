import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticsearchRetriever:
    """
    Elasticsearch retriever
    
    Supports multiple retrieval methods:
    1. Full-text retrieval (BM25)
    2. Vector retrieval (if the document contains the vector field)
    3. Hybrid retrieval (combined full-text and vector)
    """
    
    def __init__(self, 
                 index_name: str = 'financial_corpus',
                 host: str = 'localhost',
                 port: int = 9200,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 use_ssl: bool = False,
                 verify_certs: bool = False,
                 embedding_model: Optional[Any] = None,
                 search_type: str = "hybrid"):
        """
        Initialize the Elasticsearch retriever
        
        Args:
            index_name: Index name
            host: Elasticsearch host address
            port: Elasticsearch port
            username: Username (optional)
            password: Password (optional)
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify the SSL certificate
            embedding_model: Embedding model for generating query vectors (optional)
                            Should have an encode() method that takes text and returns embeddings
            search_type: Search type ("text", "vector", "hybrid")
                        - "text": BM25 only
                        - "vector": Vector search only (requires embedding_model and vector field)
                        - "hybrid": Combined BM25 + vector (requires embedding_model and vector field)
        """
        self.index_name = index_name
        self.host = host
        self.port = port
        self.embedding_model = embedding_model
        self.search_type = search_type
        
        # Build the connection configuration
        es_config = {
            'hosts': [{'host': host, 'port': port}],
            'use_ssl': use_ssl,
            'verify_certs': verify_certs
        }
        
        if username and password:
            es_config['http_auth'] = (username, password)
        
        try:
            self.es = Elasticsearch(**es_config)
            # Test the connection
            if self.es.ping():
                logger.info(f"Successfully connected to Elasticsearch: {host}:{port}")
            else:
                logger.warning(f"Cannot connect to Elasticsearch: {host}:{port}")
        except ConnectionError as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
        
        # Check if the index exists
        self._check_index()
        
        # Detect the field mapping
        self._detect_field_mapping()
    
    def _check_index(self):
        """Check if the index exists, if not then create it"""
        try:
            if not self.es.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} does not exist, creating...")
                self._create_index()
            else:
                logger.info(f"Index {self.index_name} exists")
        except Exception as e:
            logger.error(f"Check index failed: {e}")
    
    def _create_index(self):
        """Create the index and mapping"""
        try:
            # Define the index mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "ik_max_word",  # Use IK tokenizer
                            "search_analyzer": "ik_smart"
                        },
                        "title": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart"
                        },
                        "file_path": {
                            "type": "keyword"
                        },
                        "file_type": {
                            "type": "keyword"
                        },
                        "created_at": {
                            "type": "date"
                        },
                        "vector": {
                            "type": "dense_vector",
                            "dims": 768,  # Default vector dimension
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "ik_max_word": {
                                "type": "ik_max_word"
                            },
                            "ik_smart": {
                                "type": "ik_smart"
                            }
                        }
                    }
                }
            }
            
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Index {self.index_name} created successfully")
            
        except Exception as e:
            logger.error(f"Create index failed: {e}")
            # 如果创建失败，尝试创建简单的索引
            try:
                simple_mapping = {
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "title": {"type": "text"},
                            "file_path": {"type": "keyword"}
                        }
                    }
                }
                self.es.indices.create(index=self.index_name, body=simple_mapping)
                logger.info(f"Simple index {self.index_name} created successfully")
            except Exception as e2:
                logger.error(f"Create simple index also failed: {e2}")
                raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Batch add documents to the index
        
        Args:
            documents: Document list, each document should contain content, title, file_path etc.
            
        Returns:
            bool: Whether successfully
        """
        try:
            for i, doc in enumerate(documents):
                # Prepare the document data
                doc_data = {
                    'file_path': doc.get('file_path', ''),
                    'file_type': doc.get('file_type', ''),
                    'created_at': doc.get('created_at', '')
                }
                
                # Use the detected content field name
                if 'content' in doc:
                    doc_data[self.content_field] = doc['content']
                elif 'text' in doc:
                    doc_data[self.content_field] = doc['text']
                
                # If the document contains the title, also add it
                if 'title' in doc:
                    doc_data['title'] = doc['title']
                
                # If the document contains the vector, also add the vector field
                if self.vector_field and 'vector' in doc and doc['vector'] is not None:
                    doc_data[self.vector_field] = doc['vector']
                elif self.vector_field and 'text_vector' in doc and doc['text_vector'] is not None:
                    doc_data[self.vector_field] = doc['text_vector']
                
                # Index the document
                self.es.index(index=self.index_name, body=doc_data, id=f"doc_{i}")
            
            # Refresh the index
            self.es.indices.refresh(index=self.index_name)
            logger.info(f"Successfully added {len(documents)} documents to the index")
            return True
            
        except Exception as e:
            logger.error(f"Add document failed: {e}")
            return False
    
    def search(self, 
               query: str, 
               top_k: int = 5, 
               search_type: str = "hybrid",
               **kwargs) -> List[Dict[str, Any]]:
        """
        Search documents
        
        Args:
            query: Query text
            top_k: Return the number of results
            search_type: Search type ("text", "vector", "hybrid")
            **kwargs: Other search parameters
            
        Returns:
            List[Dict]: Search results list
        """
        try:
            if search_type == "text":
                return self._text_search(query, top_k, **kwargs)
            elif search_type == "vector":
                return self._vector_search(query, top_k, **kwargs)
            elif search_type == "hybrid":
                return self._hybrid_search(query, top_k, **kwargs)
            else:
                logger.warning(f"Unsupported search type: {search_type}, using hybrid search")
                return self._hybrid_search(query, top_k, **kwargs)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Tuple[str, float, str]]:
        """
        Retrieve documents (compatible interface)
        
        Args:
            query: Query text
            top_k: Return the number of results
            **kwargs: Other search parameters (can override search_type)
            
        Returns:
            List[Tuple[str, float, str]]: Retrieve results list, each element contains (document content, similarity score, document ID)
        """
        try:
            # Use the search_type from initialization or override from kwargs
            search_type = kwargs.pop('search_type', self.search_type)
            
            # If vector search is needed, generate query_vector if not provided
            if search_type in ["vector", "hybrid"] and 'query_vector' not in kwargs:
                if self.embedding_model:
                    try:
                        # Generate query vector using the embedding model
                        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
                        if isinstance(query_embedding, np.ndarray):
                            kwargs['query_vector'] = query_embedding.tolist()
                        else:
                            kwargs['query_vector'] = query_embedding
                        logger.info(f"Generated query vector using embedding model (dimension: {len(kwargs['query_vector'])})")
                    except Exception as e:
                        logger.warning(f"Failed to generate query vector: {e}. Falling back to text search.")
                        search_type = "text"
                elif not self.vector_field:
                    logger.warning("No embedding model provided and no vector field detected. Using text search only.")
                    search_type = "text"
                else:
                    logger.warning("No embedding model provided for vector search. Using text search only.")
                    search_type = "text"
            
            # Use the specified search type
            results = self.search(query, top_k, search_type=search_type, **kwargs)
            
            # Convert to compatible format
            formatted_results = []
            for result in results:
                content = result.get('content', '')
                score = result.get('score', 0.0)
                # Use _id as doc_id (Elasticsearch document ID), fallback to file_path or chunk_id
                doc_id = result.get('_id') or result.get('chunk_id') or result.get('file_path', 'unknown')
                
                formatted_results.append((content, score, doc_id))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Retrieve failed: {e}")
            return []
    
    def _text_search(self, query: str, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """Full-text search"""
        try:
            # Get index mapping to check available fields
            try:
                mapping = self.es.indices.get_mapping(index=self.index_name)
                properties = mapping.get(self.index_name, {}).get('mappings', {}).get('properties', {})
                has_title = 'title' in properties
            except:
                has_title = False
            
            # Build the query - ensure we use the correct field name
            # Priority: use detected content_field, but ensure it exists
            fields = [f"{self.content_field}^2"]
            if has_title:
                fields.append("title^3")
            
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "tie_breaker": 0.5  # Add tie_breaker like BM25Retriever
                    }
                },
                "size": top_k,
                "_source": [self.content_field, "title", "file_path", "file_type", "created_at"]
            }
            
            # Execute the search
            response = self.es.search(index=self.index_name, body=search_body)
            
            # Process the results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    '_id': hit.get('_id', ''),  # Elasticsearch document ID
                    'content': hit['_source'].get(self.content_field, ''),
                    'title': hit['_source'].get('title', ''),
                    'file_path': hit['_source'].get('file_path', ''),
                    'file_type': hit['_source'].get('file_type', ''),
                    'chunk_id': hit['_source'].get('chunk_id', ''),  # Support chunk_id field
                    'score': hit['_score']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            return []
    
    def _vector_search(self, query: str, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """Vector search"""
        try:
            # Check if there is a vector field
            if not self.vector_field:
                logger.warning("The index does not have a vector field, falling back to full-text search")
                return self._text_search(query, top_k, **kwargs)
            
            # Get the query vector (here we need to provide or use the model to generate it)
            query_vector = kwargs.get('query_vector')
            if query_vector is None:
                logger.warning("Vector search needs to provide the query_vector parameter, falling back to full-text search")
                return self._text_search(query, top_k, **kwargs)
            
            # Build the vector search query
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{self.vector_field}') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                },
                "size": top_k,
                "_source": [self.content_field, "title", "file_path", "file_type", "created_at"]
            }
            
            # Execute the search
            response = self.es.search(index=self.index_name, body=search_body)
            
            # Process the results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    '_id': hit.get('_id', ''),  # Elasticsearch document ID
                    'content': hit['_source'].get(self.content_field, ''),
                    'title': hit['_source'].get('title', ''),
                    'file_path': hit['_source'].get('file_path', ''),
                    'file_type': hit['_source'].get('file_type', ''),
                    'chunk_id': hit['_source'].get('chunk_id', ''),  # Support chunk_id field
                    'score': hit['_score']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _hybrid_search(self, query: str, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """Hybrid search (combined full-text and vector)"""
        try:
            # Get the query vector and weights
            query_vector = kwargs.get('query_vector')
            bm25_weight = kwargs.get('bm25_weight', 0.5)  # Default 50% BM25, 50% vector
            vector_weight = kwargs.get('vector_weight', 0.5)
            
            # Normalize weights
            total_weight = bm25_weight + vector_weight
            if total_weight > 0:
                bm25_weight = bm25_weight / total_weight
                vector_weight = vector_weight / total_weight
            
            # Build the hybrid query using function_score for better control
            if self.vector_field and query_vector:
                # Use function_score to combine BM25 and vector search with weights
                # This ensures both scores are normalized and combined properly
                search_body = {
                    "query": {
                        "function_score": {
                            "query": {
                                "multi_match": {
                                    "query": query,
                                    "fields": [f"{self.content_field}^2", "title^3"] if 'title' in self.es.indices.get_mapping(index=self.index_name).get(self.index_name, {}).get('mappings', {}).get('properties', {}) else [f"{self.content_field}^2"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            },
                            "functions": [
                                {
                                    "script_score": {
                                        "script": {
                                            # Normalize vector score to similar range as BM25
                                            # BM25 scores are typically 1-100+, vector scores are 0-2
                                            # We scale vector score to match BM25 range, then combine with weights
                                            "source": f"_score * {bm25_weight} + (cosineSimilarity(params.query_vector, '{self.vector_field}') + 1.0) * 50.0 * {vector_weight}",
                                            "params": {"query_vector": query_vector}
                                        }
                                    }
                                }
                            ],
                            "score_mode": "sum",
                            "boost_mode": "replace"
                        }
                    },
                    "size": top_k,
                    "_source": [self.content_field, "title", "file_path", "file_type", "created_at"]
                }
                
                # Execute the search
                response = self.es.search(index=self.index_name, body=search_body)
                
                # Process the results
            else:
                # Fallback to BM25 only if no vector
                logger.warning("No vector field or query_vector, using BM25 only")
                return self._text_search(query, top_k, **kwargs)
            
            # Process the results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    '_id': hit.get('_id', ''),  # Elasticsearch document ID
                    'content': hit['_source'].get(self.content_field, ''),
                    'title': hit['_source'].get('title', ''),
                    'file_path': hit['_source'].get('file_path', ''),
                    'file_type': hit['_source'].get('file_type', ''),
                    'chunk_id': hit['_source'].get('chunk_id', ''),  # Support chunk_id field
                    'score': hit['_score']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def delete_documents(self, file_paths: List[str]) -> bool:
        """
        Delete documents with specified file paths
        
        Args:
            file_paths: The list of file paths to delete
            
        Returns:
            bool: Whether successfully
        """
        try:
            for file_path in file_paths:
                # Build the delete query
                delete_body = {
                    "query": {
                        "term": {
                            "file_path": file_path
                        }
                    }
                }
                
                # Execute the delete
                self.es.delete_by_query(index=self.index_name, body=delete_body)
            
            # Refresh the index
            self.es.indices.refresh(index=self.index_name)
            logger.info(f"Successfully deleted {len(file_paths)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Delete documents failed: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get the index statistics"""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            return {
                'document_count': stats['indices'][self.index_name]['total']['docs']['count'],
                'index_size': stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                'index_name': self.index_name
            }
        except Exception as e:
            logger.error(f"Get the index statistics failed: {e}")
            return {}
    
    def close(self):
        """Close the connection"""
        try:
            if hasattr(self, 'es'):
                self.es.close()
                logger.info("Elasticsearch connection closed")
        except Exception as e:
            logger.error(f"Close the connection failed: {e}")
    
    def __del__(self):
        """Destructor, ensure the connection is closed"""
        self.close() 

    def _detect_field_mapping(self):
        """Detect the field mapping in the index"""
        try:
            # Get the index mapping
            mapping = self.es.indices.get_mapping(index=self.index_name)
            properties = mapping[self.index_name]['mappings']['properties']
            
            # Detect the content field
            if 'content' in properties:
                self.content_field = 'content'
            elif 'text' in properties:
                self.content_field = 'text'
            elif 'body' in properties:
                self.content_field = 'body'
            else:
                # Find the field containing text content
                for field_name, field_info in properties.items():
                    if field_info.get('type') == 'text':
                        self.content_field = field_name
                        break
                else:
                    # If no text type field is found, use the first field
                    self.content_field = list(properties.keys())[0] if properties else 'content'
            
            # Detect the vector field
            if 'vector' in properties:
                self.vector_field = 'vector'
            elif 'embedding' in properties:
                self.vector_field = 'embedding'
            elif 'vec' in properties:
                self.vector_field = 'vec'
            else:
                # Find the field containing the vector
                for field_name, field_info in properties.items():
                    if field_info.get('type') == 'dense_vector':
                        self.vector_field = field_name
                        break
                else:
                    self.vector_field = None
            
            logger.info(f"Detected the content field: {self.content_field}")
            if self.vector_field:
                logger.info(f"Detected the vector field: {self.vector_field}")
            else:
                logger.info("No vector field detected")
                
        except Exception as e:
            logger.warning(f"Field mapping detection failed: {e}")
            # Use the default value
            self.content_field = 'content'
            self.vector_field = None 