#!/usr/bin/env python3
"""
BM25 Retriever
Using Elasticsearch's BM25 algorithm for document retrieval, finding the most relevant document fragments for a question
"""

import json
import os
from typing import List, Dict, Any, Tuple
from elasticsearch import Elasticsearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, index_name: str = "financial_corpus", host: str = "localhost", port: int = 9200):
        """
        Initialize the BM25 Retriever
        
        Args:
            index_name: Elasticsearch index name
            host: Elasticsearch host address
            port: Elasticsearch port
        """
        self.index_name = index_name
        self.es = Elasticsearch([{'host': host, 'port': port}])
        
        # Check Elasticsearch connection
        if not self.es.ping():
            raise ConnectionError(f"无法连接到Elasticsearch {host}:{port}")
        
        # Check if the index exists
        if not self.es.indices.exists(index=index_name):
            raise ValueError(f"Index {index_name} does not exist, please run setup_elasticsearch.py to create the index")
        
        logger.info(f"Successfully connected to Elasticsearch index: {index_name}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Use Elasticsearch BM25 to retrieve relevant documents
        
        Args:
            query: Query text
            top_k: Return the top-k results
            
        Returns:
            Retrieval results list, each element contains (document content, similarity score, document ID)
        """
        try:
            # Build the search query
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
            
            # Execute the search
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
            
            logger.info(f"Retrieved {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
        
    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[str, float, str]]]:
        """
        Batch retrieve multiple queries
        
        Args:
            queries: Query text list
            top_k: Return the top-k results for each query
            
        Returns:
            Retrieval results list for each query
        """
        try:
            # Build the batch search request
            request = []
            for query in queries:
                req_head = {"index": self.index_name, "search_type": "dfs_query_then_fetch"}
                req_body = {
                    "_source": True,
                    "query": {
                        "multi_match": {
                            "query": query,
                            "type": "best_fields",
                            "fields": ["text"],
                            "tie_breaker": 0.5
                        }
                    },
                    "size": top_k
                }
                request.extend([req_head, req_body])
        
            # Execute the batch search
            response = self.es.msearch(body=request)
            
            all_results = []
            for resp in response["responses"]:
                results = []
                if 'hits' in resp and 'hits' in resp['hits']:
                    for hit in resp['hits']['hits']:
                        doc_id = hit['_id']
                        score = hit['_score']
                        content = hit['_source']['text']
                        results.append((content, score, doc_id))
                all_results.append(results)
            
            logger.info(f"Batch retrieval completed, processed {len(queries)} queries")
            return all_results
            
        except Exception as e:
            logger.error(f"Error during batch retrieval: {e}")
            return [[] for _ in queries]
    
    def add_document(self, content: str, doc_id: str = None):
        """
        Add a new document to the Elasticsearch index
        
        Args:
            content: Document content
            doc_id: Document ID
        """
        try:
            document = {"text": content}
            
            if doc_id:
                # Use the specified ID
                self.es.index(
                    index=self.index_name,
                    id=doc_id,
                    body=document
                )
            else:
                # Automatically generate the ID
                self.es.index(
                    index=self.index_name,
                    body=document
                )
            
            # Refresh the index
            self.es.indices.refresh(index=self.index_name)
            logger.info(f"Successfully added document to the index")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
    
    def add_documents_batch(self, documents: List[Dict[str, str]]):
        """
        Batch add documents to the Elasticsearch index
        
        Args:
            documents: Document list, each document contains content and optional id field
        """
        try:
            bulk_data = []
            for doc in documents:
                content = doc.get('content', doc.get('text', ''))
                doc_id = doc.get('id')
                
                if doc_id:
                    # Use the specified ID
                    bulk_data.append({
                        "index": {
                            "_index": self.index_name,
                            "_id": doc_id
                        }
                    })
                else:
                    # Automatically generate the ID
                    bulk_data.append({
                        "index": {
                            "_index": self.index_name
                        }
                    })
                
                bulk_data.append({"text": content})
            
            # Execute the batch operation
            if bulk_data:
                self.es.bulk(body=bulk_data)
                # Refresh the index
                self.es.indices.refresh(index=self.index_name)
                logger.info(f"Successfully added {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def delete_document(self, doc_id: str):
        """
        Delete a document from the index
        
        Args:
            doc_id: The ID of the document to delete
        """
        try:
            self.es.delete(index=self.index_name, id=doc_id)
            self.es.indices.refresh(index=self.index_name)
            logger.info(f"Successfully deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get the index statistics
        
        Returns:
            Index statistics
        """
        try:
            # Get the index statistics
            stats = self.es.indices.stats(index=self.index_name)
            index_stats = stats['indices'][self.index_name]
            
            # Get the total number of documents
            count_response = self.es.count(index=self.index_name)
            total_docs = count_response['count']
            
            return {
                    "index_name": self.index_name,
                    "total_documents": total_docs,
                    "index_size_bytes": index_stats['total']['store']['size_in_bytes'],
                    "index_size_mb": round(index_stats['total']['store']['size_in_bytes'] / (1024 * 1024), 2),
                    "segments_count": index_stats['total']['segments']['count']
                }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def search_with_filters(self, query: str, filters: Dict[str, Any] = None, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search with filters
        
        Args:
            query: Query text
            filters: Filter conditions dictionary
            top_k: Return the top-k results
            
        Returns:
            Retrieval results list
        """
        try:
            # Build the query
            search_body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": query,
                                "type": "best_fields",
                                "fields": ["text"],
                                "tie_breaker": 0.5
                            }
                        }
                    }
                },
                "size": top_k,
                "_source": ["text"]
            }
            
            # Add the filters
            if filters:
                search_body["query"]["bool"]["filter"] = filters
            
            # Execute the search
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
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search with filters: {e}")
            return []
