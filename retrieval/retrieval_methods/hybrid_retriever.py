#!/usr/bin/env python3
"""
Hybrid retrieval method implementation
Combine BM25, semantic retrieval, vector retrieval, etc., to provide more comprehensive retrieval results
Support Elasticsearch integration and local vector retrieval
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from retrieval_methods.bm25_retriever import BM25Retriever
from retrieval_methods.semantic_retriever import SemanticRetriever
from retrieval_methods.vector_retriever import VectorRetriever
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, 
                 semantic_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 index_name: str = "financial_corpus",
                 host: str = "localhost", 
                 port: int = 9200,
                 use_elasticsearch: bool = True,
                 corpus_dir: str = None,
                 corpus_path: str = "/home/yidong/DRAGIN/corpus.jsonl",
                 use_vector_retriever: bool = True):
        """
        Initialize the hybrid retriever
        
        Args:
            semantic_model_name: Semantic retrieval model name
            index_name: Elasticsearch index name
            host: Elasticsearch host address
            port: Elasticsearch port
            use_elasticsearch: Whether to use Elasticsearch
            corpus_dir: Corpus directory path (only when Elasticsearch is not used)
            corpus_path: Corpus file path for vector retrieval
            use_vector_retriever: Whether to use the vector retriever
        """
        self.use_elasticsearch = use_elasticsearch
        self.use_vector_retriever = use_vector_retriever
        
        if use_elasticsearch:
            # Use Elasticsearch mode
            print("Initializing Elasticsearch hybrid retriever...")
            
            print("Initializing BM25 retriever...")
            self.bm25_retriever = BM25Retriever(index_name, host, port)
            
            print("Initializing semantic retriever...")
            self.semantic_retriever = SemanticRetriever(
                semantic_model_name, 
                index_name, 
                host, 
                port, 
                use_elasticsearch=True
            )
        else:
            # Use local mode
            if not corpus_dir:
                raise ValueError("Local mode needs to provide the corpus_dir parameter")
                
            print("Initializing local hybrid retriever...")
            print("Initializing BM25 retriever...")
            self.bm25_retriever = BM25Retriever(corpus_dir)
            
            print("Initializing semantic retriever...")
            self.semantic_retriever = SemanticRetriever(
                semantic_model_name, 
                use_elasticsearch=False
            )
            self.semantic_retriever.load_corpus(corpus_dir)
        
        # Initialize the vector retriever
        if use_vector_retriever:
            print("Initializing vector retriever...")
            self.vector_retriever = VectorRetriever(corpus_path=corpus_path)
        else:
            self.vector_retriever = None
        
        # Weight configuration
        self.bm25_weight = 0.3
        self.semantic_weight = 0.4
        self.vector_weight = 0.3
        
        print("Hybrid retriever initialized!")
    
    def set_weights(self, bm25_weight: float, semantic_weight: float, vector_weight: float = None):
        """
        Set the retriever weights
        
        Args:
            bm25_weight: BM25 retriever weight
            semantic_weight: Semantic retriever weight
            vector_weight: Vector retriever weight (if None, automatically calculated)
        """
        if vector_weight is None:
            vector_weight = 1.0 - bm25_weight - semantic_weight
        
        if abs(bm25_weight + semantic_weight + vector_weight - 1.0) > 1e-6:
            raise ValueError("The sum of weights must be equal to 1.0")
        
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.vector_weight = vector_weight
        logger.info(f"Weights set: BM25={bm25_weight:.2f}, Semantic={semantic_weight:.2f}, Vector={vector_weight:.2f}")
    
    def retrieve(self, query: str, top_k: int = 5, use_reranking: bool = True) -> List[Tuple[str, float, str]]:
        """
        Hybrid retrieval related documents
        
        Args:
            query: Query text
            top_k: Return the top-k results
            use_reranking: Whether to use reranking
            
        Returns:
            Retrieval results list, each element contains (document content, combined score, document ID)
        """
        logger.info(f"Starting hybrid retrieval: {query}")
        
        # 1. BM25 retrieval
        logger.info("Executing BM25 retrieval...")
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        
        # 2. Semantic retrieval
        logger.info("Executing semantic retrieval...")
        if use_reranking:
            semantic_raw_results = self.semantic_retriever.retrieve_with_reranking(query, top_k=top_k * 2)
        else:
            semantic_raw_results = self.semantic_retriever.retrieve(query, top_k=top_k * 2)
        
        # Ensure semantic_results is in Tuple format
        semantic_results = []
        for result in semantic_raw_results:
            if isinstance(result, tuple) and len(result) == 3:
                semantic_results.append(result)
            elif isinstance(result, dict):
                # Convert dict to tuple format
                content = result.get('content', result.get('text', ''))
                score = result.get('score', 0.0)
                doc_id = result.get('id', result.get('doc_id', result.get('_id', '')))
                semantic_results.append((content, score, doc_id))
            else:
                logger.warning(f"Unexpected semantic result format: {type(result)}, skipping")
        
        # 3. Vector retrieval
        vector_results = []
        if self.vector_retriever:
            logger.info("Executing vector retrieval...")
            vector_raw_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
            # Convert vector results from Dict format to Tuple format
            vector_results = [
                (result.get('content', ''), result.get('score', 0.0), result.get('id', ''))
                for result in vector_raw_results
                if isinstance(result, dict)
            ]
        
        # 4. Result fusion
        logger.info("Fusing retrieval results...")
        fused_results = self._fuse_results(bm25_results, semantic_results, vector_results, top_k)
        
        return fused_results
    
    def _fuse_results(self, bm25_results: List[Tuple[str, float, str]], 
                     semantic_results: List[Tuple[str, float, str]], 
                     vector_results: List[Tuple[str, float, str]],
                     top_k: int) -> List[Tuple[str, float, str]]:
        """
        Fuse BM25, semantic retrieval, and vector retrieval results
        
        Args:
            bm25_results: BM25 retrieval results
            semantic_results: Semantic retrieval results
            vector_results: Vector retrieval results
            top_k: The final returned top-k results
            
        Returns:
            Fused results list
        """
        # Create a mapping from document ID to score
        doc_scores = {}
        
        # Process BM25 results
        for result in bm25_results:
            try:
                if isinstance(result, tuple) and len(result) == 3:
                    content, score, doc_id = result
                elif isinstance(result, dict):
                    content = result.get('content', '')
                    score = result.get('score', 0.0)
                    doc_id = result.get('id', result.get('doc_id', ''))
                else:
                    logger.warning(f"Unexpected BM25 result format: {type(result)}, skipping")
                    continue
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'content': content,
                        'bm25_score': score,
                        'semantic_score': 0.0,
                        'vector_score': 0.0,
                        'combined_score': 0.0
                    }
                else:
                    doc_scores[doc_id]['bm25_score'] = max(doc_scores[doc_id]['bm25_score'], score)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing BM25 result: {e}, result: {result}")
                continue
        
        # Process semantic retrieval results
        for result in semantic_results:
            try:
                if isinstance(result, tuple) and len(result) == 3:
                    content, score, doc_id = result
                elif isinstance(result, dict):
                    content = result.get('content', '')
                    score = result.get('score', 0.0)
                    doc_id = result.get('id', result.get('doc_id', ''))
                else:
                    logger.warning(f"Unexpected semantic result format: {type(result)}, skipping")
                    continue
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'content': content,
                        'bm25_score': 0.0,
                        'semantic_score': score,
                        'vector_score': 0.0,
                        'combined_score': 0.0
                    }
                else:
                    doc_scores[doc_id]['semantic_score'] = max(doc_scores[doc_id]['semantic_score'], score)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing semantic result: {e}, result: {result}")
                continue
        
        # Process vector retrieval results
        for result in vector_results:
            try:
                if isinstance(result, tuple) and len(result) == 3:
                    content, score, doc_id = result
                elif isinstance(result, dict):
                    content = result.get('content', '')
                    score = result.get('score', 0.0)
                    doc_id = result.get('id', result.get('doc_id', ''))
                else:
                    logger.warning(f"Unexpected vector result format: {type(result)}, skipping")
                    continue
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'content': content,
                        'bm25_score': 0.0,
                        'semantic_score': 0.0,
                        'vector_score': score,
                        'combined_score': 0.0
                    }
                else:
                    doc_scores[doc_id]['vector_score'] = max(doc_scores[doc_id]['vector_score'], score)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing vector result: {e}, result: {result}")
                continue
        
        # Calculate the combined score
        for doc_id, scores in doc_scores.items():
            # Normalize the score
            normalized_bm25 = self._normalize_score(scores['bm25_score'], bm25_results)
            normalized_semantic = self._normalize_score(scores['semantic_score'], semantic_results)
            normalized_vector = self._normalize_score(scores['vector_score'], vector_results) if vector_results else 0.0
            
            # Weighted combination
            scores['combined_score'] = (
                self.bm25_weight * normalized_bm25 + 
                self.semantic_weight * normalized_semantic +
                self.vector_weight * normalized_vector
            )
        
        # Sort by combined score
        sorted_results = sorted(
            doc_scores.items(), 
            key=lambda x: x[1]['combined_score'], 
            reverse=True
        )
        
        # Return the top-k results
        final_results = []
        for doc_id, scores in sorted_results[:top_k]:
            final_results.append((
                scores['content'],
                scores['combined_score'],
                doc_id
            ))
        
        return final_results
    
    def _normalize_score(self, score: float, results: List[Tuple[str, float, str]]) -> float:
        """
        Normalize the score
        
        Args:
            score: Original score
            results: Result list
            
        Returns:
            Normalized score
        """
        if not results:
            return 0.0
        
        scores = [r[1] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return 0.5
        
        return (score - min_score) / (max_score - min_score)
    
    def retrieve_with_feedback(self, query: str, feedback_docs: List[str], top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Retrieval based on user feedback
        
        Args:
            query: Query text
            feedback_docs: User feedback related document list
            top_k: Return the top-k results
            
        Returns:
            Retrieval results list
        """
        logger.info(f"Retrieval based on feedback: {query}")
        
        # Use feedback documents for query expansion
        expanded_query = self._expand_query(query, feedback_docs)
        
        # Execute hybrid retrieval
        results = self.retrieve(expanded_query, top_k=top_k)
        
        return results
    
    def _expand_query(self, query: str, feedback_docs: List[str]) -> str:
        """
        Query expansion based on feedback documents
        
        Args:
            query: Original query
            feedback_docs: Feedback document list
            
        Returns:
            Expanded query
        """
        # Simple query expansion: add keywords from feedback documents
        # Here you can implement more complex query expansion strategies
        
        # Extract keywords from feedback documents (simple word frequency statistics)
        word_freq = {}
        for doc in feedback_docs:
            # Simple tokenization (split by space)
            words = doc.lower().split()
            for word in words:
                if len(word) > 2:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Select top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords = [word for word, freq in top_keywords]
        
        # Build expanded query
        expanded_query = query + " " + " ".join(top_keywords)
        
        return expanded_query
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """
        Get retrieval statistics
        
        Args:
            query: Query text
            
        Returns:
            Retrieval statistics
        """
        # Execute various retrievals
        bm25_results = self.bm25_retriever.retrieve(query, top_k=10)
        semantic_results = self.semantic_retriever.retrieve(query, top_k=10)
        
        # Calculate overlap
        bm25_doc_ids = set(r[2] for r in bm25_results)
        semantic_doc_ids = set(r[2] for r in semantic_results)
        overlap = len(bm25_doc_ids.intersection(semantic_doc_ids))
        
        stats = {
            "query": query,
            "bm25_results_count": len(bm25_results),
            "semantic_results_count": len(semantic_results),
            "overlap_count": overlap,
            "overlap_ratio": overlap / len(bm25_doc_ids.union(semantic_doc_ids)) if bm25_doc_ids.union(semantic_doc_ids) else 0,
            "bm25_top_score": bm25_results[0][1] if bm25_results else 0,
            "semantic_top_score": semantic_results[0][1] if semantic_results else 0,
            "weights": {
                "bm25": self.bm25_weight,
                "semantic": self.semantic_weight,
                "vector": self.vector_weight
            }
        }
        
        # Add vector retrieval statistics (if enabled)
        if self.vector_retriever:
            vector_results = self.vector_retriever.retrieve(query, top_k=10)
            stats["vector_results_count"] = len(vector_results)
            stats["vector_top_score"] = vector_results[0][1] if vector_results else 0
            
            # Calculate overlap with vector retrieval
            vector_doc_ids = set(r[2] for r in vector_results)
            all_doc_ids = bm25_doc_ids.union(semantic_doc_ids).union(vector_doc_ids)
            stats["total_unique_docs"] = len(all_doc_ids)
        
        return stats
    
    def add_document(self, content: str, doc_id: str = None):
        """
        Add new document
        
        Args:
            content: Document content
            doc_id: Document ID
        """
        # Add to all retrievers
        self.bm25_retriever.add_document(content, doc_id)
        self.semantic_retriever.add_document(content, doc_id)
        if self.vector_retriever:
            self.vector_retriever.add_document(content, doc_id)
    
    def add_documents_batch(self, documents: List[Dict[str, str]]):
        """
        Batch add documents
        
        Args:
            documents: Document list, each document contains content and optional id field
        """
        if self.use_elasticsearch:
            # Batch add to Elasticsearch retriever
            self.bm25_retriever.add_documents_batch(documents)
            self.semantic_retriever.add_documents_batch(documents)
        else:
            # Add one by one to local retriever
            for doc in documents:
                content = doc.get('content', doc.get('text', ''))
                doc_id = doc.get('id')
                self.bm25_retriever.add_document(content, doc_id)
                self.semantic_retriever.add_document(content, doc_id)
        
        # Add to vector retriever
        if self.vector_retriever:
            self.vector_retriever.add_documents_batch(documents)
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get corpus statistics
        
        Returns:
            Corpus statistics
        """
        bm25_stats = self.bm25_retriever.get_corpus_stats()
        semantic_stats = self.semantic_retriever.get_corpus_stats()
        
        stats = {
            "bm25_stats": bm25_stats,
            "semantic_stats": semantic_stats,
            "weights": {
                "bm25": self.bm25_weight,
                "semantic": self.semantic_weight,
                "vector": self.vector_weight
            },
            "retrieval_mode": "Elasticsearch" if self.use_elasticsearch else "Local"
        }
        
        # Add vector retriever statistics
        if self.vector_retriever:
            vector_stats = self.vector_retriever.get_corpus_stats()
            stats["vector_stats"] = vector_stats
        
        return stats
    
    def search_by_title(self, title_query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """
        Search documents based on title (using vector retriever)
        
        Args:
            title_query: Title query
            top_k: Return the number of results
            
        Returns:
            List[Tuple[str, float, str]]: Search results list
        """
        if not self.vector_retriever:
            logger.warning("Vector retriever is not enabled, cannot execute title search")
            return []
        
        return self.vector_retriever.search_by_title(title_query, top_k=top_k)
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Get document information by document ID (using vector retriever)
        
        Args:
            doc_id: Document ID
            
        Returns:
            Optional[Dict]: Document information, if not exists, return None
        """
        if not self.vector_retriever:
            logger.warning("Vector retriever is not enabled, cannot get document information")
            return None
        
        return self.vector_retriever.get_document_by_id(doc_id)
