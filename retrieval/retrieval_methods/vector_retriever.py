#!/usr/bin/env python3
"""
向量检索方法实现
基于FAISS向量库和sentence-transformers进行语义相似度检索
支持从corpus.jsonl文件加载语料库
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple, Any
import logging
import time
from tqdm import tqdm
import json
import os
import faiss
import pickle
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    Vector retriever based on FAISS
    Support loading corpus from corpus.jsonl file
    Support various sentence-transformers models for calculating embeddings
    Default model can be specified via model_name parameter
    """
    
    def __init__(self, 
                 corpus_path: str = "/home/yidong/DRAGIN/enhanced_corpus.jsonl",
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 faiss_index_path: str = None,
                 metadata_path: str = None,
                 batch_size: int = 32,
                 max_length: int = 4096,
                 use_gpu: bool = True):
        """
        Initialize vector retriever
        
        Args:
            corpus_path: Corpus file path (JSONL format)
            model_name: sentence-transformers model name
            faiss_index_path: FAISS index file path (if None, saved in current working directory)
            metadata_path: Metadata file path (if None, saved in current working directory)
            batch_size: Batch size
            max_length: Maximum text length
            use_gpu: Whether to use GPU
        """
        self.corpus_path = corpus_path
        self.model_name = model_name
        
        # Automatically generate file path based on model name, saved in current working directory
        # Create a safe filename from model name (replace / with _)
        model_name_safe = model_name.replace('/', '_').replace('-', '_')
        
        if faiss_index_path is None:
            self.faiss_index_path = f"vector_embeddings_{model_name_safe}.faiss"
        else:
            self.faiss_index_path = faiss_index_path
            
        if metadata_path is None:
            self.metadata_path = f"vector_metadata_{model_name_safe}.pkl"
        else:
            self.metadata_path = metadata_path
        
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Check if the corpus file exists
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        # Print file path information
        logger.info(f"Corpus file: {corpus_path}")
        logger.info(f"FAISS index file: {os.path.abspath(self.faiss_index_path)}")
        logger.info(f"Metadata file: {os.path.abspath(self.metadata_path)}")
        
        # Initialize model
        self.model = self._load_model()
        
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize FAISS index and metadata
        self.faiss_index = None
        self.document_metadata = {}
        self.doc_id_to_index = {}  # doc_id to FAISS index mapping
        self.index_to_doc_id = {}  # FAISS index to doc_id mapping
        
        # Load or create FAISS index
        self._load_or_create_faiss_index()
        
        # If the index is empty, build index from corpus
        if self.faiss_index.ntotal == 0:
            logger.info("FAISS index is empty, building index from corpus...")
            build_result = self.build_index_from_corpus()
            if build_result["status"] == "success":
                logger.info(f"Index built successfully, saved to: {os.path.abspath(self.faiss_index_path)}")
                logger.info(f"Metadata saved to: {os.path.abspath(self.metadata_path)}")
            else:
                logger.error(f"Index building failed: {build_result.get('reason', 'Unknown error')}")
    
    def _load_model(self) -> SentenceTransformer:
        """Load sentence-transformers model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_or_create_faiss_index(self):
        """Load or create FAISS index"""
        try:
            if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
                # Load existing FAISS index and metadata
                logger.info("Loading existing FAISS index and metadata...")
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                
                with open(self.metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                
                # Rebuild mapping
                for doc_id, metadata in self.document_metadata.items():
                    index = metadata['faiss_index']
                    self.doc_id_to_index[doc_id] = index
                    self.index_to_doc_id[index] = doc_id
                
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                # Create new FAISS index
                self._create_new_faiss_index()
                
        except Exception as e:
            logger.error(f"Failed to load/create FAISS index: {e}")
            self._create_new_faiss_index()
    
    def _create_new_faiss_index(self):
        """Create new FAISS index"""
        # 获取embedding维度
        sample_text = "Sample text for dimension calculation"
        sample_embedding = self.model.encode([sample_text], convert_to_numpy=True)
        embedding_dim = sample_embedding.shape[1]
        
        # Create FAISS index (using inner product index, suitable for cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        
        # If using GPU
        if self.device.type == "cuda":
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS: {e}")
        
        self.document_metadata = {}
        self.doc_id_to_index = {}
        self.index_to_doc_id = {}
        
        logger.info(f"Created new FAISS index with dimension {embedding_dim}")
    
    def load_corpus_from_jsonl(self) -> List[Dict]:
        """
        Load corpus from JSONL file or JSON array file
        
        Returns:
            List[Dict]: Document list
        """
        documents = []
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Try to parse as JSON array first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # It's a JSON array
                    logger.info(f"Detected JSON array format, found {len(data)} documents")
                    documents = data
                else:
                    # Single JSON object, wrap in list
                    documents = [data]
            except json.JSONDecodeError:
                # If not JSON array, try JSONL format (one JSON object per line)
                logger.info("Trying JSONL format (one JSON object per line)")
                for line_num, line in enumerate(content.split('\n'), 1):
                    try:
                        line = line.strip()
                        if line:
                            doc = json.loads(line)
                            documents.append(doc)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue
            
            # Normalize document format: ensure _id field exists
            # If chunk_id exists but _id doesn't, use chunk_id as _id
            normalized_documents = []
            for doc in documents:
                normalized_doc = doc.copy()
                # If _id doesn't exist, try to use chunk_id or generate one
                if '_id' not in normalized_doc:
                    if 'chunk_id' in normalized_doc:
                        normalized_doc['_id'] = normalized_doc['chunk_id']
                    else:
                        # Generate a unique ID
                        normalized_doc['_id'] = f"doc_{len(normalized_documents)}"
                normalized_documents.append(normalized_doc)
            
            logger.info(f"Successfully loaded {len(normalized_documents)} documents from {self.corpus_path}")
            return normalized_documents
            
        except Exception as e:
            logger.error(f"Failed to load corpus from {self.corpus_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def prepare_text_for_embedding(self, document: Dict) -> str:
        """
        Prepare text for embedding calculation
        
        Args:
            document: Document information dictionary
            
        Returns:
            str: Processed text
        """
        title = document.get('title', '')
        text = document.get('text', '')
        
        # Combine title and text
        if title and text:
            combined_text = f"{title}\n\n{text}"
        elif title:
            combined_text = title
        else:
            combined_text = text
        
        # Clean text (remove extra whitespace)
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
                    
        return combined_text
    
    def calculate_embeddings_batch(self, documents: List[Dict]) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Batch calculate document embeddings
        
        Args:
            documents: Document information list
            
        Returns:
            List[Tuple[str, np.ndarray, Dict]]: (doc_id, embedding, metadata)元组列表
        """
        if not documents:
            return []
        
        # Prepare text
        logger.info(f"Preparing texts for {len(documents)} documents...")
        texts = [self.prepare_text_for_embedding(doc) for doc in tqdm(documents, desc="Preparing texts")]
        doc_ids = [doc['_id'] for doc in documents]
        
        try:
            # Calculate embeddings
            logger.info(f"Calculating embeddings for {len(texts)} documents...")
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Prepare metadata - keep full metadata information
            results = []
            for i, doc_id in enumerate(tqdm(doc_ids, desc="Preparing metadata")):
                # Keep all fields of the original document, including metadata
                metadata = {
                    'doc_id': doc_id,
                    'title': documents[i].get('title', ''),
                    'text': documents[i].get('text', ''),
                    # Keep full metadata information
                    'metadata': documents[i].get('metadata', {})
                }
                results.append((doc_id, embeddings[i], metadata))
            
            logger.info(f"Successfully calculated embeddings for {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate embeddings: {e}")
            return []
    
    def add_embeddings_to_faiss(self, embeddings_data: List[Tuple[str, np.ndarray, Dict]]) -> int:
        """
        Add embeddings to FAISS index
        
        Args:
            embeddings_data: (doc_id, embedding, metadata) tuple list
            
        Returns:
            int: Number of successfully added documents
        """
        if not embeddings_data:
            return 0
        
        added_count = 0
        for doc_id, embedding, metadata in tqdm(embeddings_data, desc="Adding to FAISS index"):
            try:
                # Add embedding to FAISS index
                embedding_reshaped = embedding.reshape(1, -1).astype('float32')
                self.faiss_index.add(embedding_reshaped)
                
                # Get index position
                index = self.faiss_index.ntotal - 1
                
                # Update metadata
                metadata['faiss_index'] = index
                self.document_metadata[doc_id] = metadata
                self.doc_id_to_index[doc_id] = index
                self.index_to_doc_id[index] = doc_id
                
                added_count += 1
                
            except Exception as e:
                logger.error(f"Failed to add embedding for document {doc_id}: {e}")
                continue
        
        logger.info(f"Successfully added {added_count} embeddings to FAISS index")
        return added_count
    
    def build_index_from_corpus(self, force_rebuild: bool = False) -> Dict:
        """
        Build FAISS index from corpus
        
        Args:
            force_rebuild: Whether to force rebuild index
            
        Returns:
            Dict: Build result statistics
        """
        if not force_rebuild and self.faiss_index.ntotal > 0:
            logger.info("FAISS index already exists, skipping build")
            return {"status": "skipped", "reason": "index_already_exists"}
        
        start_time = time.time()
        
        # Load corpus
        documents = self.load_corpus_from_jsonl()
        if not documents:
            return {"status": "error", "reason": "failed_to_load_corpus"}
        
        # Batch process documents
        total_documents = len(documents)
        total_added = 0
        error_count = 0
        
        for i in range(0, total_documents, self.batch_size):
            batch = documents[i:i + self.batch_size]
            try:
                # Calculate embeddings
                embeddings_data = self.calculate_embeddings_batch(batch)
                if embeddings_data:
                    added_count = self.add_embeddings_to_faiss(embeddings_data)
                    total_added += added_count
                else:
                    error_count += len(batch)
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                error_count += len(batch)
        
        # Save index and metadata
        logger.info("Saving index and metadata...")
        self.save_faiss_index()
        logger.info("✅ Index and metadata saved!")
        
        processing_time = time.time() - start_time
        result = {
            "status": "success",
            "total_documents": total_documents,
            "added_documents": total_added,
            "error_count": error_count,
            "processing_time": processing_time
        }
        
        logger.info(f"Index building completed: {result}")
        return result
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents
        
        Args:
            query: Query text
            top_k: Return the top-k results
            
        Returns:
            List[Dict[str, Any]]: Retrieval result list, each element contains full document information and similarity score
        """
        try:
            # Calculate query text embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            query_embedding_normalized = query_embedding.reshape(1, -1).astype('float32')
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding_normalized, top_k * 2)  # 获取更多结果用于过滤
            
            results = []
            for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
                if index == -1:  # FAISS returns -1 means invalid index
                    continue
                
                doc_id = self.index_to_doc_id.get(index)
                if not doc_id:
                    continue
                
                metadata = self.document_metadata.get(doc_id)
                if not metadata:
                    continue
                
                # Combine document content
                content = f"title: {metadata['title']}\n\ncontent: {metadata['text']}"
                
                # Build complete result dictionary, including all metadata information
                result = {
                    'id': doc_id,
                    'content': content,
                    'score': float(distance),  # Convert to Python float
                    'title': metadata['title'],
                    'text': metadata['text'],
                    'metadata': metadata.get('metadata', {})
                }
                
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar documents: {e}")
            return []
    
    def retrieve_by_keywords(self, keywords: List[str], top_k: int = 10, company_filter: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve based on keywords
        
        Args:
            keywords: Keywords list
            top_k: Return the top-k results
            company_filter: Company name filter (can be string or list)
            
        Returns:
            List[Dict[str, Any]]: Retrieval result list, each element contains full document information and similarity score
        """
        try:
            if not keywords:
                return []
            
            # Combine keywords into a query string
            query = " ".join(keywords)
            
            # Use existing retrieve method
            results = self.retrieve(query, top_k=top_k * 2)
            
            # If company filter is specified, perform filtering
            if company_filter and results:
                filtered_results = []
                
                # Process company_filter, ensure it is list format
                if isinstance(company_filter, str):
                    company_filters = [company_filter]
                elif isinstance(company_filter, list):
                    company_filters = company_filter
                else:
                    company_filters = []
                
                for result in results:
                    # Check if document content contains company name
                    content = result.get('content', '')
                    company_name = result.get('company_name', '')
                    
                    # Use company_name field first, if not found, search in content
                    if not company_name and 'metadata' in result:
                        company_name = result['metadata'].get('company_name', '')
                    
                    # Check if matches any company constraint
                    matches_any_company = False
                    for company_constraint in company_filters:
                        if company_constraint.lower() in content.lower() or company_constraint.lower() in company_name.lower():
                            filtered_results.append(result)
                            if len(filtered_results) >= top_k:
                                break
                            matches_any_company = True
                            break
                    
                    # If no company constraint matches, also add (maintain backward compatibility)
                    if not matches_any_company and len(filtered_results) < top_k:
                        filtered_results.append(result)
                
                # If filtered results are too few, return original results
                if len(filtered_results) >= top_k // 2:
                    results = filtered_results[:top_k]
                else:
                    results = results[:top_k]
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to retrieve by keywords: {e}")
            return []
    
    def retrieve_with_reranking(self, query: str, top_k: int = 10, rerank_top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve with reranking
        
        Args:
            query: Query text
            top_k: Return the top-k results
            rerank_top_k: Rerank candidate number
            
        Returns:
            List[Dict[str, Any]]: Retrieval result list
        """
        # Get more candidate results
        candidates = self.retrieve(query, top_k=rerank_top_k)
        
        if len(candidates) <= top_k:
            return candidates
        
        # Simple reranking: based on similarity score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]
    
    def add_document(self, content: str, doc_id: str = None, title: str = None):
        """
        Add new document to index
        
        Args:
            content: Document content
            doc_id: Document ID (if None, generate automatically)
            title: Document title
        """
        if doc_id is None:
            doc_id = f"doc_{int(time.time() * 1000)}"
        
        # Prepare document
        document = {
            "_id": doc_id,
            "title": title or "",
            "text": content
        }
        
        # Calculate embedding and add to index
        embeddings_data = self.calculate_embeddings_batch([document])
        if embeddings_data:
            self.add_embeddings_to_faiss(embeddings_data)
            self.save_faiss_index()
            logger.info(f"Successfully added document {doc_id} to index")
    
    def add_documents_batch(self, documents: List[Dict[str, str]]):
        """
        Batch add documents
        
        Args:
            documents: Document list, each document contains content and optional id, title fields
        """
        # Convert to standard format
        standard_docs = []
        for doc in documents:
            standard_doc = {
                "_id": doc.get('id', f"doc_{int(time.time() * 1000)}"),
                "title": doc.get('title', ''),
                "text": doc.get('content', doc.get('text', ''))
            }
            standard_docs.append(standard_doc)
        
        # Calculate embeddings and add to index
        embeddings_data = self.calculate_embeddings_batch(standard_docs)
        if embeddings_data:
            self.add_embeddings_to_faiss(embeddings_data)
            self.save_faiss_index()
            logger.info(f"Successfully added {len(standard_docs)} documents to index")
    
    def save_faiss_index(self):
        """Save FAISS index and metadata"""
        try:
            # If index is on GPU, need to convert back to CPU before saving
            if hasattr(self.faiss_index, 'getDevice') and self.faiss_index.getDevice() != -1:
                logger.info("Detected GPU index, converting to CPU index for saving...")
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
                logger.info("GPU index converted to CPU index")
            else:
                cpu_index = self.faiss_index
            
            # Save FAISS index
            logger.info(f"Saving FAISS index to: {os.path.abspath(self.faiss_index_path)}")
            faiss.write_index(cpu_index, self.faiss_index_path)
            
            # Save metadata
            logger.info(f"Saving metadata to: {os.path.abspath(self.metadata_path)}")
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            logger.info(f"✅ Successfully saved FAISS index and metadata!")
            logger.info(f"  FAISS index: {os.path.abspath(self.faiss_index_path)}")
            logger.info(f"  Metadata: {os.path.abspath(self.metadata_path)}")
            logger.info(f"  Vector number: {self.faiss_index.ntotal}")
            logger.info(f"  Document number: {len(self.document_metadata)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index: {e}")
            import traceback
            logger.error(f"Error details: {traceback.format_exc()}")
            raise
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get corpus statistics
        
        Returns:
            Dict: Statistics information
        """
        try:
            # FAISS index statistics
            faiss_stats = {
                "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
                "dimension": self.faiss_index.d if self.faiss_index else 0,
                "is_trained": self.faiss_index.is_trained if self.faiss_index else False
            }
            
            # Metadata statistics
            metadata_stats = {
                "total_documents": len(self.document_metadata),
                "unique_titles": len(set(m.get('title', '') for m in self.document_metadata.values() if m.get('title')))
            }
            
            # Title distribution (take the top 10 most common titles)
            title_counts = {}
            for metadata in self.document_metadata.values():
                title = metadata.get('title', 'Unknown')
                if title:
                    title_counts[title] = title_counts.get(title, 0) + 1
            
            top_titles = sorted(title_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "faiss_index": faiss_stats,
                "metadata": metadata_stats,
                "top_titles": dict(top_titles),
                "model_name": self.model_name,
                "corpus_path": self.corpus_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get corpus statistics: {e}")
            return {}
    
    def search_by_title(self, title_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents based on title
        
        Args:
            title_query: Title query
            top_k: Return the number of results
            
        Returns:
            List[Dict[str, Any]]: Search result list
        """
        # Use title for retrieval
        results = self.retrieve(title_query, top_k=top_k)
        
        # Filter results, only keep documents with title matching
        filtered_results = []
        for result in results:
            title = result.get('title', '').lower()
            if title_query.lower() in title:
                filtered_results.append(result)
        
        return filtered_results[:top_k]
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Get document information by document ID

        Args:
            doc_id: document ID
            
        Returns:
            Optional[Dict]: Document information, if not exists, return None
        """
        return self.document_metadata.get(doc_id)
    
    def clear_index(self) -> Dict:
        """
        Clear FAISS index and all related data
        
        Returns:
            Dict: Clear result
        """
        try:
            # Clear FAISS index
            if hasattr(self.faiss_index, 'reset'):
                self.faiss_index.reset()
            else:
                # If FAISS index does not have reset method, create an empty one
                dimension = self.model.get_sentence_embedding_dimension()
                self.faiss_index = faiss.IndexFlatIP(dimension)
                if self.device.type == "cuda" and faiss.get_num_gpus() > 0:
                    self.faiss_index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, self.faiss_index
                    )
            
            # Clear metadata
            self.document_metadata.clear()
            self.doc_id_to_index.clear()
            self.index_to_doc_id.clear()
            
            # Delete index file
            if os.path.exists(self.faiss_index_path):
                os.remove(self.faiss_index_path)
                logger.info(f"Deleted index file: {self.faiss_index_path}")
            
            # Delete metadata file
            metadata_path = self.faiss_index_path.replace('.faiss', '_metadata.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info(f"Deleted metadata file: {metadata_path}")
            
            logger.info("Vector index has been completely cleared")
            return {
                "status": "success",
                "message": "Index has been cleared",
                "deleted_files": [self.faiss_index_path, metadata_path]
            }
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
 