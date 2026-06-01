#!/usr/bin/env python3
"""
Set up Elasticsearch index and add documents
"""

import json
import os
import argparse
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from retrieval_methods.semantic_retriever import SemanticRetriever
from retrieval_methods.bm25_retriever import BM25Retriever

def clear_existing_index(host: str, port: int, index_name: str) -> bool:
    """Clear existing Elasticsearch index"""
    try:
        es = Elasticsearch([{'host': host, 'port': port}])
        
        if es.indices.exists(index=index_name):
            print(f"Found existing index: {index_name}")
            print("Deleting old index...")
            es.indices.delete(index=index_name)
            print(f"✓ Index {index_name} deleted")
            return True
        else:
            print(f"Index {index_name} does not exist, no need to delete")
            return True
            
    except Exception as e:
        print(f"Failed to clear index: {e}")
        return False

def create_bm25_index(host: str, port: int, index_name: str) -> bool:
    """Create a simple BM25 index (text only, no vectors)"""
    try:
        es = Elasticsearch([{'host': host, 'port': port}])
        
        # Simple mapping for BM25 (text only)
        mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        
        es.indices.create(index=index_name, body=mapping)
        print(f"✓ Created BM25 index {index_name} (text only, no vectors)")
        return True
        
    except Exception as e:
        print(f"Failed to create BM25 index: {e}")
        return False

def load_documents_from_jsonl(corpus_path: str) -> List[Dict[str, Any]]:
    """Load documents from JSONL file or JSON array file"""
    documents = []
    
    if not os.path.exists(corpus_path):
        print(f"Error: file {corpus_path} does not exist")
        return documents
    
    print(f"Loading documents from {corpus_path}...")
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try to parse as JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                # It's a JSON array
                print(f"Detected JSON array format, found {len(data)} documents")
                documents = data
            else:
                # Single JSON object, wrap in list
                documents = [data]
        except json.JSONDecodeError:
            # If not JSON array, try JSONL format (one JSON object per line)
            print("Trying JSONL format (one JSON object per line)")
            for line_num, line in enumerate(content.split('\n'), 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Warning: line {line_num} JSON parsing failed: {e}")
                except Exception as e:
                    print(f"Warning: error processing line {line_num}: {e}")
        
        # Normalize document format
        normalized_documents = []
        for doc in documents:
            if not isinstance(doc, dict):
                print(f"Warning: skipping non-dict document: {type(doc)}")
                continue
            
            normalized_doc = doc.copy()
            
            # Normalize ID field: use chunk_id as _id if _id doesn't exist
            if '_id' not in normalized_doc:
                if 'chunk_id' in normalized_doc:
                    normalized_doc['_id'] = normalized_doc['chunk_id']
                elif 'id' in normalized_doc:
                    normalized_doc['_id'] = normalized_doc['id']
                else:
                    # Generate a unique ID
                    normalized_doc['_id'] = f"chunk_{len(normalized_documents)}"
            
            # Standardize text field: use content field if text doesn't exist
            if 'text' not in normalized_doc:
                if 'content' in normalized_doc:
                    normalized_doc['text'] = normalized_doc['content']
            
            # Ensure necessary fields
            if 'text' in normalized_doc:
                normalized_documents.append(normalized_doc)
            else:
                print(f"Warning: document {normalized_doc.get('_id', 'unknown')} missing text/content field")
        
        print(f"Successfully loaded {len(normalized_documents)} documents")
        return normalized_documents
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        import traceback
        print(traceback.format_exc())
        return []

def main():
    parser = argparse.ArgumentParser(description="Set up Elasticsearch index and add documents")
    parser.add_argument("--corpus_path", default="/home/yidong/kdd_rag/vlo_psx_all_corpus.jsonl", help="Corpus file path (JSONL or JSON array format)")
    parser.add_argument("--index_name", default="financial_corpus", help="Elasticsearch index name")
    parser.add_argument("--host", default="localhost", help="Elasticsearch host address")
    parser.add_argument("--port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--model_name", default="FinLang/finance-embeddings-investopedia", 
                       help="Sentence Transformer model name (only used for semantic/vector search)")
    parser.add_argument("--bm25_only", action="store_true",
                       help="Use BM25 only mode (no vectors, no model needed)")
    parser.add_argument("--device", default=None,
                       help="Device to use for model ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc.). "
                            "If not specified, will auto-select an available GPU or use CPU")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding documents (default: 32)")
    
    args = parser.parse_args()
    
    print("=== Elasticsearch index setup tool ===")
    print(f"Corpus file: {args.corpus_path}")
    print(f"Index name: {args.index_name}")
    print(f"Elasticsearch address: {args.host}:{args.port}")
    if args.bm25_only:
        print("Mode: BM25 only (no vectors, no model needed)")
        print("⚠️  Model name parameter will be ignored")
    else:
        print("Mode: Semantic/Vector search")
        print(f"Model name: {args.model_name}")
    print()
    
    # Check existing index
    es = Elasticsearch([{'host': args.host, 'port': args.port}])
    if es.indices.exists(index=args.index_name):
        print(f"⚠️  警告: Index {args.index_name} already exists")
        if args.bm25_only:
            print("This operation will delete existing index and create a new BM25 index (text only)")
        else:
            print("This operation will delete existing index and create a new index with vector fields")
        print("Existing data will be lost!")
        
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled")
            return
    
    # Clear existing index
    print("=== Clear existing index ===")
    if clear_existing_index(args.host, args.port, args.index_name):
        print(f"✓ Index {args.index_name} cleared")
    else:
        print(f"Index {args.index_name} clear failed, please check connection or permissions")
    
    # Load documents
    documents = load_documents_from_jsonl(args.corpus_path)
    if not documents:
        print("No documents loaded, exiting")
        return
    
    if args.bm25_only:
        # BM25 only mode: create simple text index and use BM25Retriever
        print("=== Creating BM25 index ===")
        if not create_bm25_index(args.host, args.port, args.index_name):
            print("Failed to create BM25 index")
            return
        
        # Initialize BM25 retriever
        print("Initializing BM25 retriever...")
        try:
            retriever = BM25Retriever(
                index_name=args.index_name,
                host=args.host,
                port=args.port
            )
        except Exception as e:
            print(f"Failed to initialize BM25 retriever: {e}")
            return
        
        # Add documents to index
        print("Adding documents to index...")
        try:
            # Convert documents to format expected by BM25Retriever
            bm25_docs = []
            for doc in documents:
                doc_id = doc.get('_id', doc.get('id'))
                content = doc.get('text', doc.get('content', ''))
                bm25_doc = {'content': content}
                if doc_id:
                    bm25_doc['id'] = doc_id
                bm25_docs.append(bm25_doc)
            
            retriever.add_documents_batch(bm25_docs)
            success = True
        except Exception as e:
            print(f"Failed to add documents: {e}")
            success = False
        
        if success:
            print("\n=== Setup completed ===")
            print(f"✓ BM25 index {args.index_name} created successfully")
            print(f"✓ Added {len(documents)} documents")
            print("✓ Now you can use BM25 search (no model needed)")
            
            # Test search
            print("\nTesting BM25 search functionality...")
            try:
                test_query = "financial risk"
                results = retriever.retrieve(test_query, top_k=3)
                print(f"Test query: '{test_query}'")
                print(f"Found {len(results)} results:")
                for i, (content, score, doc_id) in enumerate(results, 1):
                    print(f"  {i}. [Score: {score:.3f}] {content[:100]}...")
            except Exception as e:
                print(f"Error testing search: {e}")
        else:
            print("\n=== Setup failed ===")
            print("Please check Elasticsearch connection and configuration")
    
    else:
        # Semantic/Vector mode: use SemanticRetriever with model
        print("=== Creating semantic/vector index ===")
        print("Initializing semantic retriever...")
        try:
            retriever = SemanticRetriever(
                model_name=args.model_name,
                index_name=args.index_name,
                host=args.host,
                port=args.port,
                use_elasticsearch=True,
                device=args.device
            )
        except Exception as e:
            print(f"Failed to initialize semantic retriever: {e}")
            return
        
        # Add documents to index
        print("Adding documents to index...")
        success = retriever.add_documents(documents, batch_size=args.batch_size)
        
        if success:
            print("\n=== Setup completed ===")
            print(f"✓ Index {args.index_name} created successfully")
            print(f"✓ Added {len(documents)} documents")
            print("✓ Now you can use semantic/vector search")
            
            # Test search
            print("\nTesting semantic search functionality...")
            try:
                test_query = "financial risk"
                results = retriever.retrieve(test_query, top_k=3)
                print(f"Test query: '{test_query}'")
                print(f"Found {len(results)} results:")
                for i, (content, score, doc_id) in enumerate(results, 1):
                    print(f"  {i}. [Score: {score:.3f}] {content[:100]}...")
            except Exception as e:
                print(f"Error testing search: {e}")
        else:
            print("\n=== Setup failed ===")
            print("Please check Elasticsearch connection and configuration")

if __name__ == "__main__":
    main() 