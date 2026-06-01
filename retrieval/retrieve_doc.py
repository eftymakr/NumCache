#!/usr/bin/env python3

import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional
import argparse
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
retrieval_methods_dir = os.path.join(script_dir, 'retrieval_methods')
if retrieval_methods_dir not in sys.path:
    sys.path.insert(0, retrieval_methods_dir)

from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from elasticsearch_retriever import ElasticsearchRetriever
from vector_retriever import VectorRetriever

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

class DocumentRetriever:
    def __init__(self, retrieval_method: str = "vector",
                 corpus_dir: str = None, corpus_path: str = None, 
                 elasticsearch_config: Dict[str, Any] = None,
                 vector_model: str = None,
                 use_reranker: bool = False,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cuda:2"):
        """
        Initialize document retriever
        
        Args:
            retrieval_method: Retrieval method ("bm25", "hybrid", "elasticsearch", "vector")
            corpus_dir: Corpus directory path (for BM25 retrieval)
            corpus_path: Corpus file path (for vector retrieval, JSONL format)
            elasticsearch_config: Elasticsearch configuration dictionary
            vector_model: Vector retrieval model ("finlang" or "minilm"), default "finlang"
            use_reranker: Whether to use reranker
            reranker_model: Reranker model name
            device: GPU device (e.g. "cuda:2", "cuda:0", "cpu"), default "cuda:2"
        """
        print(f"Initializing {retrieval_method} retriever...")
        self.retrieval_method = retrieval_method
        self.corpus_dir = corpus_dir
        self.elasticsearch_config = elasticsearch_config
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.device = device
        print(f"Using device: {device}")
        
        self.reranker = None
        if use_reranker:
            if not RERANKER_AVAILABLE:
                raise ImportError("FlagEmbedding not installed, cannot use reranker functionality. Please run: pip install FlagEmbedding")
            print(f"Initializing reranker: {reranker_model} (device: {device})...")
            try:
                if device.startswith("cuda:"):
                    gpu_id = int(device.split(":")[1])
                    self.reranker = FlagReranker(reranker_model, use_fp16=True, devices=[f"cuda:{gpu_id}"])
                else:
                    self.reranker = FlagReranker(reranker_model, use_fp16=True)
                print(f"Reranker initialized!")
            except Exception as e:
                print(f"Warning: Reranker initialization failed: {e}")
                print("Will continue using original retrieval results, without reranking")
                self.use_reranker = False
                self.reranker = None
        
        if retrieval_method == "bm25":
            if not elasticsearch_config:
                raise ValueError("BM25 retrieval method requires elasticsearch_config parameter")
            self.retriever = BM25Retriever(
                index_name=elasticsearch_config.get('index_name', 'financial_corpus'),
                host=elasticsearch_config.get('host', 'localhost'),
                port=elasticsearch_config.get('port', 9200)
            )
        elif retrieval_method == "elasticsearch":
            if not elasticsearch_config:
                raise ValueError("Elasticsearch retrieval method requires elasticsearch_config parameter")
            
            # If vector search is needed, load embedding model
            embedding_model = None
            search_type = elasticsearch_config.get('search_type', 'hybrid')  # Default hybrid search
            
            if search_type in ["vector", "hybrid"]:
                # Select model based on vector_model (if provided)
                # Default use finlang (FinLang/finance-embeddings-investopedia) to match 768-dimensional vector index
                if vector_model:
                    if vector_model == "finlang":
                        model_name = "FinLang/finance-embeddings-investopedia"
                    elif vector_model == "minilm":
                        model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    else:
                        model_name = vector_model
                else:
                    # If vector_model is not provided, default use finlang to match index
                    model_name = "FinLang/finance-embeddings-investopedia"
                    print(f"No vector_model provided, using default model: {model_name} (matching 768-dimensional vector index)")
                    
                print(f"Loading embedding model for Elasticsearch vector search: {model_name} (device: {device})")
                try:
                    from sentence_transformers import SentenceTransformer
                    embedding_model = SentenceTransformer(model_name, device=device)
                    print(f"✓ Embedding model loaded successfully, dimension: {embedding_model.get_sentence_embedding_dimension()}")
                except Exception as e:
                    print(f"⚠️ Loading embedding model failed: {e}")
                    print("Will only use BM25 text search")
                    search_type = "text"
                    embedding_model = None
            
            self.retriever = ElasticsearchRetriever(
                index_name=elasticsearch_config.get('index_name', 'financial_corpus'),
                host=elasticsearch_config.get('host', 'localhost'),
                port=elasticsearch_config.get('port', 9200),
                username=elasticsearch_config.get('username'),
                password=elasticsearch_config.get('password'),
                embedding_model=embedding_model,
                search_type=search_type
            )
        elif retrieval_method == "hybrid":
            # Extract parameters from elasticsearch_config
            es_host = elasticsearch_config.get('host', 'localhost') if elasticsearch_config else 'localhost'
            es_port = elasticsearch_config.get('port', 9200) if elasticsearch_config else 9200
            es_index = elasticsearch_config.get('index_name', 'financial_corpus') if elasticsearch_config else 'financial_corpus'
            
            self.retriever = HybridRetriever(
                index_name=es_index,
                host=es_host,
                port=es_port,
                use_elasticsearch=True if elasticsearch_config else False,
                corpus_dir=corpus_dir
            )
        elif retrieval_method == "vector":
            if not corpus_path:
                raise ValueError("Vector retrieval method requires corpus_path parameter")
            
            # Select model based on vector_model
            if vector_model is None or vector_model == "finlang":
                model_name = "FinLang/finance-embeddings-investopedia"
            elif vector_model == "minilm":
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                raise ValueError(f"Unsupported vector model: {vector_model}, supported: 'finlang' or 'minilm'")
            
            print(f"Using vector model: {model_name} (device: {device})")
            # For vector retrieval, use specified model and device
            use_gpu = device.startswith("cuda")
            self.retriever = VectorRetriever(corpus_path=corpus_path, model_name=model_name, use_gpu=use_gpu)
            # If specific GPU is specified, need to move model to that device
            if device.startswith("cuda:"):
                try:
                    gpu_id = int(device.split(":")[1])
                    if hasattr(self.retriever, 'model'):
                        # Move model to specified GPU
                        self.retriever.model = self.retriever.model.to(f"cuda:{gpu_id}")
                        self.retriever.device = torch.device(f"cuda:{gpu_id}")
                        print(f"✓ Vector model moved to device: cuda:{gpu_id}")
                except Exception as e:
                    print(f"⚠️ Moving model to specified device failed: {e}")
                    print(f"    Will use default device")
        else:
            raise ValueError(f"Unsupported retrieval method: {retrieval_method}")
        
        print(f"{retrieval_method} retriever initialized!")
    
    def retrieve_doc_ids(self, question: str, top_k: int = 10) -> List[str]:
        """
        Retrieve document IDs related to the question
        
        Args:
            question: Question
            top_k: Top-k documents to retrieve (final number returned)
            
        Returns:
            List[str]: Retrieved document IDs list
        """
        try:
            # If reranker is used, first retrieve more candidate documents (top20)
            retrieve_k = top_k * 2 if self.use_reranker and self.reranker else top_k
            
            # Execute retrieval
            results = self.retriever.retrieve(question, top_k=retrieve_k)
            
            if not results:
                print(f"Warning: No documents related to the question: {question[:50]}...")
                return []
            
            # Prepare reranker data
            if self.use_reranker and self.reranker and len(results) > top_k:
                print(f"Retrieved {len(results)} candidate documents, using reranker to rerank...")
                reranked_results = self._rerank_results(question, results, top_k)
                results = reranked_results
                print(f"Reranking completed, keeping top {len(results)} documents")
            elif self.use_reranker and self.reranker and len(results) <= top_k:
                # If number of results is less than or equal to top_k, still can perform reranking to optimize order
                print(f"Retrieved {len(results)} candidate documents, using reranker to rerank...")
                reranked_results = self._rerank_results(question, results, len(results))
                results = reranked_results
                print(f"Reranking completed")
            
            # Extract doc_ids
            doc_ids = []
            if self.retrieval_method == "vector":
                # VectorRetriever returns format: List[Dict[str, Any]], containing 'id' field
                doc_ids = [result.get('id', '') for result in results if result.get('id')]
            else:
                # Other retrievers return format: List[Tuple[str, float, str]] (content, score, doc_id)
                doc_ids = [doc_id for _, _, doc_id in results if doc_id]
            
            print(f"Retrieved {len(doc_ids)} related document IDs")
            return doc_ids
            
        except Exception as e:
            print(f"Error retrieving document IDs: {e}")
            import traceback
            print(f"Detailed error stack: {traceback.format_exc()}")
            return []
    
    def _rerank_results(self, question: str, results: List, top_k: int) -> List:
        """
        Use reranker to rerank retrieval results
        
        Args:
            question: Query question
            results: Original retrieval results
            top_k: Top-k results returned
            
        Returns:
            List: Reranked results list
        """
        try:
            # Prepare reranker data: build (query, document) pairs
            pairs = []
            original_results = []
            
            for result in results:
                if self.retrieval_method == "vector":
                    # VectorRetriever returns format: Dict[str, Any]
                    content = result.get('content', '')
                    doc_id = result.get('id', '')
                    score = result.get('score', 0.0)
                    original_results.append((content, score, doc_id))
                else:
                    # Other retrievers return format: Tuple[str, float, str]
                    content, score, doc_id = result
                    original_results.append((content, score, doc_id))
                
                # Build (query, document) pairs for reranking
                pairs.append([question, content])
            
            # Execute reranking
            rerank_scores = self.reranker.compute_score(pairs)
            
            # If returned is a single score, convert to list
            if isinstance(rerank_scores, (int, float)):
                rerank_scores = [rerank_scores]
            
            # Combine original results and reranking scores
            scored_results = []
            for i, (content, original_score, doc_id) in enumerate(original_results):
                rerank_score = rerank_scores[i] if i < len(rerank_scores) else 0.0
                scored_results.append((content, rerank_score, doc_id, original_score))
            
            # Sort by reranking scores
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Convert back to original format and return top_k
            reranked_results = []
            for content, rerank_score, doc_id, original_score in scored_results[:top_k]:
                if self.retrieval_method == "vector":
                    # Return Dict format
                    reranked_results.append({
                        'id': doc_id,
                        'content': content,
                        'score': float(rerank_score),
                        'original_score': float(original_score)
                    })
                else:
                    # Return Tuple format
                    reranked_results.append((content, rerank_score, doc_id))
            
            return reranked_results
            
        except Exception as e:
            print(f"Error reranking: {e}")
            import traceback
            print(f"Detailed error stack: {traceback.format_exc()}")
            # If reranking fails, return top_k of original results
            return results[:top_k]
    
    def process_qa_file(self, file_path: str, top_k: int = 10) -> Dict[str, List[str]]:
        """
        Process QA file, retrieve doc_ids for each question
        
        Args:
            file_path: QA file path
            top_k: Top-k value to retrieve
            
        Returns:
            Dict[str, List[str]]: q_id -> doc_ids mapping
        """
        print(f"\n📖 Reading file: {os.path.basename(file_path)}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure list format
        if not isinstance(data, list):
            print(f"❌ File format error: expected list format, actual is {type(data)}")
            return {}
        
        print(f"📝 Detected {len(data)} questions")
        
        # Process each QA pair
        q_id_to_doc_ids = {}
        total_questions = len(data)
        
        for i, qa_pair in enumerate(data, 1):
            # Check necessary fields: support q_id or qid fields
            q_id = qa_pair.get('q_id') or qa_pair.get('qid')
            if not q_id:
                print(f"⚠️  Skip QA pair with missing q_id/qid fields (index {i})")
                continue
            
            if 'question' not in qa_pair:
                print(f"⚠️  Skip QA pair with missing question field (q_id: {q_id})")
                continue
            
            question = qa_pair['question']
            
            try:
                print(f"\n[{i}/{total_questions}] Processing q_id: {q_id}")
                print(f"Question: {question[:80]}...")
                
                # Retrieve document IDs
                doc_ids = self.retrieve_doc_ids(question, top_k=top_k)
                
                # Save results
                q_id_to_doc_ids[q_id] = doc_ids
                
                print(f"✓ q_id: {q_id} -> {len(doc_ids)} document IDs")
                
            except Exception as e:
                print(f"✗ Error processing q_id {q_id}: {e}")
                # Even if error, record empty list
                q_id_to_doc_ids[q_id] = []
        
        print(f"\n✅ File processing completed, processed {len(q_id_to_doc_ids)} questions")
        return q_id_to_doc_ids

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Retrieve document IDs script")
    parser.add_argument("--corpus_dir", type=str, default="/home/yidong/qa_dataset",
                       help="Corpus directory path (for BM25 retrieval)")
    parser.add_argument("--corpus_path", type=str, default="/home/yidong/kdd_rag/eval_clean_23chunks.json",
                       help="Corpus file path (for vector retrieval, JSONL or JSON array format)")
    parser.add_argument("--qa_dir", type=str, default="/home/yidong/qa",
                       help="QA dataset directory path (used when --qa_path is not specified)")
    parser.add_argument("--qa_path", type=str, default=None,
                       help="Specify the path of a single QA JSON file to process (if specified, only process this file)")
    parser.add_argument("--output_dir", type=str, default="/home/yidong/retrieval_with_llm/retrieval_results",
                       help="Output directory path (base directory, will create subdirectories according to retrieval method and QA category)")
    parser.add_argument("--output_structure", type=str, default="method_category",
                       choices=["method_category", "flat"],
                       help="Output directory structure: 'method_category' (retrieval method/QA category) or 'flat' (flat structure)")
    parser.add_argument("--retrieval_method", type=str, default="vector", 
                       choices=["bm25", "hybrid", "elasticsearch", "vector"],
                       help="Retrieval method")
    parser.add_argument("--vector_model", type=str, default="finlang",
                       choices=["finlang", "minilm"],
                       help="Vector retrieval model: 'finlang' (FinLang/finance-embeddings-investopedia) or 'minilm' (sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Top-k value to retrieve (final number of documents returned)")
    parser.add_argument("--use_reranker", action="store_true",
                       help="Whether to use reranker (retrieve top20, rerank and return top10)")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3",
                       help="Reranker model name (default: BAAI/bge-reranker-v2-m3)")
    
    # Elasticsearch related parameters
    parser.add_argument("--es_index", type=str, default="financial_corpus",
                       help="Elasticsearch index name")
    parser.add_argument("--es_host", type=str, default="localhost",
                       help="Elasticsearch host address")
    parser.add_argument("--es_port", type=int, default=9200,
                       help="Elasticsearch port")
    parser.add_argument("--es_username", type=str, default=None,
                       help="Elasticsearch username")
    parser.add_argument("--es_password", type=str, default=None,
                       help="Elasticsearch password")
    parser.add_argument("--es_search_type", type=str, default="hybrid",
                       choices=["text", "vector", "hybrid"],
                       help="Elasticsearch search type: 'text' (BM25 only), 'vector' (vector only), 'hybrid' (BM25 + vector)")
    parser.add_argument("--device", type=str, default="cuda:2",
                       help="GPU device (e.g. 'cuda:2', 'cuda:0', 'cpu'), default is 'cuda:2'")
    
    args = parser.parse_args()
           
    if args.qa_path:
        if not os.path.exists(args.qa_path):
            print(f"Error: QA file does not exist: {args.qa_path}")
            return
        if not args.qa_path.endswith('.json'):
            print(f"Warning: specified file is not JSON format: {args.qa_path}")
    else:
        if not os.path.exists(args.qa_dir):
            print(f"Error: qa directory does not exist: {args.qa_dir}")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories for each retrieval method
    # For vector retrieval, include model name
    if args.retrieval_method == "vector":
        method_name = f"vector_{args.vector_model}"
    else:
        method_name = args.retrieval_method
    
    # If using reranker, add _reranker suffix to folder name
    if args.use_reranker:
        method_name = f"{method_name}_reranker"
    
    # Determine directory organization based on output structure
    if args.output_structure == "method_category":
        # Structure: {output_dir}/{method_name}/{qa_category}/
        base_output_dir = os.path.join(args.output_dir, method_name)
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Base output directory: {args.output_dir}")
        print(f"Retrieval method directory: {method_name}")
        print(f"Will create subdirectories by QA category")
    else:
        # Flat structure: {output_dir}/{method_name}/
        base_output_dir = os.path.join(args.output_dir, method_name)
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Base output directory: {args.output_dir}")
        print(f"Retrieval method output directory: {base_output_dir}")
    
    if args.use_reranker:
        print(f"Using reranker model: {args.reranker_model}")
    
    # Build Elasticsearch configuration
    elasticsearch_config = None
    if args.retrieval_method in ["elasticsearch", "hybrid", "bm25"]:
        elasticsearch_config = {
            "index_name": args.es_index,
            "host": args.es_host,
            "port": args.es_port,
            "username": args.es_username,
            "password": args.es_password,
            "search_type": getattr(args, 'es_search_type', 'hybrid')  # Default hybrid search
        }
        print(f"Elasticsearch配置: {elasticsearch_config}")
    
    # Check corpus path (for vector retrieval)
    if args.retrieval_method == "vector":
        if not args.corpus_path:
            print(f"Error: vector retrieval method needs to provide corpus_path parameter")
            return
        
        if not os.path.exists(args.corpus_path):
            print(f"Error: corpus file does not exist: {args.corpus_path}")
            return
    
    # For elasticsearch method, if using vector search, need vector_model
    if args.retrieval_method == "elasticsearch" and args.es_search_type in ["vector", "hybrid"]:
        if not args.vector_model:
            print(f"Warning: Elasticsearch vector search needs to provide --vector_model parameter, using default value 'finlang' (FinLang/finance-embeddings-investopedia)")
            args.vector_model = "finlang"
    
    # Initialize document retriever
    try:
        retriever = DocumentRetriever(
            retrieval_method=args.retrieval_method,
            corpus_dir=args.corpus_dir,
            corpus_path=args.corpus_path,
            elasticsearch_config=elasticsearch_config,
            vector_model=args.vector_model if args.retrieval_method in ["vector", "elasticsearch"] else None,
            use_reranker=args.use_reranker,
            reranker_model=args.reranker_model,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return
    
    # Process each file
    all_results = {}
    
    if args.qa_path:
        # If qa_path is specified, only process this file
        file_path = args.qa_path
        file_name = os.path.basename(file_path)
        
        print(f"\n🚀 Starting to process the specified file...")
        print(f"{'='*80}")
        print(f"\n📄 Processing file: {file_name}")
        print(f"    Full path: {file_path}")
        print(f"{'='*80}")
        
        try:
            # Process file
            q_id_to_doc_ids = retriever.process_qa_file(file_path, top_k=args.top_k)
            
            # Determine output directory and file name
            base_name = os.path.splitext(file_name)[0]
            if args.output_structure == "method_category":
                # Use file name as QA category directory name
                qa_category_dir = os.path.join(base_output_dir, base_name)
                os.makedirs(qa_category_dir, exist_ok=True)
                output_file = os.path.join(qa_category_dir, f"{base_name}_retrieved_doc_ids.json")
            else:
                output_file = os.path.join(base_output_dir, f"{base_name}_retrieved_doc_ids.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(q_id_to_doc_ids, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Results saved to: {output_file}")
            print(f"    Total {len(q_id_to_doc_ids)} questions, retrieved document IDs")
            
            # Merge to total results
            all_results[file_name] = q_id_to_doc_ids
            
        except Exception as e:
            print(f"❌ Error processing file {file_name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # If qa_path is not specified, process all JSON files in qa_dir directory
        import glob
        json_files = glob.glob(os.path.join(args.qa_dir, "*.json"))
        
        if not json_files:
            print(f"⚠️  No JSON files found in {args.qa_dir} directory")
            return
        
        # Only process file names, not including path
        target_files = [os.path.basename(f) for f in json_files]
        
        print(f"\n🚀 Starting to process {len(target_files)} files...")
        print(f"{'='*80}")
        
        for file_name in target_files:
            file_path = os.path.join(args.qa_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"⚠️  File does not exist, skipping: {file_path}")
                continue
            
            print(f"\n📄 Processing file: {file_name}")
            print(f"{'='*80}")
            
            try:
                # Process file
                q_id_to_doc_ids = retriever.process_qa_file(file_path, top_k=args.top_k)
                
                # Determine output directory and file name
                base_name = os.path.splitext(file_name)[0]
                if args.output_structure == "method_category":
                    # Use file name as QA category directory name
                    qa_category_dir = os.path.join(base_output_dir, base_name)
                    os.makedirs(qa_category_dir, exist_ok=True)
                    output_file = os.path.join(qa_category_dir, f"{base_name}_retrieved_doc_ids.json")
                else:
                    output_file = os.path.join(base_output_dir, f"{base_name}_retrieved_doc_ids.json")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(q_id_to_doc_ids, f, ensure_ascii=False, indent=2)
                
                print(f"✅ Results saved to: {output_file}")
                print(f"    Total {len(q_id_to_doc_ids)} questions, retrieved document IDs")
                
                # Merge to total results
                all_results[file_name] = q_id_to_doc_ids
                
            except Exception as e:
                print(f"❌ Error processing file {file_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save merged results
    if all_results:
        if args.output_structure == "method_category":
            merged_output_file = os.path.join(base_output_dir, "all_retrieved_doc_ids.json")
        else:
            merged_output_file = os.path.join(base_output_dir, "all_retrieved_doc_ids.json")
        with open(merged_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ All results merged and saved to: {merged_output_file}")
        
        # Statistics information
        total_questions = sum(len(doc_ids) for doc_ids in all_results.values())
        total_doc_ids = sum(len(doc_ids) for q_id_to_doc_ids in all_results.values() for doc_ids in q_id_to_doc_ids.values())
        
        print(f"\n📊 Statistics information:")
        print(f"    Total number of files processed: {len(all_results)}")
        print(f"    Total number of questions: {total_questions}")
        print(f"    Total number of document IDs: {total_doc_ids}")
        print(f"    Average number of documents retrieved per question: {total_doc_ids / total_questions if total_questions > 0 else 0:.2f}")
    
    print(f"\n{'='*80}")
    print(f"🎉 Processing completed!")

if __name__ == "__main__":
    main()
