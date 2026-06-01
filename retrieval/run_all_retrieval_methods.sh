#!/bin/bash

CORPUS_PATH="/home/yidong/DRAGIN/enhanced_corpus_new.jsonl"
#QA_PATH="/home/yidong/kdd_rag/eval_clean_23chunks.json"
OUTPUT_DIR="/home/yidong/retrieval_with_llm/retrieval_results_2"
TOP_K=10

ES_INDEX="financial_corpus"
ES_HOST="localhost"
ES_PORT=9200

echo "=========================================="
echo "Running all retrieval methods"
echo "=========================================="
echo "Corpus: $CORPUS_PATH"
echo "QA file: $QA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Top-K: $TOP_K"
echo "=========================================="
echo ""



echo "=========================================="
echo "[5/10] Elasticsearch (no reranker)"
echo "=========================================="
python3 retrieve_doc.py \
    --corpus_path "$CORPUS_PATH" \
    --qa_path "$QA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --retrieval_method elasticsearch \
    --top_k $TOP_K \
    --es_index "$ES_INDEX" \
    --es_host "$ES_HOST" \
    --es_port $ES_PORT
echo ""

echo "=========================================="
echo "[6/10] Elasticsearch (with reranker)"
echo "=========================================="
python3 retrieve_doc.py \
    --corpus_path "$CORPUS_PATH" \
    --qa_path "$QA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --retrieval_method elasticsearch \
    --top_k $TOP_K \
    --use_reranker \
    --es_index "$ES_INDEX" \
    --es_host "$ES_HOST" \
    --es_port $ES_PORT
echo ""


echo "=========================================="
echo "✅ All retrieval methods completed!"
echo "=========================================="
echo "Results saved in: $OUTPUT_DIR"
echo ""
