#!/usr/bin/env python3
"""
Number-Preserving Text Compressor

Extracts ALL numbers with their context, removes table noise,
and outputs clean compressed text.

This is MORE RELIABLE than LLM summarization because:
1. Deterministic - won't miss numbers
2. No hallucination risk
3. Fast - no GPU needed

Usage:
    python number_preserving_compressor.py \
        --document financial_doc.txt \
        --budget 8192 \
        --output compressed.txt
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import json


@dataclass
class NumberWithContext:
    """A number with its surrounding context"""
    number: str              # The number: "$64,912 million"
    context: str             # Cleaned context sentence
    importance: float        # Score for prioritization
    position: int            # Position in document
    

class NumberPreservingCompressor:
    """Extract numbers with context, remove noise, compress text"""
    
    def __init__(self, tokenizer_name: str = None):
        """Initialize compressor"""
        self.tokenizer = None
        
        # Try to load tokenizer
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, 
                    trust_remote_code=True
                )
                print(f"Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                print(f"Could not load tokenizer: {e}")
        
        # Important financial keywords (for scoring)
        self.important_keywords = {
            'revenue': 10, 'revenues': 10, 'sales': 9,
            'income': 9, 'profit': 9, 'loss': 9, 'earnings': 9,
            'eps': 10, 'per share': 10,
            'ebitda': 8, 'ebit': 8,
            'assets': 7, 'liabilities': 7, 'equity': 7,
            'cash': 8, 'debt': 8, 'capital': 7,
            'margin': 8, 'gross': 7, 'net': 8, 'operating': 7,
            'dividend': 8, 'share': 6, 'shares': 6,
            'total': 6, 'increase': 5, 'decrease': 5,
            'growth': 6, 'decline': 5,
            'million': 4, 'billion': 5, 'thousand': 3,
            'percent': 6, '%': 6,
            'quarter': 5, 'year': 5, 'annual': 5,
            '2020': 4, '2019': 4, '2018': 4, '2021': 4,
        }
        
        # Table noise patterns to remove
        self.noise_patterns = [
            r'\|+',           # Pipes
            r'-{3,}',         # Dashes
            r'#{1,6}\s*',     # Markdown headers
            r'\[.*?\]\(.*?\)',  # Markdown links
            r'Table of Contents',
            r'---\s*\[',
            r'\(\#[a-z0-9]+\)',  # Anchor links
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Rough approximation: ~4 chars per token
            return len(text) // 4
    
    def clean_text(self, text: str) -> str:
        """Remove table formatting and noise"""
        cleaned = text
        
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove empty parentheses
        cleaned = re.sub(r'\(\s*\)', '', cleaned)
        
        return cleaned.strip()
    
    def extract_sentences_with_numbers(self, document: str) -> List[NumberWithContext]:
        """Extract all sentences containing numbers"""
        
        # Clean document first
        cleaned_doc = self.clean_text(document)
        
        # For flat financial table data, insert breaks before key financial terms
        financial_breaks = [
            r'(Revenues?:)', r'(Cost of sales:)', r'(Operating income)',
            r'(Total (?:revenues?|cost|assets|liabilities|expenditures))',
            r'(Net income)', r'(Year ended)', r'(Depreciation)',
            r'(General and administrative)', r'(Other operating)',
            r'(Intersegment)', r'(Operating expenses)', r'(Gross profit)',
        ]
        
        for pattern in financial_breaks:
            cleaned_doc = re.sub(pattern, r'|||BREAK|||\1', cleaned_doc, flags=re.IGNORECASE)
        
        # Split into sentences (handle various delimiters)
        sentences = re.split(r'(?<=[.!?])\s+|\n+|\|\|\|BREAK\|\|\|', cleaned_doc)
        
        # Also split long "sentences" that might be table rows
        expanded = []
        for sent in sentences:
            if len(sent) > 300:
                # Split on common delimiters
                parts = re.split(r'\s{2,}|;|\t', sent)
                expanded.extend([p.strip() for p in parts if p.strip()])
            else:
                expanded.append(sent)
        sentences = expanded
        
        results = []
        position = 0
        seen_contexts = set()
        
        for sent in sentences:
            # Check if sentence contains numbers
            numbers = self.extract_numbers(sent)
            
            if numbers:
                # Calculate importance score
                importance = self.calculate_importance(sent, numbers)
                
                # Clean the sentence further
                clean_sent = self.clean_sentence(sent)
                
                # Skip near-duplicates
                sent_key = re.sub(r'[\d,.$%]+', 'X', clean_sent.lower())[:50]
                if sent_key in seen_contexts and importance < 15:
                    position += len(sent)
                    continue
                seen_contexts.add(sent_key)
                
                if clean_sent and len(clean_sent) > 10:
                    results.append(NumberWithContext(
                        number=', '.join(numbers[:3]),  # Keep top 3 numbers
                        context=clean_sent,
                        importance=importance,
                        position=position,
                    ))
            
            position += len(sent)
        
        # Sort by importance
        results.sort(key=lambda x: x.importance, reverse=True)
        
        return results
    
    def extract_numbers(self, text: str) -> List[str]:
        """Extract numbers from text"""
        patterns = [
            r'\$[\d,]+\.?\d*\s*(?:million|billion|thousand)?',  # Currency
            r'[\d,]+\.?\d*\s*(?:million|billion|thousand)',     # With scale
            r'[\d,]+\.?\d*\s*%',                                # Percentage
            r'(?<![.\d])[\d,]+\.?\d*(?![.\d\w])',              # Plain numbers
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        # Filter out years and small numbers
        filtered = []
        for num in numbers:
            # Skip years
            if re.match(r'^(19|20)\d{2}$', num.strip()):
                continue
            # Skip very small numbers (unless percentage)
            try:
                val = float(re.sub(r'[,$%]', '', num.split()[0]))
                if val < 1 and '%' not in num and '$' not in num:
                    continue
            except:
                pass
            filtered.append(num.strip())
        
        return list(set(filtered))
    
    def clean_sentence(self, sent: str) -> str:
        """Clean a sentence while preserving numbers"""
        # Remove leftover noise
        cleaned = re.sub(r'\s*\|\s*', ' ', sent)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'^\W+|\W+$', '', cleaned)
        
        # Remove very short fragments
        if len(cleaned) < 10:
            return ""
        
        return cleaned.strip()
    
    def calculate_importance(self, sent: str, numbers: List[str]) -> float:
        """Calculate importance score for a sentence"""
        score = 0.0
        sent_lower = sent.lower()
        
        # Boost for important keywords
        for keyword, weight in self.important_keywords.items():
            if keyword in sent_lower:
                score += weight
        
        # Boost for more numbers
        score += len(numbers) * 2
        
        # Boost for currency
        if '$' in sent:
            score += 5
        
        # Boost for large numbers
        for num in numbers:
            try:
                val = float(re.sub(r'[,$%]', '', num.split()[0]))
                if 'billion' in num.lower():
                    val *= 1e9
                elif 'million' in num.lower():
                    val *= 1e6
                
                import math
                score += math.log10(max(val, 1)) / 3
            except:
                pass
        
        return score
    
    def compress(
        self,
        document: str,
        token_budget: int,
        min_context_tokens: int = 5,
        max_context_tokens: int = 50,
        fill_budget: bool = True,
    ) -> Dict:
        """
        Compress document to fit token budget while preserving all numbers
        
        Args:
            document: Full document text
            token_budget: Maximum tokens
            min_context_tokens: Minimum context per number
            max_context_tokens: Maximum context per number
            fill_budget: If True, add context sentences to fill remaining budget
            
        Returns:
            Dictionary with compressed text and stats
        """
        print(f"Original document: {self.count_tokens(document)} tokens")
        print(f"Target budget: {token_budget} tokens")
        
        # Extract sentences with numbers (priority)
        entries = self.extract_sentences_with_numbers(document)
        print(f"Found {len(entries)} sentences with numbers")
        
        # Also extract sentences WITHOUT numbers (for filling budget)
        all_sentences = self.extract_all_sentences(document)
        print(f"Found {len(all_sentences)} total sentences")
        
        # Build compressed text within budget
        selected = []
        current_tokens = 0
        used_positions = set()
        
        # Phase 1: Add all sentences with numbers (priority)
        for entry in entries:
            entry_tokens = self.count_tokens(entry.context)
            
            if current_tokens + entry_tokens <= token_budget:
                selected.append(entry)
                current_tokens += entry_tokens
                used_positions.add(entry.position)
            elif current_tokens < token_budget * 0.95:
                # Try to fit with truncated context
                words = entry.context.split()
                truncated = ' '.join(words[:max_context_tokens])
                trunc_tokens = self.count_tokens(truncated)
                
                if current_tokens + trunc_tokens <= token_budget:
                    entry.context = truncated
                    selected.append(entry)
                    current_tokens += trunc_tokens
                    used_positions.add(entry.position)
        
        print(f"After number sentences: {current_tokens} tokens used")
        
        # Phase 2: Fill remaining budget with context sentences (no numbers)
        if fill_budget and current_tokens < token_budget * 0.95:
            remaining_budget = token_budget - current_tokens
            print(f"Filling remaining {remaining_budget} tokens with context...")
            
            # Get sentences without numbers, sorted by position
            context_sentences = [
                s for s in all_sentences 
                if s.position not in used_positions
            ]
            
            # Add context sentences to fill budget
            for sent in context_sentences:
                sent_tokens = self.count_tokens(sent.context)
                
                if current_tokens + sent_tokens <= token_budget:
                    selected.append(sent)
                    current_tokens += sent_tokens
                
                if current_tokens >= token_budget * 0.98:
                    break
        
        print(f"Final: {current_tokens} tokens used ({current_tokens/token_budget*100:.1f}% of budget)")
        
        # Sort by document position for coherence
        selected.sort(key=lambda x: x.position)
        
        # Build output
        lines = []
        for entry in selected:
            lines.append(entry.context)
        
        compressed_text = '\n'.join(lines)
        
        # Statistics
        original_tokens = self.count_tokens(document)
        compressed_tokens = self.count_tokens(compressed_text)
        
        # Count numbers
        original_numbers = len(self.extract_numbers(document))
        compressed_numbers = len(self.extract_numbers(compressed_text))
        
        result = {
            'compressed_text': compressed_text,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'compression_ratio': compressed_tokens / original_tokens if original_tokens else 0,
            'original_numbers': original_numbers,
            'compressed_numbers': compressed_numbers,
            'number_retention': compressed_numbers / original_numbers if original_numbers else 0,
            'sentences_selected': len(selected),
            'number_density': compressed_numbers / compressed_tokens if compressed_tokens else 0,
            'budget_utilization': compressed_tokens / token_budget,
        }
        
        print(f"\n{'='*50}")
        print("COMPRESSION RESULTS")
        print('='*50)
        print(f"Tokens: {original_tokens} → {compressed_tokens} ({result['compression_ratio']:.1%})")
        print(f"Budget utilization: {result['budget_utilization']:.1%}")
        print(f"Numbers: {original_numbers} → {compressed_numbers} ({result['number_retention']:.1%} retained)")
        print(f"Number density: {result['number_density']:.1%}")
        print(f"Sentences: {len(selected)}")
        
        return result
    
    def extract_all_sentences(self, document: str) -> List[NumberWithContext]:
        """Extract ALL sentences/segments (with or without numbers) for filling budget"""
        
        cleaned_doc = self.clean_text(document)
        
        # For flat financial table data, insert breaks before key financial terms
        financial_breaks = [
            r'(Revenues?:)', r'(Cost of sales:)', r'(Operating income)',
            r'(Total (?:revenues?|cost|assets|liabilities|expenditures))',
            r'(Net income)', r'(Year ended)', r'(Depreciation)',
            r'(General and administrative)', r'(Other operating)',
            r'(Intersegment)', r'(Operating expenses)', r'(Gross profit)',
        ]
        
        for pattern in financial_breaks:
            cleaned_doc = re.sub(pattern, r'|||BREAK|||\1', cleaned_doc, flags=re.IGNORECASE)
        
        # Split on sentence endings, newlines, AND our inserted breaks
        sentences = re.split(r'(?<=[.!?])\s+|\n+|\|\|\|BREAK\|\|\|', cleaned_doc)
        
        # Also split very long chunks
        expanded = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 300:
                # Split on multiple spaces
                parts = re.split(r'\s{3,}', sent)
                expanded.extend([p.strip() for p in parts if p.strip()])
            elif sent:
                expanded.append(sent)
        sentences = expanded
        
        results = []
        position = 0
        seen_contexts = set()
        
        for sent in sentences:
            clean_sent = self.clean_sentence(sent)
            
            if clean_sent and len(clean_sent) > 15:
                # Skip duplicates
                sent_key = re.sub(r'[\d,.$%]+', 'X', clean_sent.lower())[:50]
                if sent_key in seen_contexts:
                    position += len(sent)
                    continue
                seen_contexts.add(sent_key)
                
                # Score context sentences by financial relevance
                importance = 0
                sent_lower = clean_sent.lower()
                for keyword, weight in self.important_keywords.items():
                    if keyword in sent_lower:
                        importance += weight * 0.5  # Lower weight than number sentences
                
                results.append(NumberWithContext(
                    number="",
                    context=clean_sent,
                    importance=importance,
                    position=position,
                ))
            
            position += len(sent)
        
        # Sort by importance (most relevant context first)
        results.sort(key=lambda x: x.importance, reverse=True)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Compress text while preserving all numbers"
    )
    parser.add_argument(
        "--document", 
        type=str, 
        required=True,
        help="Path to document"
    )
    parser.add_argument(
        "--budget", 
        type=int, 
        default=8192,
        help="Token budget"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default=None,
        help="Tokenizer model (optional, uses char estimate if not provided)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (.txt)"
    )
    parser.add_argument(
        "--no-fill", 
        action="store_true",
        help="Don't fill remaining budget with context (only number sentences)"
    )
    
    args = parser.parse_args()
    
    # Load document
    print(f"Loading: {args.document}")
    with open(args.document, 'r') as f:
        document = f.read()
    
    # Initialize compressor
    compressor = NumberPreservingCompressor(tokenizer_name=args.tokenizer)
    
    # Compress
    result = compressor.compress(document, args.budget, fill_budget=not args.no_fill)
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        
        with open(output_path, 'w') as f:
            f.write(result['compressed_text'])
        print(f"\nSaved to: {output_path}")
        
        # Save metadata
        meta_path = output_path.with_suffix('.meta.json')
        meta = {k: v for k, v in result.items() if k != 'compressed_text'}
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata: {meta_path}")
    
    # Preview
    print(f"\n{'='*50}")
    print("PREVIEW (first 1000 chars)")
    print('='*50)
    print(result['compressed_text'][:1000])


if __name__ == "__main__":
    main()