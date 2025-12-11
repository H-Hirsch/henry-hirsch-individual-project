"""
Federal Register Clean Text Fetcher

Usage:
    In Jupyter/Python:
        from federal_register_fetcher import CleanFederalRegisterFetcher
        df = pd.read_csv('fr_tracking.csv')
        fetcher = CleanFederalRegisterFetcher(delay=0.5)
        df_with_text = fetcher.process_dataframe(df, 'output.csv')
    
    Command line:
        python federal_register_fetcher.py fr_tracking.csv
"""

import pandas as pd
import requests
import time
from tqdm import tqdm
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import html
import sys
import re


class CleanFederalRegisterFetcher:
    """
    Fetches Federal Register text using the cleanest available method.
    Tries XML first (cleanest), falls back to body_html if needed.
    """
    
    BASE_URL = "https://www.federalregister.gov/api/v1"
    
    def __init__(self, delay=0.5):
        self.delay = delay
        self.session = requests.Session()
    
    def get_metadata(self, document_number):
        """Fetch JSON metadata for a document"""
        url = f"{self.BASE_URL}/documents/{document_number}.json"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching metadata for {document_number}: {e}")
            return None
    
    def get_text_from_xml(self, xml_url):
        """
        Parse XML to extract clean text.
        XML is the cleanest format - no HTML encoding issues.
        """
        try:
            response = self.session.get(xml_url, timeout=60)
            response.raise_for_status()
            
            # parse XML
            root = ET.fromstring(response.content)
            
            # extract all text content
            text_parts = []
            
            def extract_text(element):
                """Recursively extract text from XML element"""
                if element.text:
                    text_parts.append(element.text.strip())
                for child in element:
                    extract_text(child)
                    if child.tail:
                        text_parts.append(child.tail.strip())
            
            extract_text(root)
            
            # join with newlines and clean up
            full_text = '\n'.join(part for part in text_parts if part)
            
            return full_text
            
        except Exception as e:
            print(f"  XML parsing failed: {e}")
            return None
    
    def get_text_from_html(self, html_url):
        """
        Parse body HTML to extract text.
        Fallback if XML fails.
        """
        try:
            response = self.session.get(html_url, timeout=60)
            response.raise_for_status()
            
            # parse HTML and extract text
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # get text from <pre> if it exists (common in FR format)
            pre_tag = soup.find('pre')
            if pre_tag:
                text = pre_tag.get_text(separator='\n')
            else:
                text = soup.get_text(separator='\n')
            
            # decode HTML entities
            text = html.unescape(text)
            
            # clean excessive whitespace
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            print(f"  HTML parsing failed: {e}")
            return None
    
    def get_text(self, document_number, retries=3):
        """
        Fetch clean text for a document.
        Tries XML first (cleanest), falls back to HTML.
        
        Returns dict with text and method used.
        """
        for attempt in range(retries):
            try:
                # get metadata (JSON) first
                metadata = self.get_metadata(document_number)
                if not metadata:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                    continue
                
                # try method 1: XML (cleanest, no HTML artifacts)
                if metadata.get('full_text_xml_url'):
                    text = self.get_text_from_xml(metadata['full_text_xml_url'])
                    if text:
                        time.sleep(self.delay)
                        return {
                            'text': text,
                            'method': 'xml',
                            'success': True
                        }
                
                # try method 2: body HTML (fallback)
                if metadata.get('body_html_url'):
                    text = self.get_text_from_html(metadata['body_html_url'])
                    if text:
                        time.sleep(self.delay)
                        return {
                            'text': text,
                            'method': 'body_html',
                            'success': True
                        }
                
                # try method 3: raw text URL (last resort)
                if metadata.get('raw_text_url'):
                    text = self.get_text_from_html(metadata['raw_text_url'])
                    if text:
                        time.sleep(self.delay)
                        return {
                            'text': text,
                            'method': 'raw_text',
                            'success': True
                        }
                
                # if we get here, all methods failed this attempt
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to fetch {document_number} after {retries} attempts: {e}")
                    return {'text': None, 'method': None, 'success': False}
                time.sleep(2 ** attempt)
        
        return {'text': None, 'method': None, 'success': False}
    
    def process_dataframe(self, df, output_file, checkpoint_freq=50):
        """
        Process entire dataframe and fetch clean text for all documents.
        """
        # initialize columns
        if 'full_text' not in df.columns:
            df['full_text'] = None
        if 'text_fetched' not in df.columns:
            df['text_fetched'] = False
        if 'fetch_method' not in df.columns:
            df['fetch_method'] = None
        
        total = len(df)
        fetched_count = df['text_fetched'].sum()
        
        print(f"Starting fetch: {fetched_count}/{total} already completed")
        print(f"Will try XML first (cleanest), fall back to HTML if needed\n")
        
        for idx, row in tqdm(df.iterrows(), total=total, desc="Fetching documents"):
            # skip if already fetched successfully
            if row['text_fetched']:
                continue
            
            doc_number = row['document_number']
            
            # fetch the text
            result = self.get_text(doc_number)
            
            if result['success']:
                df.at[idx, 'full_text'] = result['text']
                df.at[idx, 'text_fetched'] = True
                df.at[idx, 'fetch_method'] = result['method']
            else:
                df.at[idx, 'text_fetched'] = False
            
            # save checkpoint
            if (idx + 1) % checkpoint_freq == 0:
                df.to_csv(output_file, index=False)
                success_so_far = df.iloc[:idx+1]['text_fetched'].sum()
                print(f"\nCheckpoint: {idx + 1}/{total} processed, {success_so_far} successful")
        
        # final save
        df.to_csv(output_file, index=False)
        
        # summary statistics
        success_count = df['text_fetched'].sum()
        print(f"\n{'='*60}")
        print(f"Complete! Results saved to {output_file}")
        print(f"Successfully fetched: {success_count}/{total} ({100*success_count/total:.1f}%)")
        print(f"Failed: {total - success_count}")
        
        # show method breakdown
        if 'fetch_method' in df.columns:
            method_counts = df[df['text_fetched'] == True]['fetch_method'].value_counts()
            print(f"\nFetch method breakdown:")
            for method, count in method_counts.items():
                print(f"  {method}: {count}")
        
        return df


def inspect_samples(df, num_samples=2):
    """Show sample text to verify quality"""
    successful = df[df['text_fetched'] == True]
    
    if len(successful) == 0:
        print("No successful fetches to inspect")
        return
    
    samples = successful.sample(min(num_samples, len(successful)))
    
    for idx, row in samples.iterrows():
        print(f"\n{'='*70}")
        print(f"Document: {row['document_number']}")
        print(f"Method: {row.get('fetch_method', 'unknown')}")
        print(f"Title: {row.get('title', 'N/A')[:80]}...")
        print(f"Text length: {len(row['full_text'])} characters")
        print(f"\nFirst 400 characters:")
        print(row['full_text'][:400])
        print("\n[...]")


def main_command_line():
    """Run from command line"""
    if len(sys.argv) < 2:
        print("Usage: python federal_register_fetcher.py <input_csv>")
        print("Example: python federal_register_fetcher.py fr_tracking.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.csv', '_with_text.csv')
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, encoding='latin1')
    print(f"Found {len(df)} documents")
    
    # create fetcher
    fetcher = CleanFederalRegisterFetcher(delay=0.5)
    
    # process
    df_with_text = fetcher.process_dataframe(
        df, 
        output_file=output_file,
        checkpoint_freq=100
    )
    
    # show samples
    print("\n\nInspecting sample documents:")
    inspect_samples(df_with_text, num_samples=3)
    
    print(f"\n✓ Done! Results saved to: {output_file}")


# only run command-line version if executed directly (not imported)
if __name__ == "__main__":
    main_command_line()