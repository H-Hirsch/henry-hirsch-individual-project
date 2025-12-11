"""
federal register enhanced preprocessing

extracts different sentences optimized for each classification task:

input:  fr_tracking_with_text.csv
output: fr_tracking_processed_enhanced.csv
"""

import pandas as pd
import re
import nltk
from tqdm import tqdm

# download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK sentence tokenizer...")
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize


# category-specific sentence extraction functions

def extract_sentences_weighted(text, keyword_weights, max_chars=2000):
    """
    extract sentences containing keywords with importance weighting.
    generic function used by all category-specific extractors.
    """
    if not text or pd.isna(text) or len(str(text)) < 100:
        return str(text) if text else ""

    text = str(text)

    # clean
    text = re.sub(r'\[\[Page \d+\]\]', '', text)
    text = re.sub(r'\n+', ' ', text)

    # tokenize
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = re.split(r'[.!?]+\s+', text)

    # score each sentence
    sentence_scores = []

    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        score = 0

        # ddd weights for matching keywords
        for keyword, weight in keyword_weights.items():
            if keyword in sentence_lower:
                score += weight

        # bonus for early sentences
        if i < 20:
            score += (20 - i) * 0.5

        sentence_scores.append((score, i, sentence))

    # sort by score
    sentence_scores.sort(reverse=True)

    # select top sentences
    selected_sentences = []
    total_chars = 0

    for score, idx, sentence in sentence_scores:
        if score == 0:
            continue

        if total_chars + len(sentence) > max_chars:
            break

        selected_sentences.append((idx, sentence))
        total_chars += len(sentence)

    # sort by original order
    selected_sentences.sort(key=lambda x: x[0])

    if selected_sentences:
        extracted_text = ' '.join([sent for idx, sent in selected_sentences])
        return extracted_text.strip()

    # fallback
    first_15 = ' '.join(sentences[:15])
    return first_15[:2000] if len(first_15) > 2000 else first_15


def extract_significant_text(text):
    """
    Extract text for SIGNIFICANT classification.
    Focus: EO 12866, OIRA review, "significant regulatory action" language
    MINIMAL OVERLAP with other categories
    """
    keyword_weights = {
        # executive orders - UNIQUE to significant
        'executive order 12866': 10,
        'e.o. 12866': 10,
        'executive order 13563': 8,
        'e.o. 13563': 8,
        'section 3(f)': 9,

        # significance statements - UNIQUE
        'significant regulatory action': 10,
        'not a significant regulatory action': 10,
        'is a significant regulatory action': 10,
        'not a significant': 9,
        'is not a significant': 9,
        'is a significant': 9,

        # OIRA review - UNIQUE to significant
        'oira': 9,
        'oira review': 10,
        'reviewed by oira': 10,
        'office of information and regulatory affairs': 9,
        'office of management and budget': 7,
        'omb review': 8,

        # section 3(f)(2) - Agency coordination - UNIQUE
        'serious inconsistency': 8,
        'interfere with an action': 8,
        'another agency': 6,

        # section 3(f)(3) - Budgetary impacts - UNIQUE
        'materially alter': 7,
        'budgetary impact': 8,
        'entitlements': 6,
        'user fees': 6,

        # section 3(f)(4) - Novel issues - UNIQUE
        'novel legal': 8,
        'novel policy': 8,
        "president's priorities": 8,

        # general regulatory terms - low weight
        'regulatory flexibility act': 4,
        'unfunded mandates': 4,
    }

    return extract_sentences_weighted(text, keyword_weights, max_chars=2000)


def extract_econ_significant_text(text):
    """
    Extract text for ECONOMICALLY SIGNIFICANT classification.
    Focus: Dollar thresholds ($100M/$200M) and "economically significant" phrase
    MINIMAL OVERLAP - avoid general economic terms
    """
    keyword_weights = {
        # dollar thresholds - UNIQUE identifiers
        '$100 million': 10,
        '100 million': 9,
        '$100m': 10,
        '$100 m': 9,
        '$200 million': 10,
        '200 million': 9,
        '$200m': 10,
        '$200 m': 9,

        # "economically significant" phrase - UNIQUE
        'economically significant': 10,
        'not economically significant': 10,
        'is economically significant': 10,
        'economic significance': 9,

        # section 3(f)(1) references - UNIQUE
        'section 3(f)(1)': 10,
        '3(f)(1)': 10,
        'subsection 3(f)(1)': 9,

        # "annual effect" with dollar context - somewhat unique
        'annual effect on the economy': 8,
        'annual effect of': 7,

        # executive orders - relevant but not unique
        'executive order 12866': 6,
        'executive order 14094': 8,
        'e.o. 14094': 8,

        # determination language
        'determined to be economically': 8,
        'determination as economically': 7,
    }

    return extract_sentences_weighted(text, keyword_weights, max_chars=2000)


def extract_3f1_significant_text(text):
    """
    Extract text for 3(f)(1) SIGNIFICANT classification.
    Focus: $200M threshold (2023-2025 period), section 3(f)(1)
    """
    keyword_weights = {
        # dollar thresholds (highest priority)
        '$200 million': 10,
        '200 million': 9,
        '$200m': 10,
        '200m': 9,

        # section references
        'section 3(f)(1)': 10,
        '3(f)(1)': 10,
        '3(f)(1) significant': 10,

        # related EOs
        'executive order 14094': 10,
        'e.o. 14094': 10,
        'executive order 14148': 9,
        'e.o. 14148': 9,

        # economic impact (similar to econ_significant)
        'economically significant': 9,
        'economic significance': 8,
        'annual effect on the economy': 7,
        'effect on the economy': 6,
        'material way': 6,

        # base references
        'executive order 12866': 7,
        'e.o. 12866': 7,
    }

    return extract_sentences_weighted(text, keyword_weights, max_chars=2000)


def extract_major_text(text):
    """
    Extract text for MAJOR rule classification.
    Focus: "major rule" phrase, CRA (5 USC 804), congressional submission
    MINIMAL OVERLAP - avoid economic terms that overlap with econ_significant
    """
    keyword_weights = {
        # "major rule" phrase - UNIQUE identifier
        'major rule': 10,
        'major rules': 10,
        'not a major rule': 10,
        'is not a major rule': 10,
        'is a major rule': 10,
        'not a major': 9,
        'is a major': 9,

        # CRA legal citations - UNIQUE
        '5 u.s.c. 804': 10,
        '5 usc 804': 10,
        'section 804': 9,
        '5 u.s.c. 801': 9,
        '5 u.s.c. 802': 8,
        '5 usc 801': 9,

        # Congressional Review Act - UNIQUE
        'congressional review act': 10,
        'congressional review': 8,
        'cra': 7,  # Lower weight - ambiguous acronym

        # congressional submission - UNIQUE to major
        'submitted to congress': 10,
        'submit to congress': 10,
        'submitted to the congress': 10,
        'transmit to congress': 9,
        'transmission to congress': 9,

        # GAO review - UNIQUE to major
        'comptroller general': 10,
        'government accountability office': 9,
        'gao report': 9,
        'report to congress': 8,

        # effective date delays - UNIQUE to major
        'delayed effective date': 9,
        '60 days after': 8,
        '60-day period': 8,
        'delay the effective date': 9,
        'take effect on': 7,

        # major determination language
        'determined to be a major': 9,
        'determined not to be a major': 9,
        'determination as a major': 8,
        'qualifies as a major': 8,
        'does not qualify as a major': 8,
    }

    return extract_sentences_weighted(text, keyword_weights, max_chars=2000)


# label fixing function

def fix_labels(df):
    """Fix label columns: replace "." with 0 and convert to integers"""
    print(f"\n{'=' * 70}")
    print("FIXING LABEL VALUES")
    print(f"{'=' * 70}")

    label_columns = ['significant', 'econ_significant', '3(f)(1) significant', 'Major']

    for col in label_columns:
        if col in df.columns:
            dots_before = (df[col] == '.').sum()

            # replace dots with 0
            df[col] = df[col].replace('.', 0)

            # convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

            if dots_before > 0:
                print(f"  {col}: {dots_before} dots → 0")

    return df


# main preprocessing function

def main():
    """Complete preprocessing with category-specific extraction"""
    input_file = 'fr_tracking_with_text.csv'
    output_file = 'fr_tracking_processed_enhanced.csv'

    print("=" * 70)
    print("FEDERAL REGISTER ENHANCED PREPROCESSING")
    print("Category-Specific Sentence Extraction")
    print("=" * 70)

    # load data
    print(f"\nStep 1: Loading {input_file}...")
    try:
        df = pd.read_csv(input_file, encoding='latin1')
        print(f"✓ Loaded {len(df):,} documents")
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found")
        return

    # filter to successfully fetched
    if 'text_fetched' in df.columns:
        original_count = len(df)
        df = df[df['text_fetched'] == True].copy()
        print(f"✓ {len(df):,} documents with text successfully fetched")
        print(f"  (Skipped {original_count - len(df):,} documents without text)")

    # fix labels
    df = fix_labels(df)

    # extract sentences for each category
    print(f"\n{'=' * 70}")
    print("EXTRACTING CATEGORY-SPECIFIC TEXT")
    print(f"{'=' * 70}")

    print(f"\nExtracting from {len(df):,} documents...")
    print("This will take 15-20 minutes...")

    print("\n1. Extracting for SIGNIFICANT classification...")
    tqdm.pandas(desc="Significant")
    df['extracted_text_significant'] = df['full_text'].progress_apply(extract_significant_text)

    print("\n2. Extracting for ECONOMICALLY SIGNIFICANT classification...")
    tqdm.pandas(desc="Econ Sig")
    df['extracted_text_econ'] = df['full_text'].progress_apply(extract_econ_significant_text)

    print("\n3. Extracting for 3(f)(1) SIGNIFICANT classification...")
    tqdm.pandas(desc="3(f)(1) Sig")
    df['extracted_text_3f1'] = df['full_text'].progress_apply(extract_3f1_significant_text)

    print("\n4. Extracting for MAJOR rule classification...")
    tqdm.pandas(desc="Major")
    df['extracted_text_major'] = df['full_text'].progress_apply(extract_major_text)

    print(f"\n✓ Extraction complete")

    # analysis
    print(f"\n{'=' * 70}")
    print("QUALITY ANALYSIS")
    print(f"{'=' * 70}")

    original_length = df['full_text'].str.len().mean()

    print(f"\nText length statistics:")
    print(f"  Original:               {original_length:>10,.0f} chars")
    print(f"  Significant extraction: {df['extracted_text_significant'].str.len().mean():>10,.0f} chars")
    print(f"  Econ extraction:        {df['extracted_text_econ'].str.len().mean():>10,.0f} chars")
    print(f"  3(f)(1) extraction:     {df['extracted_text_3f1'].str.len().mean():>10,.0f} chars")
    print(f"  Major extraction:       {df['extracted_text_major'].str.len().mean():>10,.0f} chars")

    # BERT compatibility
    print(f"\nBERT compatibility (512 tokens ≈ 2000 chars):")
    for col in ['extracted_text_significant', 'extracted_text_econ', 'extracted_text_3f1', 'extracted_text_major']:
        fits = (df[col].str.len() <= 2000).sum()
        print(f"  {col:30s}: {fits:>5,}/{len(df):,} ({100 * fits / len(df):>5.1f}%)")

    # keyword presence by category
    print(f"\n{'=' * 70}")
    print("KEYWORD PRESENCE BY CATEGORY")
    print(f"{'=' * 70}")

    print(f"\nSignificant extraction:")
    print(
        f"  'executive order 12866': {df['extracted_text_significant'].str.contains('executive order 12866', case=False, na=False).sum():,}")
    print(
        f"  'significant': {df['extracted_text_significant'].str.contains('significant', case=False, na=False).sum():,}")

    print(f"\nEcon significant extraction:")
    print(
        f"  '$100 million': {df['extracted_text_econ'].str.contains(r'\$100 million', case=False, na=False, regex=True).sum():,}")
    print(
        f"  'economically significant': {df['extracted_text_econ'].str.contains('economically significant', case=False, na=False).sum():,}")

    print(f"\n3(f)(1) significant extraction:")
    print(
        f"  '$200 million': {df['extracted_text_3f1'].str.contains(r'\$200 million', case=False, na=False, regex=True).sum():,}")
    print(
        f"  '3(f)(1)': {df['extracted_text_3f1'].str.contains(r'3\(f\)\(1\)', case=False, na=False, regex=True).sum():,}")

    print(f"\nMajor extraction:")
    print(f"  'major rule': {df['extracted_text_major'].str.contains('major rule', case=False, na=False).sum():,}")
    print(f"  '5 u.s.c. 804': {df['extracted_text_major'].str.contains('5 u.s.c. 804', case=False, na=False).sum():,}")

    # show samples
    print(f"\n{'=' * 70}")
    print("SAMPLE EXTRACTIONS")
    print(f"{'=' * 70}")

    sample = df.sample(1).iloc[0]

    print(f"\nDocument: {sample.get('document_number', 'N/A')}")
    print(f"Significant: {sample.get('significant', 'N/A')}")
    print(f"Original length: {len(str(sample['full_text'])):,} chars")

    print(f"\nSignificant extraction ({len(str(sample['extracted_text_significant'])):,} chars):")
    print(f"  {str(sample['extracted_text_significant'])[:200]}...")

    print(f"\nEcon extraction ({len(str(sample['extracted_text_econ'])):,} chars):")
    print(f"  {str(sample['extracted_text_econ'])[:200]}...")

    # save
    print(f"\n{'=' * 70}")
    print("SAVING")
    print(f"{'=' * 70}")

    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  New columns: extracted_text_significant, extracted_text_econ,")
    print(f"               extracted_text_3f1, extracted_text_major")

    # final verification
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print(f"{'=' * 70}")

    valid_labels = df['significant'].isin([0, 1]).sum()
    print(f"\nDocuments with valid labels: {valid_labels:,}/{len(df):,}")

    print(f"\n{'=' * 70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()