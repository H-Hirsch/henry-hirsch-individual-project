"""
Enhanced False Negative Analysis:
1. Shows actual text of false negative rules
2. Compares class distribution in train vs test sets
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

print("="*70)
print("ENHANCED FALSE NEGATIVE ANALYSIS")
print("="*70)


print("\nLoading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

with open('test_results_all_categories.json', 'r') as f:
    results = json.load(f)

print(f"✓ Train set: {len(train_df):,} rules")
print(f"✓ Test set: {len(test_df):,} rules")



print("\n" + "="*70)
print("1. IDENTIFYING FALSE NEGATIVE RULES")
print("="*70)

categories_info = [
    ('significant', 'significant', 'extracted_text_significant'),
    ('economically_significant', 'econ_sig', 'extracted_text_econ'),
    ('Major', 'major', 'extracted_text_major')
]

all_false_negatives = {}

for category, short_name, text_column in categories_info:
    print(f"\n--- {category.upper()} ---")
    
    if category not in test_df.columns or text_column not in test_df.columns:
        print(f"Skipping - columns not found")
        continue
    
    # load model
    model_path = f'./bert_{short_name}_model'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # load threshold
        with open('test_results_all_categories.json', 'r') as f:
            threshold = results[category].get('threshold', 0.5)
        
        print(f"Loaded model with threshold: {threshold:.3f}")
        
        # prepare test data
        test_texts = test_df[text_column].fillna('').tolist()
        test_labels = test_df[category].tolist()
        
        test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # get predictions
        predictions_list = []
        probabilities_list = []
        
        with torch.no_grad():
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                inputs = {
                    'input_ids': item['input_ids'].unsqueeze(0),
                    'attention_mask': item['attention_mask'].unsqueeze(0)
                }
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                prob_positive = probs[0][1].item()
                pred = 1 if prob_positive >= threshold else 0
                
                predictions_list.append(pred)
                probabilities_list.append(prob_positive)
        
        # add to dataframe
        test_df[f'{category}_pred'] = predictions_list
        test_df[f'{category}_prob'] = probabilities_list
        
        # find false negatives
        fn_mask = (test_df[category] == 1) & (test_df[f'{category}_pred'] == 0)
        false_negatives = test_df[fn_mask].copy()
        
        print(f"False negatives found: {len(false_negatives)}")
        
        all_false_negatives[category] = false_negatives
        
    except Exception as e:
        print(f"Could not load model: {e}")
        continue



print("\n" + "="*70)
print("2. FALSE NEGATIVE TEXT ANALYSIS")
print("="*70)

fn_report = []

for category, fn_df in all_false_negatives.items():
    if len(fn_df) == 0:
        continue
    
    print(f"\n{'='*70}")
    print(f"{category.upper()} - FALSE NEGATIVES ({len(fn_df)} rules)")
    print(f"{'='*70}")
    
    fn_report.append(f"\n{'='*70}")
    fn_report.append(f"{category.upper()} - FALSE NEGATIVES ({len(fn_df)} rules)")
    fn_report.append(f"{'='*70}\n")
    
    # get text column
    text_col_map = {
        'significant': 'extracted_text_significant',
        'economically_significant': 'extracted_text_econ',
        'Major': 'extracted_text_major'
    }
    text_col = text_col_map.get(category, 'extracted_text_significant')
    
    for idx, (i, row) in enumerate(fn_df.iterrows(), 1):
        print(f"\n--- False Negative #{idx} ---")
        print(f"Document Number: {row.get('document_number', 'N/A')}")
        print(f"Title: {row.get('title', 'N/A')[:100]}...")
        print(f"Model Probability: {row.get(f'{category}_prob', 0):.3f} (threshold: {results[category].get('threshold', 0.5):.3f})")
        print(f"Publication Date: {row.get('publication_date', 'N/A')}")
        
        # show extracted text
        extracted_text = str(row.get(text_col, ''))
        print(f"\nExtracted Text ({len(extracted_text)} chars):")
        print(f"{extracted_text[:500]}...")
        
        # check for keywords
        keywords_found = []
        if category == 'significant':
            keywords = ['executive order 12866', 'oira', 'significant regulatory action']
        elif category == 'economically_significant':
            keywords = ['$100 million', '$200 million', 'economically significant', '3(f)(1)']
        else:  # major
            keywords = ['major rule', '5 u.s.c. 804', 'congressional review', 'submitted to congress']
        
        for kw in keywords:
            if kw.lower() in extracted_text.lower():
                keywords_found.append(kw)
        
        print(f"\nKeywords found: {keywords_found if keywords_found else 'None'}")
        
        # add to report
        fn_report.append(f"\n--- False Negative #{idx} ---")
        fn_report.append(f"Document: {row.get('document_number', 'N/A')}")
        fn_report.append(f"Title: {row.get('title', 'N/A')}")
        fn_report.append(f"Probability: {row.get(f'{category}_prob', 0):.3f}")
        fn_report.append(f"Text: {extracted_text[:300]}...")
        fn_report.append(f"Keywords: {keywords_found}\n")
        
        print("-" * 70)

# save detailed FN report
with open('false_negative_detailed_report.txt', 'w') as f:
    f.write('\n'.join(fn_report))

print(f"\n✓ Saved: false_negative_detailed_report.txt")


print("\n" + "="*70)
print("3. CREATING TRAIN VS TEST DISTRIBUTION COMPARISON")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

categories = ['significant', 'economically_significant', 'Major']
titles = ['Significant', 'Economically Significant', 'Major']

for idx, (category, title) in enumerate(zip(categories, titles)):
    if category not in train_df.columns or category not in test_df.columns:
        continue
    
    # calculate distributions
    train_pos = (train_df[category] == 1).sum()
    train_neg = (train_df[category] == 0).sum()
    test_pos = (test_df[category] == 1).sum()
    test_neg = (test_df[category] == 0).sum()
    
    # create grouped bar chart
    x = np.arange(2)
    width = 0.35
    
    train_counts = [train_neg, train_pos]
    test_counts = [test_neg, test_pos]
    
    bars1 = axes[idx].bar(x - width/2, train_counts, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = axes[idx].bar(x + width/2, test_counts, width, label='Test', color='#2ecc71', alpha=0.8)
    
    axes[idx].set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{title}', fontsize=14, fontweight='bold')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(['Negative', 'Positive'], fontsize=11)
    axes[idx].legend(fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)
    
    # add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height):,}',
                          ha='center', va='bottom', fontsize=9)
    
    # add percentage labels
    train_pct = 100 * train_pos / (train_pos + train_neg)
    test_pct = 100 * test_pos / (test_pos + test_neg)
    
    axes[idx].text(0.5, 0.95, 
                  f'Train: {train_pct:.1f}% positive\nTest: {test_pct:.1f}% positive',
                  transform=axes[idx].transAxes,
                  ha='center', va='top', fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('train_test_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_distribution_comparison.png")
plt.close()



print("\n" + "="*70)
print("4. CREATING DISTRIBUTION PIE CHARTS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (category, title) in enumerate(zip(categories, titles)):
    if category not in train_df.columns or category not in test_df.columns:
        continue
    
    # train distribution (top row)
    train_counts = [
        (train_df[category] == 0).sum(),
        (train_df[category] == 1).sum()
    ]
    
    axes[0, idx].pie(train_counts, labels=['Negative', 'Positive'],
                     autopct='%1.1f%%', startangle=90,
                     colors=['#95a5a6', '#3498db'])
    axes[0, idx].set_title(f'Train Set - {title}\n(n={len(train_df):,})',
                          fontsize=12, fontweight='bold')
    
    # test distribution (bottom row)
    test_counts = [
        (test_df[category] == 0).sum(),
        (test_df[category] == 1).sum()
    ]
    
    axes[1, idx].pie(test_counts, labels=['Negative', 'Positive'],
                     autopct='%1.1f%%', startangle=90,
                     colors=['#95a5a6', '#2ecc71'])
    axes[1, idx].set_title(f'Test Set - {title}\n(n={len(test_df):,})',
                          fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('train_test_pie_charts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_pie_charts.png")
plt.close()


print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

print("\nFiles created:")
print(" false_negative_detailed_report.txt - Full text of missed rules")
print("train_test_distribution_comparison.png - Bar chart comparison")
print("train_test_pie_charts.png - Pie chart distributions")

print("\n" + "="*70)
