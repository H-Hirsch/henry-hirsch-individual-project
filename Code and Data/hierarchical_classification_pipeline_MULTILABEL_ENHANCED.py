
# federal Register multi-label hierarchical classification

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import json
import os
from datetime import datetime


# helper funcs

def parse_publication_date(date_str):
    """Parse publication date to datetime"""
    try:
        return pd.to_datetime(date_str)
    except:
        return None


def get_significance_category(row):
    """
    Determine which significance category applies based on date
    Returns: 'econ_significant' or '3(f)(1)_significant' or None
    """
    pub_date = parse_publication_date(row.get('publication_date'))

    if pub_date is None:
        return None

    # temporal thresholds (for econ sig threhold)
    eo14094_start = datetime(2023, 4, 6)
    eo14094_end = datetime(2025, 1, 20)

    if eo14094_start <= pub_date <= eo14094_end:
        return '3(f)(1)_significant'  # $200M threshold period
    else:
        return 'econ_significant'  # $100M threshold period



# data prep

def load_and_prepare_data(input_file='fr_tracking_processed_enhanced.csv'):
    """Load data and prepare for multi-label modeling"""
    print("=" * 70)
    print("PHASE 1: DATA PREPARATION (MULTI-LABEL)")
    print("=" * 70)

    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file, encoding='latin1')
    print(f"✓ Loaded {len(df):,} total documents")

    if 'text_fetched' in df.columns:
        df = df[df['text_fetched'] == True].copy()
        print(f"✓ {len(df):,} documents with text fetched")

    # ensure all label columns exist and are numeric
    label_columns = ['significant', 'econ_significant', '3(f)(1) significant', 'Major']
    for col in label_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # combine econ_significant and 3(f)(1)_significant into single category
    df['economically_significant'] = (
            (df['econ_significant'] == 1) | (df['3(f)(1) significant'] == 1)
    ).astype(int)

    print(f"\nOriginal counts:")
    print(f"  econ_significant: {df['econ_significant'].sum()}")
    print(f"  3(f)(1)_significant: {df['3(f)(1) significant'].sum()}")
    print(f"  Combined economically_significant: {df['economically_significant'].sum()}")

    df_labeled = df[df['significant'].isin([0, 1])].copy()
    print(f"✓ {len(df_labeled):,} documents with valid labels")

    print(f"\n{'=' * 70}")
    print("CLASS DISTRIBUTION")
    print(f"{'=' * 70}")

    # show all labels including combined
    display_labels = ['significant', 'economically_significant', 'Major']

    for col in display_labels:
        if col in df_labeled.columns:
            counts = df_labeled[col].value_counts()
            print(f"\n{col}:")
            print(counts)
            if 1 in counts.index and counts[1] > 0:
                print(f"  Imbalance ratio: 1:{counts[0] / counts[1]:.1f}")

    # temporal analysis
    print(f"\n{'=' * 70}")
    print("TEMPORAL THRESHOLD ANALYSIS")
    print(f"{'=' * 70}")

    if 'publication_date' in df_labeled.columns:
        df_labeled['pub_date_parsed'] = df_labeled['publication_date'].apply(parse_publication_date)
        df_labeled['significance_category'] = df_labeled.apply(get_significance_category, axis=1)

        print("\nRules by temporal category:")
        if 'significance_category' in df_labeled.columns:
            print(df_labeled['significance_category'].value_counts())

            print("\nEconomically significant by period:")
            econ_period = df_labeled[df_labeled['significance_category'] == 'econ_significant']
            print(f"  $100M threshold periods: {len(econ_period)} rules")
            print(f"    - Labeled as econ_significant: {econ_period['econ_significant'].sum()}")

            print("\n3(f)(1) significant period (2023-04-06 to 2025-01-20):")
            f31_period = df_labeled[df_labeled['significance_category'] == '3(f)(1)_significant']
            print(f"  $200M threshold period: {len(f31_period)} rules")
            print(f"    - Labeled as 3(f)(1)_significant: {f31_period['3(f)(1) significant'].sum()}")

    print(f"\n{'=' * 70}")
    print("VALIDATING HIERARCHY")
    print(f"{'=' * 70}")

    # validate and fix hierarchy violations
    violations = []

    # if economically_significant, must be significant
    v1 = df_labeled[(df_labeled['economically_significant'] == 1) & (df_labeled['significant'] == 0)]
    if len(v1) > 0:
        violations.append(f"economically_significant without significant: {len(v1)}")
        df_labeled.loc[df_labeled['economically_significant'] == 1, 'significant'] = 1

    if violations:
        print(f"⚠️  WARNING: Found and fixed violations:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("✓ No hierarchy violations")

    print(f"\n{'=' * 70}")
    print("SPLITTING DATA")
    print(f"{'=' * 70}")

    train_df, temp_df = train_test_split(
        df_labeled,
        test_size=0.3,
        random_state=42,
        stratify=df_labeled['significant']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['significant']
    )

    print(f"Train: {len(train_df):,} ({len(train_df) / len(df_labeled) * 100:.1f}%)")
    print(f"Val:   {len(val_df):,} ({len(val_df) / len(df_labeled) * 100:.1f}%)")
    print(f"Test:  {len(test_df):,} ({len(test_df) / len(df_labeled) * 100:.1f}%)")

    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print(f"\n✓ Saved splits to train.csv, val.csv, test.csv")

    return train_df, val_df, test_df


# baseline models for all categories

def train_baseline_models(train_df, val_df):
    """Train baseline models for all 4 categories using category-specific text"""
    print(f"\n{'=' * 70}")
    print("PHASE 2: BASELINE MODELS (ALL CATEGORIES)")
    print("Using category-specific text extraction")
    print(f"{'=' * 70}")

    baseline_results = {}

    # map each category to its specific text column
    # economically_significant uses combined econ text (will include both $100M and $200M keywords)
    categories_config = {
        'significant': 'extracted_text_significant',
        'economically_significant': 'extracted_text_econ',  # uses econ text which has both thresholds
        'Major': 'extracted_text_major'
    }

    for category, text_column in categories_config.items():
        if category not in train_df.columns:
            print(f"Skipping {category} - column not found")
            continue

        if text_column not in train_df.columns:
            print(f"Skipping {category} - text column {text_column} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"Training baseline for: {category}")
        print(f"Using text column: {text_column}")
        print(f"{'=' * 70}")

        # get category-specific text
        X_train = train_df[text_column].fillna('')
        X_val = val_df[text_column].fillna('')
        y_train = train_df[category]
        y_val = val_df[category]

        # skip if no positive examples
        if y_train.sum() == 0:
            print(f"Skipping {category} - no positive examples in training set")
            continue

        # create TF-IDF features for this category
        print(f"Creating TF-IDF features...")
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            stop_words='english'
        )

        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)
        print(f"✓ Feature shape: {X_train_tfidf.shape}")

        # compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        model = LogisticRegression(
            class_weight=class_weight_dict,
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train_tfidf, y_train)

        y_pred_proba = model.predict_proba(X_val_tfidf)[:, 1]

        # optimize for F1
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        optimal_idx = f1_scores.argmax()
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        print(f"Threshold: {optimal_threshold:.3f}")
        print(f"Recall: {recall_score(y_val, y_pred):.3f}")
        print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.3f}")
        print(f"F1: {f1_score(y_val, y_pred):.3f}")

        baseline_results[category] = {
            'model': model,
            'tfidf': tfidf,
            'threshold': optimal_threshold,
            'text_column': text_column,
            'recall': recall_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred)
        }

    import joblib
    joblib.dump(baseline_results, 'baseline_models_all.pkl')
    print(f"\n✓ Saved all baseline models")

    return baseline_results


# BERT models for all categories

def train_bert_classifier(train_df, val_df, category, output_dir, text_column):
    """Train BERT model for a specific category using category-specific text"""
    print(f"\n{'=' * 70}")
    print(f"BERT CLASSIFIER: {category}")
    print(f"Using text column: {text_column}")
    print(f"{'=' * 70}")

    y_train = train_df[category]
    y_val = val_df[category]

    # skip if insufficient positive examples
    if y_train.sum() < 10:
        print(f"Skipping {category} - insufficient positive examples ({y_train.sum()})")
        return None, None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # use category-specific text column
    train_texts = train_df[text_column].fillna('').tolist()
    train_labels = y_train.tolist()
    val_texts = val_df[text_column].fillna('').tolist()
    val_labels = y_val.tolist()

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

    print(f"Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # moderate class weights
    pos_weight = 3.0
    neg_weight = 1.0

    print(f"Class weights - Negative: {neg_weight:.2f}, Positive: {pos_weight:.2f}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([neg_weight, pos_weight]).to(model.device)
            )
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        acc = accuracy_score(labels, predictions)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'./results_{output_dir}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs_{output_dir}',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    print(f"Training BERT model for {category}...")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    model_path = f'./bert_{output_dir}_model'
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✓ Saved model to {model_path}")

    # get validation results
    results = trainer.evaluate()
    print(f"\nValidation results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # optimize threshold for F1
    predictions = trainer.predict(val_dataset)
    y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    precisions, recalls, thresholds = precision_recall_curve(val_labels, y_pred_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    print(f"  Recall: {recalls[optimal_idx]:.3f}")
    print(f"  Precision: {precisions[optimal_idx]:.3f}")
    print(f"  F1: {f1_scores[optimal_idx]:.3f}")

    return trainer, tokenizer, optimal_threshold


def train_all_bert_models(train_df, val_df):
    """Train BERT models for all categories using category-specific text"""
    print(f"\n{'=' * 70}")
    print("PHASE 3: BERT MODELS (ALL CATEGORIES)")
    print("Using category-specific text extraction")
    print(f"{'=' * 70}")

    bert_models = {}

    # map each category to its specific text column
    # economically_significant uses econ text (has both $100M and $200M keywords)
    categories = [
        ('significant', 'significant', 'extracted_text_significant'),
        ('economically_significant', 'econ_sig', 'extracted_text_econ'),
        ('Major', 'major', 'extracted_text_major')
    ]

    for category, short_name, text_column in categories:
        if category not in train_df.columns:
            print(f"Column {category} not found, skipping...")
            continue

        if text_column not in train_df.columns:
            print(f"Text column {text_column} not found, skipping...")
            continue

        trainer, tokenizer, threshold = train_bert_classifier(
            train_df, val_df, category, short_name, text_column
        )

        if trainer is not None:
            bert_models[category] = {
                'trainer': trainer,
                'tokenizer': tokenizer,
                'threshold': threshold,
                'short_name': short_name,
                'text_column': text_column
            }

    return bert_models


# evalutaion

def evaluate_all_models(test_df, bert_models):
    """Evaluate all BERT models on test set using category-specific text"""
    print(f"\n{'=' * 70}")
    print("PHASE 4: EVALUATION (ALL CATEGORIES)")
    print(f"{'=' * 70}")

    all_results = {}

    for category, model_info in bert_models.items():
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {category}")
        print(f"{'=' * 70}")

        trainer = model_info['trainer']
        tokenizer = model_info['tokenizer']
        threshold = model_info['threshold']
        text_column = model_info['text_column']

        # use category-specific text column
        test_texts = test_df[text_column].fillna('').tolist()
        test_labels = test_df[category].tolist()

        test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        predictions = trainer.predict(test_dataset)
        y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_true = test_labels

        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\nTest Results:")
        print(f"  Recall: {recall:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  F1: {f1:.3f}")

        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        all_results[category] = {
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'threshold': float(threshold)
        }

    with open('test_results_all_categories.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved results for all categories")

    return all_results


# main pipeline

def main():
    """Run complete multi-label hierarchical classification"""
    print("=" * 70)
    print("FEDERAL REGISTER MULTI-LABEL HIERARCHICAL CLASSIFICATION")
    print("=" * 70)
    print("\nClassifying 3 categories:")
    print("  1. Significant")
    print("  2. Economically Significant (combined econ + 3(f)(1) thresholds)")
    print("  3. Major")
    print("=" * 70)

    input("\nPress Enter to start...")

    train_df, val_df, test_df = load_and_prepare_data('fr_tracking_processed_enhanced.csv')

    baseline_results = train_baseline_models(train_df, val_df)

    bert_models = train_all_bert_models(train_df, val_df)

    all_results = evaluate_all_models(test_df, bert_models)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE!")
    print(f"{'=' * 70}")

    print(f"\nResults summary:")
    for category, results in all_results.items():
        print(f"\n{category}:")
        print(f"  Recall: {results['recall']:.1%}")
        print(f"  Precision: {results['precision']:.1%}")
        print(f"  F1: {results['f1']:.3f}")


if __name__ == "__main__":
    main()