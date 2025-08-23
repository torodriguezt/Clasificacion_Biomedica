"""
Text augmentation utilities for medical classification.

This module provides text augmentation techniques specifically designed
for medical and biomedical text data to improve model robustness.
"""

import random
import re
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd


class MedicalTextAugmenter:
    """Text augmentation for medical classification tasks."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Medical abbreviations and their expansions
        self.medical_abbreviations = {
            'pt': 'patient',
            'pts': 'patients',
            'hx': 'history',
            'dx': 'diagnosis',
            'rx': 'treatment',
            'sx': 'symptoms',
            'tx': 'treatment',
            'fu': 'follow up',
            'f/u': 'follow up',
            'w/': 'with',
            'w/o': 'without',
            'b/l': 'bilateral',
            'r/o': 'rule out',
            'c/o': 'complains of',
            's/p': 'status post',
            'h/o': 'history of',
            'p/w': 'presents with',
            'p.o.': 'by mouth',
            'i.v.': 'intravenous',
            'i.m.': 'intramuscular',
            'b.i.d.': 'twice daily',
            't.i.d.': 'three times daily',
            'q.i.d.': 'four times daily',
            'prn': 'as needed',
            'stat': 'immediately',
            'npo': 'nothing by mouth',
            'sob': 'shortness of breath',
            'dob': 'difficulty breathing',
            'loc': 'loss of consciousness',
            'n/v': 'nausea and vomiting',
            'abd': 'abdominal',
            'ext': 'extremity',
            'bilat': 'bilateral',
            'neg': 'negative',
            'pos': 'positive',
            'wnl': 'within normal limits',
            'unremarkable': 'normal',
            'remarkable': 'abnormal'
        }
        
        # Medical synonyms for replacement
        self.medical_synonyms = {
            'pain': ['discomfort', 'ache', 'soreness', 'tenderness'],
            'severe': ['intense', 'acute', 'extreme', 'significant'],
            'mild': ['slight', 'minor', 'minimal', 'light'],
            'chronic': ['persistent', 'long-standing', 'ongoing', 'longterm'],
            'acute': ['sudden', 'rapid', 'immediate', 'sharp'],
            'patient': ['individual', 'case', 'subject'],
            'procedure': ['intervention', 'operation', 'treatment'],
            'medication': ['drug', 'medicine', 'therapeutic agent'],
            'symptom': ['sign', 'manifestation', 'indication'],
            'diagnosis': ['condition', 'disorder', 'disease'],
            'treatment': ['therapy', 'intervention', 'management'],
            'examination': ['assessment', 'evaluation', 'inspection'],
            'normal': ['typical', 'standard', 'regular', 'usual'],
            'abnormal': ['atypical', 'irregular', 'unusual', 'deviant'],
            'increase': ['elevation', 'rise', 'escalation'],
            'decrease': ['reduction', 'decline', 'drop'],
            'improve': ['enhance', 'better', 'ameliorate'],
            'worsen': ['deteriorate', 'decline', 'aggravate']
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded abbreviations
        """
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Clean word for matching
            clean_word = re.sub(r'[^\w.]', '', word.lower())
            
            if clean_word in self.medical_abbreviations:
                # Keep original case pattern
                if word.isupper():
                    replacement = self.medical_abbreviations[clean_word].upper()
                elif word[0].isupper():
                    replacement = self.medical_abbreviations[clean_word].capitalize()
                else:
                    replacement = self.medical_abbreviations[clean_word]
                
                # Preserve punctuation
                punctuation = re.sub(r'[\w.]', '', word)
                expanded_words.append(replacement + punctuation)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def contract_abbreviations(self, text: str) -> str:
        """
        Contract common medical terms to abbreviations.
        
        Args:
            text: Input text
            
        Returns:
            Text with contracted terms
        """
        # Reverse mapping for contraction
        contractions = {v: k for k, v in self.medical_abbreviations.items()}
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_terms = sorted(contractions.keys(), key=len, reverse=True)
        
        text_lower = text.lower()
        for term in sorted_terms:
            if term in text_lower:
                # Use regex for word boundary matching
                pattern = r'\b' + re.escape(term) + r'\b'
                text = re.sub(pattern, contractions[term], text, flags=re.IGNORECASE)
        
        return text
    
    def synonym_replacement(self, text: str, probability: float = 0.2) -> str:
        """
        Replace words with medical synonyms.
        
        Args:
            text: Input text
            probability: Probability of replacing each word
            
        Returns:
            Text with synonym replacements
        """
        words = text.split()
        augmented_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if (clean_word in self.medical_synonyms and 
                random.random() < probability):
                
                synonyms = self.medical_synonyms[clean_word]
                replacement = random.choice(synonyms)
                
                # Preserve original case and punctuation
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                punctuation = re.sub(r'[\w]', '', word)
                augmented_words.append(replacement + punctuation)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert medical filler words.
        
        Args:
            text: Input text
            n: Number of words to insert
            
        Returns:
            Text with random insertions
        """
        filler_words = [
            'notably', 'particularly', 'specifically', 'especially',
            'additionally', 'furthermore', 'moreover', 'also',
            'currently', 'presently', 'recently', 'previously',
            'clinically', 'medically', 'typically', 'generally'
        ]
        
        words = text.split()
        
        for _ in range(n):
            if len(words) == 0:
                break
                
            filler = random.choice(filler_words)
            position = random.randint(0, len(words))
            words.insert(position, filler)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap adjacent words.
        
        Args:
            text: Input text
            n: Number of swaps to perform
            
        Returns:
            Text with random swaps
        """
        words = text.split()
        
        for _ in range(n):
            if len(words) < 2:
                break
                
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, probability: float = 0.1) -> str:
        """
        Randomly delete words (excluding important medical terms).
        
        Args:
            text: Input text
            probability: Probability of deleting each word
            
        Returns:
            Text with random deletions
        """
        # Important medical terms that should not be deleted
        protected_terms = {
            'patient', 'diagnosis', 'treatment', 'medication', 'surgery',
            'pain', 'symptom', 'condition', 'disease', 'disorder',
            'acute', 'chronic', 'severe', 'mild', 'normal', 'abnormal',
            'positive', 'negative', 'history', 'examination', 'procedure'
        }
        
        words = text.split()
        remaining_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Keep protected terms and random selection of others
            if (clean_word in protected_terms or 
                random.random() > probability):
                remaining_words.append(word)
        
        # Ensure we don't delete too much
        if len(remaining_words) < len(words) * 0.7:
            return text
        
        return ' '.join(remaining_words)
    
    def back_translation_simulation(self, text: str) -> str:
        """
        Simulate back-translation effects by minor rephrasing.
        
        Args:
            text: Input text
            
        Returns:
            Text with simulated back-translation changes
        """
        # Simple transformations that simulate back-translation
        transformations = [
            (r'\bthe patient\b', 'this patient'),
            (r'\ba patient\b', 'one patient'),
            (r'\bwas diagnosed\b', 'received a diagnosis'),
            (r'\bshowed\b', 'demonstrated'),
            (r'\bfound\b', 'discovered'),
            (r'\bsaid\b', 'reported'),
            (r'\bhad\b', 'experienced'),
            (r'\bgave\b', 'provided'),
            (r'\bgot\b', 'received'),
            (r'\bwent\b', 'proceeded'),
            (r'\bcame\b', 'arrived'),
            (r'\bleft\b', 'departed')
        ]
        
        result = text
        for pattern, replacement in transformations:
            if random.random() < 0.3:  # 30% chance for each transformation
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def augment_text(
        self,
        text: str,
        techniques: List[str] = None,
        num_augmented: int = 1
    ) -> List[str]:
        """
        Apply multiple augmentation techniques to generate variants.
        
        Args:
            text: Input text
            techniques: List of techniques to use
            num_augmented: Number of augmented versions to generate
            
        Returns:
            List of augmented text variants
        """
        if techniques is None:
            techniques = [
                'synonym_replacement',
                'expand_abbreviations',
                'contract_abbreviations',
                'random_insertion',
                'back_translation_simulation'
            ]
        
        augmented_texts = []
        
        for _ in range(num_augmented):
            current_text = text
            
            # Randomly select and apply techniques
            selected_techniques = random.sample(
                techniques,
                k=random.randint(1, min(3, len(techniques)))
            )
            
            for technique in selected_techniques:
                if technique == 'synonym_replacement':
                    current_text = self.synonym_replacement(current_text)
                elif technique == 'expand_abbreviations':
                    current_text = self.expand_abbreviations(current_text)
                elif technique == 'contract_abbreviations':
                    current_text = self.contract_abbreviations(current_text)
                elif technique == 'random_insertion':
                    current_text = self.random_insertion(current_text)
                elif technique == 'random_swap':
                    current_text = self.random_swap(current_text)
                elif technique == 'random_deletion':
                    current_text = self.random_deletion(current_text)
                elif technique == 'back_translation_simulation':
                    current_text = self.back_translation_simulation(current_text)
            
            augmented_texts.append(current_text)
        
        return augmented_texts
    
    def augment_dataset(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_columns: List[str],
        augmentation_ratio: float = 0.2,
        techniques: List[str] = None
    ) -> pd.DataFrame:
        """
        Augment entire dataset with balanced augmentation.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_columns: Names of label columns
            augmentation_ratio: Ratio of augmented samples to add
            techniques: Augmentation techniques to use
            
        Returns:
            Augmented dataframe
        """
        if techniques is None:
            techniques = ['synonym_replacement', 'expand_abbreviations']
        
        # Calculate number of samples to augment
        num_to_augment = int(len(df) * augmentation_ratio)
        
        # Prioritize minority classes for augmentation
        label_sums = df[label_columns].sum()
        minority_threshold = label_sums.mean()
        
        augmented_rows = []
        augmented_count = 0
        
        for idx, row in df.iterrows():
            if augmented_count >= num_to_augment:
                break
            
            # Check if this sample has minority class labels
            has_minority_label = any(
                row[col] == 1 and label_sums[col] < minority_threshold
                for col in label_columns
            )
            
            if has_minority_label or random.random() < 0.1:  # 10% random augmentation
                original_text = row[text_column]
                augmented_texts = self.augment_text(
                    original_text,
                    techniques=techniques,
                    num_augmented=1
                )
                
                for aug_text in augmented_texts:
                    new_row = row.copy()
                    new_row[text_column] = aug_text
                    augmented_rows.append(new_row)
                    augmented_count += 1
        
        # Combine original and augmented data
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            return pd.concat([df, augmented_df], ignore_index=True)
        else:
            return df


def apply_noise_injection(text: str, noise_level: float = 0.1) -> str:
    """
    Inject character-level noise to simulate OCR errors or typos.
    
    Args:
        text: Input text
        noise_level: Probability of character modification
        
    Returns:
        Text with injected noise
    """
    chars = list(text)
    
    for i in range(len(chars)):
        if random.random() < noise_level and chars[i].isalpha():
            # Character substitution with similar characters
            similar_chars = {
                'a': ['o', 'e'], 'e': ['a', 'i'], 'i': ['e', 'l'],
                'o': ['a', 'u'], 'u': ['o', 'v'], 'l': ['i', '1'],
                'c': ['o'], 'g': ['q'], 'm': ['n'], 'n': ['m'],
                'p': ['b'], 'b': ['p'], 'd': ['b'], 'q': ['g']
            }
            
            char_lower = chars[i].lower()
            if char_lower in similar_chars:
                replacement = random.choice(similar_chars[char_lower])
                if chars[i].isupper():
                    replacement = replacement.upper()
                chars[i] = replacement
    
    return ''.join(chars)


def create_medical_templates() -> Dict[str, List[str]]:
    """
    Create templates for generating synthetic medical text.
    
    Returns:
        Dictionary of medical text templates by category
    """
    templates = {
        'chief_complaint': [
            "Patient presents with {symptom} for {duration}",
            "{age} year old {gender} complaining of {symptom}",
            "Chief complaint: {symptom} since {timeframe}",
            "Patient reports {symptom} that started {timeframe}"
        ],
        'history': [
            "Patient has a history of {condition}",
            "Past medical history significant for {condition}",
            "History notable for {condition}",
            "Previously diagnosed with {condition}"
        ],
        'examination': [
            "Physical examination reveals {finding}",
            "On examination, patient shows {finding}",
            "Clinical findings include {finding}",
            "Examination notable for {finding}"
        ],
        'assessment': [
            "Assessment: {diagnosis}",
            "Impression: {diagnosis}",
            "Diagnosis: {diagnosis}",
            "Clinical diagnosis of {diagnosis}"
        ],
        'plan': [
            "Plan: {treatment}",
            "Treatment plan includes {treatment}",
            "Recommend {treatment}",
            "Will proceed with {treatment}"
        ]
    }
    
    return templates
