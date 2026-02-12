# Medical entity extraction

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
import logging
from typing import List, Dict

from config import CONFIG
from schemas import EntityFields

logger = logging.getLogger(__name__)


class MedicalNERExtractor:
    
    def __init__(self):
        logger.info(f"Loading NER model: {CONFIG['ner']['model_name']}")
        
        try:

            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['ner']['model_name'])
            self.model = AutoModelForTokenClassification.from_pretrained(CONFIG['ner']['model_name'])
            

            device = 0 if torch.cuda.is_available() else -1
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=device
            )
            

            self.category_mappings = CONFIG['entity_categories']
            self.default_category = CONFIG['entity_default_category']
            
            logger.info(f"[OK] NER model loaded (device: {'GPU' if device == 0 else 'CPU'})")
            logger.info(f"[OK] Entity categories: {list(self.category_mappings.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict]:
        try:
            logger.info("Extracting medical entities...")
            

            max_chars = CONFIG['text_limits']['ner_max_chars']
            if len(text) > max_chars:
                logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
                text = text[:max_chars]
            

            raw_entities = self.ner_pipeline(text)
            

            threshold = CONFIG['ner']['confidence_threshold']
            min_length = CONFIG['text_limits']['min_entity_length']
            
            formatted = []
            seen_texts = set()
            
            for ent in raw_entities:
                if ent['score'] < threshold:
                    continue
                
                text_clean = ent['word'].strip()
                
                if len(text_clean) <= min_length:
                    continue
                
                if text_clean.lower() in seen_texts:
                    continue
                
                seen_texts.add(text_clean.lower())
                entity = EntityFields.create(
                    text=text_clean,
                    entity_type=ent['entity_group'],
                    confidence=ent['score'],
                    start=ent.get('start', 0),
                    end=ent.get('end', 0)
                )
                formatted.append(entity)
            
            logger.info(f"[OK] Extracted {len(formatted)} high-confidence entities "
                       f"(from {len(raw_entities)} total, threshold={threshold})")
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def categorize_entities(self, entities: List[Dict]) -> Dict[str, List[Dict]]:

        categories = {cat: [] for cat in self.category_mappings.keys()}
        categories[self.default_category] = []
        
        for entity in entities:
            entity_type = entity[EntityFields.TYPE].upper()
            

            matched = False
            for category, keywords in self.category_mappings.items():

                if any(keyword.upper() in entity_type for keyword in keywords):
                    categories[category].append(entity)
                    matched = True
                    break
            

            if not matched:
                categories[self.default_category].append(entity)
        

        summary_parts = []
        for cat, entities_list in categories.items():
            if entities_list:
                summary_parts.append(f"{len(entities_list)} {cat}")
        
        if summary_parts:
            logger.info(f"[OK] Categorized: {', '.join(summary_parts)}")
        else:
            logger.warning("[WARNING] No entities categorized")
        
        return categories
