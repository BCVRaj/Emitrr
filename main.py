# Main pipeline for medical transcript analysis

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

from config import CONFIG
from schemas import (
    OutputFields,
    MedicalSummaryFields,
    SentimentIntentFields,
    SOAPFields
)
from src.preprocessing import TranscriptPreprocessor
from src.ner_extractor import MedicalNERExtractor
from src.llm_extractor import LLMExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MedicalTranscriptPipeline:
    
    def __init__(self):
        logger.info("=" * 70)
        logger.info("MEDICAL TRANSCRIPT ANALYSIS PIPELINE")
        logger.info("NER: Local Model | LLM: Gemini API | Schemas: Centralized")
        logger.info("=" * 70)
        
        try:
            logger.info("\n[1/3] Initializing Preprocessor...")
            self.preprocessor = TranscriptPreprocessor()
            
            logger.info("\n[2/3] Loading NER Model...")
            logger.info("(First time: downloads ~300MB)")
            self.ner = MedicalNERExtractor()
            
            logger.info("\n[3/3] Initializing Gemini API...")
            self.llm = LLMExtractor()
            
            logger.info("\n[SUCCESS] All components initialized successfully!\n")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def process(self, transcript_path: str) -> Dict:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("=" * 70)
            logger.info(f"Processing: {transcript_path}")
            logger.info(f"Timestamp: {timestamp}")
            logger.info("=" * 70)
            
            # Load and clean transcript
            logger.info("\n[STEP 1] Loading and preprocessing...")
            with open(transcript_path, 'r', encoding='utf-8') as f:
                raw_transcript = f.read()
            
            logger.info(f"   Loaded: {len(raw_transcript)} characters")
            
            cleaned = self.preprocessor.clean_transcript(raw_transcript)
            speakers = self.preprocessor.split_speakers(cleaned)
            
            if not self.preprocessor.validate_transcript(speakers):
                raise ValueError("Invalid transcript format")
            
            logger.info(f"   [OK] {speakers['metadata']['doctor_turns']} doctor, "
                       f"{speakers['metadata']['patient_turns']} patient turns")
            
            # Extract medical entities
            logger.info("\n[STEP 2] Extracting entities (LOCAL NER)...")
            entities = self.ner.extract_entities(speakers['full_text'])
            categorized = self.ner.categorize_entities(entities)
            
            logger.info(f"   [OK] {len(entities)} entities extracted")
            
            # Generate medical summary
            logger.info("\n[STEP 3] Generating summary (GEMINI)...")
            summary = self.llm.extract_medical_summary(
                speakers['full_text'],
                entities
            )
            
            # Analyze sentiment and intent
            logger.info("\n[STEP 4] Analyzing sentiment/intent (GEMINI)...")
            sentiment_intent = self.llm.analyze_sentiment_intent(
                speakers['patient']
            )
            
            # Generate SOAP note
            logger.info("\n[STEP 5] Generating SOAP note (GEMINI)...")
            soap_note = self.llm.generate_soap_note(
                speakers['full_text'],
                entities,
                speakers
            )
            
            # Compile all results
            logger.info("\n[STEP 6] Compiling results...")
            
            results = {
                OutputFields.TRANSCRIPT_INFO: {
                    OutputFields.SOURCE_FILE: str(transcript_path),
                    OutputFields.TIMESTAMP: timestamp,
                    OutputFields.PROCESSING_DATE: datetime.now().isoformat(),
                    OutputFields.METADATA: speakers['metadata']
                },
                OutputFields.ENTITIES: {
                    OutputFields.ALL_ENTITIES: entities,
                    OutputFields.CATEGORIZED: categorized,
                    OutputFields.STATISTICS: {
                        "total": len(entities),
                        "by_category": {
                            cat: len(ent_list) 
                            for cat, ent_list in categorized.items()
                        },
                        "average_confidence": round(
                            sum(e['confidence'] for e in entities) / len(entities), 3
                        ) if entities else 0.0
                    }
                },
                OutputFields.MEDICAL_SUMMARY: summary,
                OutputFields.SENTIMENT_INTENT: sentiment_intent,
                OutputFields.SOAP_NOTE: soap_note
            }
            
            self._save_outputs(results, timestamp)
            
            logger.info("\n" + "=" * 70)
            logger.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            
            return results
            
        except Exception as e:
            logger.error(f"\n[ERROR] Pipeline failed: {e}", exc_info=True)
            raise
    
    def _save_outputs(self, results: Dict, timestamp: str):
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n[SAVING] Saving outputs...")
        

        filenames = CONFIG['output']['filenames']
        

        outputs = {
            filenames['summary']: results[OutputFields.MEDICAL_SUMMARY],
            filenames['sentiment']: results[OutputFields.SENTIMENT_INTENT],
            filenames['soap']: results[OutputFields.SOAP_NOTE],
            filenames['entities']: results[OutputFields.ENTITIES]
        }
        
        for filename, data in outputs.items():
            filepath = output_dir / f"{timestamp}_{filename}"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"   [OK] {filepath.name}")
        

        complete_path = output_dir / f"{timestamp}_{filenames['complete']}"
        with open(complete_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"   [OK] {complete_path.name}")
    
    def print_summary(self, results: Dict):
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        info = results[OutputFields.TRANSCRIPT_INFO]
        print(f"\nSource: {info[OutputFields.SOURCE_FILE]}")
        print(f"Processed: {info[OutputFields.PROCESSING_DATE]}")
        print(f"Total Turns: {info[OutputFields.METADATA]['total_turns']}")
        
        stats = results[OutputFields.ENTITIES][OutputFields.STATISTICS]
        print(f"\nENTITIES: {stats['total']} total")
        for cat, count in stats['by_category'].items():
            if count > 0:
                print(f"   {cat}: {count}")
        print(f"   Avg Confidence: {stats['average_confidence']:.1%}")
        
        summary = results[OutputFields.MEDICAL_SUMMARY]
        print(f"\nSUMMARY:")
        print(f"   Patient: {summary[MedicalSummaryFields.PATIENT_NAME]}")
        print(f"   Diagnosis: {summary[MedicalSummaryFields.DIAGNOSIS]}")
        print(f"   Symptoms: {len(summary[MedicalSummaryFields.SYMPTOMS])}")
        print(f"   Current Status: {summary[MedicalSummaryFields.CURRENT_STATUS]}")
        
        si = results[OutputFields.SENTIMENT_INTENT]
        print(f"\nSENTIMENT: {si[SentimentIntentFields.SENTIMENT]}")
        print(f"INTENT: {si[SentimentIntentFields.INTENT]}")
        
        print("\nAll outputs saved to data/output/")
        print("=" * 70 + "\n")


def main():
    print("\nMedical Transcript Analysis Pipeline")
    print("   NER: d4data/biomedical-ner-all (Local)")
    print("   LLM: Google Gemini API (Cloud)")
    print("   Architecture: Schema-based, Config-driven")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\n[ERROR] No transcript file specified")
        print("\nUsage:")
        print("   python main.py <path_to_transcript>")
        print("\nExample:")
        print("   python main.py data/input/transcript.txt")
        print()
        sys.exit(1)
    
    transcript_path = sys.argv[1]
    
    if not Path(transcript_path).exists():
        print(f"\n[ERROR] File '{transcript_path}' not found\n")
        sys.exit(1)
    
    try:
        pipeline = MedicalTranscriptPipeline()
        results = pipeline.process(transcript_path)
        pipeline.print_summary(results)
        
        print("[SUCCESS] Processing complete!\n")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user\n")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n[ERROR] Fatal error: {e}\n")
        print("Check pipeline.log for details.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
