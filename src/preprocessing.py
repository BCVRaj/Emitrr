# Transcript preprocessing

import re
import logging
from typing import Dict, List

from config import CONFIG

logger = logging.getLogger(__name__)


class TranscriptPreprocessor:
    
    def __init__(self):

        self.filler_words = CONFIG['preprocessing']['filler_words']
        self.remove_fillers = CONFIG['preprocessing']['remove_fillers']
        self.normalize_whitespace = CONFIG['preprocessing']['normalize_whitespace']
        

        self.doctor_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in CONFIG['speakers']['patterns']['doctor']
        ]
        self.patient_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in CONFIG['speakers']['patterns']['patient']
        ]
        
        logger.info("[OK] Preprocessor initialized with configurable patterns")
        logger.info(f"   Doctor patterns: {len(self.doctor_patterns)}")
        logger.info(f"   Patient patterns: {len(self.patient_patterns)}")
    
    def clean_transcript(self, text: str) -> str:
        try:
            logger.info("Cleaning transcript...")
            
            if self.normalize_whitespace:
                text = re.sub(r'[ \t]+', ' ', text)
            
            if self.remove_fillers:
                for filler in self.filler_words:
                    text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
            
            text = text.strip()
            logger.info(f"[OK] Cleaned transcript: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning transcript: {e}")
            return text
    
    def split_speakers(self, text: str) -> Dict[str, List[str]]:
        try:
            logger.info("Splitting speakers...")
            
            lines = text.split('\n')
            doctor_lines = []
            patient_lines = []
            unmatched_lines = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                matched = False
                

                for pattern in self.doctor_patterns:
                    if pattern.match(line):
                        clean_line = pattern.sub('', line).strip()
                        if clean_line:  # Only add non-empty lines
                            doctor_lines.append(clean_line)
                        matched = True
                        break
                
                if matched:
                    continue
                

                for pattern in self.patient_patterns:
                    if pattern.match(line):
                        clean_line = pattern.sub('', line).strip()
                        if clean_line:  
                            patient_lines.append(clean_line)
                        matched = True
                        break
                
                if not matched:
                    unmatched_lines += 1
            
            result = {
                "doctor": doctor_lines,
                "patient": patient_lines,
                "full_text": text,
                "metadata": {
                    "total_turns": len(doctor_lines) + len(patient_lines),
                    "doctor_turns": len(doctor_lines),
                    "patient_turns": len(patient_lines),
                    "total_characters": len(text),
                    "unmatched_lines": unmatched_lines
                }
            }
            
            logger.info(f"[OK] Split complete: {len(doctor_lines)} doctor, "
                       f"{len(patient_lines)} patient turns")
            
            if unmatched_lines > 0:
                logger.warning(f"[WARNING] {unmatched_lines} lines didn't match any speaker pattern")
            
            return result
            
        except Exception as e:
            logger.error(f"Error splitting speakers: {e}")
            return {
                "doctor": [], 
                "patient": [], 
                "full_text": text, 
                "metadata": {}
            }
    
    def validate_transcript(self, speakers: Dict) -> bool:
        if not speakers['doctor']:
            logger.warning("[WARNING] No doctor utterances found!")
            return False
        
        if not speakers['patient']:
            logger.warning("[WARNING] No patient utterances found!")
            return False
        
        if len(speakers['full_text']) < 50:
            logger.warning("[WARNING] Transcript seems too short!")
            return False
        
        return True
