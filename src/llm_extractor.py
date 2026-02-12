# Gemini API integration for text analysis

import google.generativeai as genai
import json
import logging
from typing import Dict, List
from transformers import pipeline

from config import CONFIG
from schemas import (
    MedicalSummaryFields,
    SentimentIntentFields,
    SOAPFields,
    EntityFields
)

logger = logging.getLogger(__name__)


class LLMExtractor:
    
    def __init__(self):
        logger.info(f"Initializing Gemini API: {CONFIG['gemini']['model_name']}")
        
        try:

            genai.configure(api_key=CONFIG['gemini']['api_key'])
            

            self.model = genai.GenerativeModel(
                model_name=CONFIG['gemini']['model_name'],
                safety_settings=CONFIG['gemini']['safety_settings']
            )
            

            self.generation_config = {
                "temperature": CONFIG['gemini']['temperature'],
                "max_output_tokens": CONFIG['gemini']['max_tokens'],
                "top_p": CONFIG['gemini']['top_p'],
                "top_k": CONFIG['gemini']['top_k']
            }
            
            logger.info("[OK] Gemini API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise
        
        logger.info(f"Loading DistilBERT sentiment model: {CONFIG['sentiment']['model_name']}")
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=CONFIG['sentiment']['model_name'],
                device=-1 if CONFIG['sentiment']['device'] == 'auto' else CONFIG['sentiment']['device']
            )
            logger.info("[OK] DistilBERT sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def _generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        

        if response.startswith('```'):
            parts = response.split('```')
            if len(parts) >= 2:
                response = parts[1]

                if response.startswith('json') or response.startswith('JSON'):
                    response = response[4:]
        
        return response.strip()
    
    def extract_medical_summary(self, transcript: str, entities: List[Dict]) -> Dict:
        """
        Generate structured medical summary using Gemini + NER entities
        Uses MedicalSummaryFields schema for consistency (matches task specification)
        """
        logger.info("Generating medical summary with Gemini API...")
        

        max_entities = CONFIG['text_limits']['top_entities_for_llm']
        entity_list = "\n".join([
            f"- {e[EntityFields.TEXT]} ({e[EntityFields.TYPE]}, confidence: {e[EntityFields.CONFIDENCE]})" 
            for e in entities[:max_entities]
        ])
        

        max_chars = CONFIG['text_limits']['llm_max_chars']
        transcript_excerpt = transcript[:max_chars] if len(transcript) > max_chars else transcript
        
      
        prompt = f"""You are a medical AI assistant. Analyze this medical transcript and extract key information.

TRANSCRIPT:
{transcript_excerpt}

DETECTED MEDICAL ENTITIES (from NER model):
{entity_list}

Generate a JSON summary with these EXACT fields (must be valid JSON):
{{
  "{MedicalSummaryFields.PATIENT_NAME}": "patient name if mentioned, otherwise 'Unknown' (string)",
  "{MedicalSummaryFields.SYMPTOMS}": ["list", "of", "symptoms"],
  "{MedicalSummaryFields.DIAGNOSIS}": "primary diagnosis (single string, not array)",
  "{MedicalSummaryFields.TREATMENT}": ["list", "of", "treatments/medications"],
  "{MedicalSummaryFields.CURRENT_STATUS}": "patient's current condition/status (string)",
  "{MedicalSummaryFields.PROGNOSIS}": "expected outcome/recovery (string)"
}}

Rules:
- Output ONLY valid JSON, nothing else
- {MedicalSummaryFields.DIAGNOSIS} must be a STRING, not an array (e.g., "Whiplash injury")
- If multiple diagnoses exist, combine into one string (e.g., "Whiplash injury and lower back strain")
- If information is missing, use "Unknown" for strings or empty array [] for arrays
- Use the detected entities to help accuracy
- Be concise but accurate

JSON OUTPUT:"""

        try:
            response = self._generate(prompt)
            response = self._clean_json_response(response)
            
           
            summary = json.loads(response)
            
            
            summary = MedicalSummaryFields.validate(summary)
            
            logger.info("[OK] Medical summary generated")
            return summary
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Gemini: {e}")
            logger.debug(f"Raw response: {response}")
            return MedicalSummaryFields.get_default()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return MedicalSummaryFields.get_default()
    
    def analyze_sentiment_intent(self, patient_utterances: List[str]) -> Dict:
        """
        Hybrid approach: DistilBERT for sentiment (local) + Gemini for intent (cloud)
        Uses SentimentIntentFields schema for consistency (matches task specification)
        """
        logger.info("Analyzing sentiment (DistilBERT) and intent (Gemini)...")
        
        if not patient_utterances:
            return SentimentIntentFields.get_default()
        
        patient_text = " ".join(patient_utterances)
        
        sentiment = self._analyze_sentiment_with_distilbert(patient_text)
        
        intent = self._analyze_intent_with_gemini(patient_utterances)
        
        result = {
            SentimentIntentFields.SENTIMENT: sentiment,
            SentimentIntentFields.INTENT: intent
        }
        
        logger.info(f"[OK] Sentiment: {sentiment} (DistilBERT), Intent: {intent} (Gemini)")
        return result
    
    def _analyze_sentiment_with_distilbert(self, text: str) -> str:
        """
        Use DistilBERT to classify sentiment locally
        Maps POSITIVE/NEGATIVE to medical context: Reassured/Neutral/Anxious
        """
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            label = result['label']
            score = result['score']
            
            positive_threshold = CONFIG['sentiment']['positive_threshold']
            negative_threshold = CONFIG['sentiment']['negative_threshold']
            
            if label == 'POSITIVE' and score >= positive_threshold:
                return "Reassured"
            elif label == 'NEGATIVE' and score >= negative_threshold:
                return "Anxious"
            else:
                return "Neutral"
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return "Neutral"
    
    def _analyze_intent_with_gemini(self, patient_utterances: List[str]) -> str:
        """
        Use Gemini API to detect patient intent from statements
        """
        patient_text = "\n".join([f"{i+1}. {utt}" for i, utt in enumerate(patient_utterances)])
        
        intent_options = ", ".join(CONFIG['intent_labels'])
        
        prompt = f"""You are a medical AI assistant. Analyze the patient's primary intent from their statements.

PATIENT STATEMENTS:
{patient_text}

Determine the patient's PRIMARY intent. Choose EXACTLY ONE from these options:
{intent_options}

Intent Guidelines:
- "Reporting symptoms" = describing what's wrong, symptoms, pain
- "Seeking reassurance" = looking for comfort, worried about condition
- "Expressing improvement" = feeling better, progress updates
- "Asking questions" = inquiring about diagnosis, treatment, prognosis
- "Neutral update" = general information, no specific goal

Output ONLY the intent label, nothing else.

INTENT:"""

        try:
            response = self._generate(prompt).strip()
            
            if response in CONFIG['intent_labels']:
                return response
            
            for intent in CONFIG['intent_labels']:
                if intent.lower() in response.lower():
                    return intent
            
            logger.warning(f"Invalid intent from Gemini, defaulting to first label")
            return CONFIG['intent_labels'][0]
            
        except Exception as e:
            logger.error(f"Error in intent detection: {e}")
            return CONFIG['intent_labels'][0]
    
    def generate_soap_note(self, transcript: str, entities: List[Dict], speakers: Dict) -> Dict:
        """
        Generate SOAP note using Gemini
        Uses SOAPFields schema for consistency (matches task specification with nested structure)
        """
        logger.info("Generating SOAP note with Gemini API...")
        

        max_entities = CONFIG['text_limits']['top_entities_for_llm']
        entity_list = "\n".join([
            f"- {e[EntityFields.TEXT]} ({e[EntityFields.TYPE]})" 
            for e in entities[:max_entities]
        ])
        
        # Truncate transcript
        max_chars = CONFIG['text_limits']['llm_max_chars']
        transcript_excerpt = transcript[:max_chars] if len(transcript) > max_chars else transcript
        

        speaker_info = f"Doctor turns: {speakers['metadata']['doctor_turns']}, Patient turns: {speakers['metadata']['patient_turns']}"
        
        prompt = f"""You are a medical AI assistant. Generate a SOAP note from this medical transcript.

TRANSCRIPT:
{transcript_excerpt}

DETECTED MEDICAL ENTITIES:
{entity_list}

SPEAKER INFO:
{speaker_info}

Generate a SOAP note in JSON format with NESTED STRUCTURE and these EXACT fields (must be valid JSON):
{{
  "{SOAPFields.SUBJECTIVE}": {{
    "{SOAPFields.SUBJECTIVE_CHIEF_COMPLAINT}": "patient's main complaint (string)",
    "{SOAPFields.SUBJECTIVE_HISTORY}": "patient's history and symptoms description (string)"
  }},
  "{SOAPFields.OBJECTIVE}": {{
    "{SOAPFields.OBJECTIVE_PHYSICAL_EXAM}": "physical examination findings (string)",
    "{SOAPFields.OBJECTIVE_OBSERVATIONS}": "clinical observations (string)"
  }},
  "{SOAPFields.ASSESSMENT}": {{
    "{SOAPFields.ASSESSMENT_DIAGNOSIS}": "medical diagnosis (string)",
    "{SOAPFields.ASSESSMENT_SEVERITY}": "severity level (e.g., Mild, Moderate, Severe) (string)"
  }},
  "{SOAPFields.PLAN}": {{
    "{SOAPFields.PLAN_TREATMENT}": "treatment plan and interventions (string)",
    "{SOAPFields.PLAN_FOLLOWUP}": "follow-up instructions (string)"
  }}
}}

SOAP Guidelines:
- Subjective: Focus on what the PATIENT reports (symptoms, history, concerns)
- Objective: Focus on what the DOCTOR observes/measures (physical exam findings)
- Assessment: The doctor's diagnosis and severity assessment
- Plan: Treatment plan, medications, and follow-up schedule

CRITICAL: Output must be NESTED JSON with sub-objects, not flat strings!

Output ONLY valid JSON, nothing else.

JSON OUTPUT:"""

        try:
            response = self._generate(prompt)
            response = self._clean_json_response(response)
            soap = json.loads(response)
            
            soap = SOAPFields.validate(soap)
            
            logger.info("[OK] SOAP note generated")
            return soap
            
        except Exception as e:
            logger.error(f"Error generating SOAP: {e}")
            return SOAPFields.get_default()
