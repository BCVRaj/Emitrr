# Field definitions for JSON outputs

from typing import Dict, List, Any


class MedicalSummaryFields:
    """Field names for medical summary JSON output (matches task specification)"""
    
    PATIENT_NAME = "Patient_Name"
    SYMPTOMS = "Symptoms"
    DIAGNOSIS = "Diagnosis"
    TREATMENT = "Treatment"
    CURRENT_STATUS = "Current_Status"
    PROGNOSIS = "Prognosis"
    
    @classmethod
    def get_default(cls) -> Dict[str, Any]:
        return {
            cls.PATIENT_NAME: "Unknown",
            cls.SYMPTOMS: [],
            cls.DIAGNOSIS: "Unknown",
            cls.TREATMENT: [],
            cls.CURRENT_STATUS: "Unknown",
            cls.PROGNOSIS: "Unknown"
        }
    
    @classmethod
    def validate(cls, data: dict) -> dict:
        """Validate and fill missing fields with defaults"""
        default = cls.get_default()
        for field, default_value in default.items():
            if field not in data:
                data[field] = default_value
        return data
    
    @classmethod
    def get_field_list(cls) -> List[str]:
        return [
            cls.PATIENT_NAME,
            cls.SYMPTOMS,
            cls.DIAGNOSIS,
            cls.TREATMENT,
            cls.CURRENT_STATUS,
            cls.PROGNOSIS
        ]


class SentimentIntentFields:
    """Field names for sentiment and intent analysis JSON output (matches task specification)"""
    
    SENTIMENT = "Sentiment"
    INTENT = "Intent"
    
    @classmethod
    def get_default(cls) -> Dict[str, Any]:
        """Get default empty sentiment/intent result"""
        return {
            cls.SENTIMENT: "Neutral",
            cls.INTENT: "Unknown"
        }
    
    @classmethod
    def validate(cls, data: dict) -> dict:
        """Validate and fill missing fields with defaults"""
        default = cls.get_default()
        for field, default_value in default.items():
            if field not in data:
                data[field] = default_value
        return data


class SOAPFields:
    """Field names for SOAP note JSON output (matches task specification with nested structure)"""
    
    SUBJECTIVE = "Subjective"
    SUBJECTIVE_CHIEF_COMPLAINT = "Chief_Complaint"
    SUBJECTIVE_HISTORY = "History_of_Present_Illness"
    
    OBJECTIVE = "Objective"
    OBJECTIVE_PHYSICAL_EXAM = "Physical_Exam"
    OBJECTIVE_OBSERVATIONS = "Observations"
    
    ASSESSMENT = "Assessment"
    ASSESSMENT_DIAGNOSIS = "Diagnosis"
    ASSESSMENT_SEVERITY = "Severity"
    
    PLAN = "Plan"
    PLAN_TREATMENT = "Treatment"
    PLAN_FOLLOWUP = "Follow-Up"
    
    @classmethod
    def get_default(cls) -> Dict[str, Any]:
        """Get default empty SOAP note with nested structure"""
        return {
            cls.SUBJECTIVE: {
                cls.SUBJECTIVE_CHIEF_COMPLAINT: "Not documented",
                cls.SUBJECTIVE_HISTORY: "Not documented"
            },
            cls.OBJECTIVE: {
                cls.OBJECTIVE_PHYSICAL_EXAM: "Not documented",
                cls.OBJECTIVE_OBSERVATIONS: "Not documented"
            },
            cls.ASSESSMENT: {
                cls.ASSESSMENT_DIAGNOSIS: "Not documented",
                cls.ASSESSMENT_SEVERITY: "Not documented"
            },
            cls.PLAN: {
                cls.PLAN_TREATMENT: "Not documented",
                cls.PLAN_FOLLOWUP: "Not documented"
            }
        }
    
    @classmethod
    def validate(cls, data: dict) -> dict:
        """Validate and fill missing/empty fields with defaults"""
        default = cls.get_default()
        
        # Validate main sections
        for section in [cls.SUBJECTIVE, cls.OBJECTIVE, cls.ASSESSMENT, cls.PLAN]:
            if section not in data or not isinstance(data[section], dict):
                data[section] = default[section]
            else:
                # Validate sub-fields within each section
                for subfield, default_value in default[section].items():
                    if subfield not in data[section] or not data[section][subfield] or str(data[section][subfield]).strip() == "":
                        data[section][subfield] = default_value
        
        return data


class EntityFields:
    
    TEXT = "text"
    TYPE = "type"
    CONFIDENCE = "confidence"
    START = "start"
    END = "end"
    
    @classmethod
    def create(cls, text: str, entity_type: str, confidence: float, 
               start: int = 0, end: int = 0) -> Dict[str, Any]:
        return {
            cls.TEXT: text,
            cls.TYPE: entity_type,
            cls.CONFIDENCE: float(round(confidence, 3)),
            cls.START: int(start),
            cls.END: int(end)
        }


class OutputFields:
    
    TRANSCRIPT_INFO = "transcript_info"
    ENTITIES = "entities"
    MEDICAL_SUMMARY = "medical_summary"
    SENTIMENT_INTENT = "sentiment_intent"
    SOAP_NOTE = "soap_note"
    

    SOURCE_FILE = "source_file"
    TIMESTAMP = "timestamp"
    PROCESSING_DATE = "processing_date"
    METADATA = "metadata"
    

    ALL_ENTITIES = "all_entities"
    CATEGORIZED = "categorized"
    STATISTICS = "statistics"
