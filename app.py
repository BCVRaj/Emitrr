import streamlit as st
import json
import tempfile
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import MedicalTranscriptPipeline
from schemas import MedicalSummaryFields, SentimentIntentFields, SOAPFields, EntityFields

st.set_page_config(
    page_title="Medical Transcript Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def display_medical_summary(summary):
    patient_name = summary.get(MedicalSummaryFields.PATIENT_NAME, 'Unknown')
    st.markdown(f"**Patient:** {patient_name}")
    
    diagnosis = summary.get(MedicalSummaryFields.DIAGNOSIS, 'Unknown')
    st.markdown(f"### 🔍 Diagnosis: `{diagnosis}`")
    
    st.markdown("**Symptoms:**")
    symptoms = summary.get(MedicalSummaryFields.SYMPTOMS, [])
    if symptoms:
        for symptom in symptoms:
            st.markdown(f"- {symptom}")
    else:
        st.info("No symptoms documented")
    
    st.markdown("**Treatment Plan:**")
    treatments = summary.get(MedicalSummaryFields.TREATMENT, [])
    if treatments:
        for i, treatment in enumerate(treatments, 1):
            st.markdown(f"{i}. {treatment}")
    else:
        st.info("No treatment documented")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_status = summary.get(MedicalSummaryFields.CURRENT_STATUS, 'Unknown')
        st.info(f"**Current Status:**\n\n{current_status}")
    
    with col2:
        prognosis = summary.get(MedicalSummaryFields.PROGNOSIS, 'Unknown')
        st.success(f"**Prognosis:**\n\n{prognosis}")

def display_sentiment_intent(data):
    sentiment = data.get(SentimentIntentFields.SENTIMENT, 'Unknown')
    intent = data.get(SentimentIntentFields.INTENT, 'Unknown')
    
    sentiment_icons = {
        'Anxious': '🔴',
        'Neutral': '⚪',
        'Reassured': '🟢'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        icon = sentiment_icons.get(sentiment, '⚪')
        st.metric(
            label="Patient Sentiment",
            value=f"{icon} {sentiment}"
        )
        
        if sentiment == 'Anxious':
            st.error("Patient shows signs of anxiety")
        elif sentiment == 'Reassured':
            st.success("Patient appears reassured")
        else:
            st.info("Patient shows neutral emotion")
    
    with col2:
        st.metric(
            label="Primary Intent",
            value=intent
        )
        st.info(f"Patient is: {intent.lower()}")

def display_entities(entities_data):
    all_entities = entities_data.get('all_entities', [])
    categorized = entities_data.get('categorized', {})
    stats = entities_data.get('statistics', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entities", stats.get('total', 0))
    with col2:
        st.metric("Categories", len(categorized))
    with col3:
        avg_conf = stats.get('average_confidence', 0)
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    st.markdown("### By Category")
    
    for category, entities in categorized.items():
        if entities:
            with st.expander(f"**{category.title()}** ({len(entities)})", expanded=(len(entities) > 0)):
                for entity in entities:
                    conf = entity.get(EntityFields.CONFIDENCE, 0)
                    st.markdown(
                        f"- **{entity.get(EntityFields.TEXT)}** "
                        f"({entity.get(EntityFields.TYPE)}) - "
                        f"Confidence: {conf:.1%}"
                    )
    
    st.markdown("### All Entities Table")
    if all_entities:
        df_data = []
        for entity in all_entities:
            df_data.append({
                'Entity': entity.get(EntityFields.TEXT),
                'Type': entity.get(EntityFields.TYPE),
                'Confidence': f"{entity.get(EntityFields.CONFIDENCE, 0):.1%}"
            })
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

def display_soap_note(soap_data):
    with st.expander("📋 **S - Subjective**", expanded=True):
        subjective = soap_data.get(SOAPFields.SUBJECTIVE, {})
        st.markdown("**Chief Complaint:**")
        st.write(subjective.get(SOAPFields.SUBJECTIVE_CHIEF_COMPLAINT, 'Not documented'))
        
        st.markdown("**History of Present Illness:**")
        st.write(subjective.get(SOAPFields.SUBJECTIVE_HISTORY, 'Not documented'))
    
    with st.expander("🔍 **O - Objective**", expanded=True):
        objective = soap_data.get(SOAPFields.OBJECTIVE, {})
        st.markdown("**Physical Exam:**")
        st.write(objective.get(SOAPFields.OBJECTIVE_PHYSICAL_EXAM, 'Not documented'))
        
        st.markdown("**Observations:**")
        st.write(objective.get(SOAPFields.OBJECTIVE_OBSERVATIONS, 'Not documented'))
    
    with st.expander("📊 **A - Assessment**", expanded=True):
        assessment = soap_data.get(SOAPFields.ASSESSMENT, {})
        st.markdown("**Diagnosis:**")
        st.write(assessment.get(SOAPFields.ASSESSMENT_DIAGNOSIS, 'Not documented'))
        
        st.markdown("**Severity:**")
        st.write(assessment.get(SOAPFields.ASSESSMENT_SEVERITY, 'Not documented'))
    
    with st.expander("📝 **P - Plan**", expanded=True):
        plan = soap_data.get(SOAPFields.PLAN, {})
        st.markdown("**Treatment:**")
        st.write(plan.get(SOAPFields.PLAN_TREATMENT, 'Not documented'))
        
        st.markdown("**Follow-Up:**")
        st.write(plan.get(SOAPFields.PLAN_FOLLOWUP, 'Not documented'))

def create_download_button(data, filename, label):
    json_str = json.dumps(data, indent=2)
    
    st.download_button(
        label=label,
        data=json_str,
        file_name=filename,
        mime='application/json',
        use_container_width=True
    )

st.title("🏥 Medical Transcript Analysis")
st.caption("AI-powered analysis using Biomedical NER, DistilBERT Sentiment & Gemini API")

st.divider()

st.subheader("📁 Upload Transcript")

uploaded_file = st.file_uploader(
    "Choose a transcript file",
    type=['txt'],
    help="Upload a .txt file containing doctor-patient conversation"
)

if uploaded_file is not None:
    transcript_text = uploaded_file.read().decode('utf-8')
    
    st.success(f"✅ File uploaded: {uploaded_file.name} ({len(transcript_text)} characters)")
    
    with st.expander("📄 Preview Transcript", expanded=True):
        st.text_area(
            "Transcript Content",
            value=transcript_text,
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )
    
    st.divider()
    st.subheader("🔬 Analyze")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Analyze Transcript",
            type="primary",
            use_container_width=True
        )
    
    if analyze_button:
        try:
            with st.spinner("⏳ Processing transcript..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📝 Saving transcript...")
                progress_bar.progress(20)
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                    tmp.write(transcript_text)
                    temp_path = tmp.name
                
                status_text.text("🔧 Initializing pipeline...")
                progress_bar.progress(40)
                
                pipeline = MedicalTranscriptPipeline()
                
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(60)
                
                results = pipeline.process(temp_path)
                
                os.unlink(temp_path)
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
            
            st.session_state['results'] = results
            st.session_state['analyzed'] = True
            
            st.success("🎉 Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

if 'analyzed' in st.session_state and st.session_state['analyzed']:
    
    st.divider()
    st.header("📊 Results")
    
    results = st.session_state['results']
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Medical Summary",
        "💭 Sentiment & Intent",
        "🔍 Entities",
        "📝 SOAP Note",
        "💾 Downloads"
    ])
    
    with tab1:
        st.subheader("Medical Summary")
        display_medical_summary(results['medical_summary'])
        
        with st.expander("🔍 View Raw JSON"):
            st.json(results['medical_summary'])
    
    with tab2:
        st.subheader("Sentiment & Intent")
        display_sentiment_intent(results['sentiment_intent'])
        
        with st.expander("🔍 View Raw JSON"):
            st.json(results['sentiment_intent'])
    
    with tab3:
        st.subheader("Medical Entities")
        display_entities(results['entities'])
        
        with st.expander("🔍 View Raw JSON"):
            st.json(results['entities'])
    
    with tab4:
        st.subheader("SOAP Note")
        display_soap_note(results['soap_note'])
        
        with st.expander("🔍 View Raw JSON"):
            st.json(results['soap_note'])
    
    with tab5:
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_download_button(
                results['medical_summary'],
                'medical_summary.json',
                '📋 Download Medical Summary'
            )
            
            create_download_button(
                results['sentiment_intent'],
                'sentiment_intent.json',
                '💭 Download Sentiment & Intent'
            )
            
            create_download_button(
                results['entities'],
                'entities.json',
                '🔍 Download Entities'
            )
        
        with col2:
            create_download_button(
                results['soap_note'],
                'soap_note.json',
                '📝 Download SOAP Note'
            )
            
            create_download_button(
                results,
                'complete_results.json',
                '💾 Download Complete Results'
            )

with st.sidebar:
    st.header("ℹ️ About")
    
    st.markdown("""
    **Medical Transcript Analyzer**
    
    **Technology:**
    - Biomedical NER
    - DistilBERT Sentiment
    - Gemini API
    """)
    
    st.divider()
    
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        st.success("✅ Analysis complete")
    else:
        st.info("⏳ Ready")
    
    st.divider()
    
    if st.button("🔄 Clear Results", use_container_width=True):
        if 'results' in st.session_state:
            del st.session_state['results']
        if 'analyzed' in st.session_state:
            del st.session_state['analyzed']
        st.rerun()
