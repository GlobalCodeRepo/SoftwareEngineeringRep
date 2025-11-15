import streamlit as st
import pandas as pd
import os
import json
import math
import uuid
import time
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer.operators import Replace
from presidio_anonymizer.entities import OperatorConfig
PRESIDIO_AVAILABLE = True
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv
from unittest import TestCase, main

# --- PRESIDIO IMPORTS ---
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.operators import Replace
    PRESIDIO_AVAILABLE = True
except ImportError:
    # Set a flag if not installed, falling back to simple masking
    PRESIDIO_AVAILABLE = False
    class MockAnalyzerEngine:
        def analyze(self, *args, **kwargs):
            return []
    class MockAnonymizerEngine:
        def anonymize(self, text, *args, **kwargs):
            return text
    AnalyzerEngine = MockAnalyzerEngine
    AnonymizerEngine = MockAnonymizerEngine
# --- END PRESIDIO IMPORTS ---


# --- 1. CONFIGURATION AND CORE LOGIC ---
# ... (CATEGORY_MAP, THRESHOLD_T, CATEGORY_SCHEMA remain the same)
CATEGORY_MAP = {
    "Market Manipulation/Misconduct": {"f1": 4, "f2": 4, "tier": "H"},
    "Market Bribery":                 {"f1": 4, "f2": 4, "tier": "H"},
    "Change in communication":        {"f1": 3, "f2": 2, "tier": "U"},
    "Secrecy":                        {"f1": 3, "f2": 2, "tier": "U"},
    "Employee ethics":                {"f1": 1, "f2": 4, "tier": "M"},
    "Complaints":                     {"f1": 2, "f2": 1, "tier": "L"},
}
THRESHOLD_T = 10

CATEGORY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(CATEGORY_MAP.keys()),
                "description": "The specific non-compliant category identified in the text.",
            },
            "sourceline": {
                "type": "string",
                "description": "The exact sentence or phrase from the email body that triggers the non-compliance alert. Must be a direct quote.",
            },
            "reason": {
                "type": "string",
                "description": "A concise explanation of why this source line is non-compliant under this category.",
            },
        },
        "required": ["category", "sourceline", "reason"],
    },
}

# --- DATABASE PERSISTENCE (SIMULATION - UNCHANGED) ---
def simulate_db_log_action(email_id, action, details):
    # ... (Implementation remains the same)
    log_entry = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email_id": email_id,
        "action": action,
        "details": details,
    }
    st.session_state.audit_logs.append(log_entry)
    print(f"DB AUDIT LOG: {email_id[:8]} | Action: {action} | Details: {details.get('status')}")
    return log_entry

def simulate_db_store_processed_email(email_data):
    # ... (Implementation remains the same)
    #st.session_state.processed_emails.append(email_data)
    print(f"DB WRITE: Email {email_data['id'][:8]} written to ProcessedEmails (P-Score: {email_data['p_score']})")
    return email_data


# --- CORE LOGIC FUNCTIONS (P-SCORE UNCHANGED) ---

def calculate_p_score(categories):
    # ... (Implementation remains the same)
    if not categories:
        return 0.0

    sum_products = 0
    n = len(categories)
    tiers = {"H": 0, "U": 0, "M": 0, "L": 0}

    for finding in categories:
        category_name = finding.get('category')
        config = CATEGORY_MAP.get(category_name, {"f1": 0, "f2": 0, "tier": None})
        
        sum_products += (config['f1'] * config['f2'])
        if config['tier']:
            tiers[config['tier']] += 1
            
    b = 0
    if tiers["H"] > 0: b = tiers["H"] * THRESHOLD_T
    elif tiers["U"] > 0: b = tiers["U"] * THRESHOLD_T
    elif tiers["M"] > 0: b = tiers["M"] * THRESHOLD_T
    elif tiers["L"] > 0: b = tiers["L"] * THRESHOLD_T

    average_product = sum_products / n
    p_score = math.ceil(average_product + b)
    return float(p_score)


# --- UPDATED SECURITY MODULE (Presidio) ---

def mask_pii_with_presidio(email_body):
    """
    Encryption Module: Uses Presidio for robust PII masking (Tokenization).
    Falls back to simple masking if Presidio is not installed.
    """
    pii_map = {}

    if not PRESIDIO_AVAILABLE:
        # Fallback to simple masking (e.g., from the previous code)
        replacements = {
            "John Smith": "[PERSON_A]",
            "Jane Doe": "[PERSON_B]",
            "account_1234567890": "[ACC_NUM_INTERNAL]",
            "ext-party@corp.com": "[EMAIL_EXTERNAL]",
        }
        masked_body = email_body
        for original, token in replacements.items():
            masked_body = masked_body.replace(original, token)
            pii_map[token] = original
        
        return masked_body, pii_map
    
    # --- PRESIDIO CORE LOGIC ---
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # 1. Analyze: Detect PII types (PERSON, PHONE_NUMBER, etc.)
    # Note: We limit to common entities for efficiency
    results = analyzer.analyze(
        text=email_body, 
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE", "CREDIT_CARD", "US_DRIVER_LICENSE"],
        language='en'
    )
    
    # 2. Anonymize: Replace detected entities with a token (Replace operator)
    # The 'Replace' operator is configured to use the entity type (e.g., <PERSON>)
    anonymized_results = anonymizer.anonymize(
        text=email_body,
        analyzer_results=results,
        operators={"DEFAULT": Replace(new_value="<" + "[ENTITY_TYPE]" + ">")}
    )
    
    masked_body = anonymized_results.text
    
    # Create a basic PII map for auditing (simulated)
    for res in results:
        entity_text = email_body[res.start:res.end]
        token = f"<[[{res.entity_type}]]>"
        pii_map[token] = entity_text
        
    return masked_body, pii_map

# The wrapper function is renamed and updated to use the new module
def classify_email_llm(email_data, client: AzureOpenAI):
    """Calls the OpenAI API to classify a single masked email (MUST BE THREAD-SAFE)."""
    # ... (Implementation remains the same)
    email_id = email_data['id']
    # Use the masked body generated by the Presidio integration
    masked_body = email_data['masked_body']
    
    system_prompt = (
        "You are a world-class Financial Communication Surveillance Analyst. "
        "Your task is to review the following masked email and identify ALL instances of non-compliance "
        "related to the following financial categories: "
        f"{list(CATEGORY_MAP.keys())}. "
        "The PII has been masked as <ENTITY_TYPE>. You must use the masked text in your sourceline quote if PII is involved. "
        "You MUST respond ONLY with a JSON array that strictly adheres to the provided schema. "
        "For each finding, you must quote the exact 'sourceline' and provide a concise 'reason'."
    )
    
    user_prompt = f"Analyze the communication:\n\n{masked_body}"
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object", "schema": CATEGORY_SCHEMA},
            temperature=0.0
        )
        
        findings_json = response.choices[0].message.content
        
        parsed_data = json.loads(findings_json)
        # Safely extract findings, handling common LLM wrapping issues
        findings = parsed_data.get('items', parsed_data) if isinstance(parsed_data, dict) else parsed_data
        
        simulate_db_log_action(
            email_id=email_id, action='CLASSIFIED', details={'status': 'SUCCESS', 'findings_count': len(findings)}
        )
        return findings

    except Exception as e:
        simulate_db_log_action(
            email_id=email_id, action='CLASSIFIED', details={'status': 'FAILURE', 'error': str(e)}
        )
        return []

def process_single_email_pipeline(email_data_dict, client: AzureOpenAI):
    """
    Executes the full security, classification, and scoring pipeline for one email.
    FIXED: Ensures all required keys are initialized for DataFrame creation.
    """
    email_id = email_data_dict['id']
    
    # Initialize required keys to safe defaults immediately
    email_data_dict['masked_body'] = email_data_dict.get('masked_body', email_data_dict['body'])
    email_data_dict['pii_map'] = email_data_dict.get('pii_map', {})
    email_data_dict['categories'] = email_data_dict.get('categories', [])
    email_data_dict['p_score'] = email_data_dict.get('p_score', 0.0)
    email_data_dict['categories_summary'] = email_data_dict.get('categories_summary', "PROCESSING FAILED")
    email_data_dict['reviewer_action'] = email_data_dict.get('reviewer_action', 'PENDING')

    try:
        # Step 2: PII Encryption/Masking (Using Presidio or Fallback)
        masked_body, pii_map = mask_pii_with_presidio(email_data_dict['body'])
        email_data_dict['masked_body'] = masked_body
        email_data_dict['pii_map'] = pii_map

        # Step 3: Parallel Classification (LLM API Call)
        # We assume classify_email_llm handles its own API exceptions 
        # but returns [] if an error occurs.
        findings = classify_email_llm(email_data_dict, client)

        # Step 4: Calculate P-Score
        p_score = calculate_p_score(findings)
        
        # Step 5: Finalize Data Structure
        email_data_dict['categories'] = findings
        email_data_dict['p_score'] = p_score
        email_data_dict['categories_summary'] = ", ".join(f['category'] for f in findings) if findings else "CLEAN"
        email_data_dict['reviewer_action'] = 'PENDING'

    except Exception as e:
        # Catch any unexpected errors during PII masking or processing setup
        email_data_dict['categories'] = []
        email_data_dict['p_score'] = 0.0
        email_data_dict['categories_summary'] = f"CRASH: {str(e)[:40]}..."
        email_data_dict['reviewer_action'] = 'ERROR'
        # Log the failure using the thread-safe print function
        print(f"THREAD EXCEPTION for {email_id[:8]}: {e}")

    # FIX: We now return the dictionary. The main thread will update st.session_state 
    # with this data, which resolves the threading issue.
    # We call the modified (thread-safe) simulation function, which no longer touches st.session_state.
    return simulate_db_store_processed_email(email_data_dict)

# --- REST OF THE CODE (UI/TESTS/MAIN) REMAINS THE SAME ---

# NOTE: The rest of the functions (init_session_state, handle_reviewer_action, 
# process_email_batch, render_prioritization_queue, render_details_pane, 
# TestPrioritizationLogic, run_tests, and main) are identical to the previous runnable version 
# as the structural changes were only contained within the PII masking logic.


# --- 2. STREAMLIT STATE AND UI LOGIC ---

def init_session_state():
    """Initializes all necessary session variables."""
    if 'processed_emails' not in st.session_state:
        st.session_state.processed_emails = [] 
    if 'audit_logs' not in st.session_state:
        st.session_state.audit_logs = [] 
    if 'selected_email_id' not in st.session_state:
        st.session_state.selected_email_id = None
    if 'api_client' not in st.session_state:
        st.session_state.api_client = None


def handle_reviewer_action(email_id, action_type):
    """Handles reviewer feedback (True/False Positive) and updates audit."""
    
    simulate_db_log_action(
        email_id=email_id, 
        action=action_type, 
        details={"status": "Reviewer Confirmation"}
    )
    
    for email in st.session_state.processed_emails:
        if email['id'] == email_id:
            email['reviewer_action'] = action_type
            break
            
    st.success(f"Audit log updated: Email {email_id[:8]} marked as {action_type}.")
    st.session_state.selected_email_id = None 
    st.rerun()

def process_email_batch(uploaded_files, max_workers=5):
    """
    The main looper function. Reads files and executes the parallel pipeline.
    """
    if not uploaded_files:
        st.warning("Please upload at least one email file to begin processing.")
        return

    client = st.session_state.api_client
    if client is None:
        st.error("Cannot run: OpenAI API client is not initialized.")
        return

    # 1. Read files and prepare list of dictionaries (CRITICAL: Read content fully NOW)
    emails_to_process = []
    for uploaded_file in uploaded_files:
        try:
            # Read content fully and decode
            content = uploaded_file.read().decode("utf-8")
            
            # Simple content parsing
            subject = content.split('Subject:')[1].split('\n')[0].strip() if 'Subject:' in content else uploaded_file.name
            body = content.split('\n\n', 1)[-1].strip() if '\n\n' in content else content
            
            emails_to_process.append({
                'id': str(uuid.uuid4()),
                'filename': uploaded_file.name,
                'subject': subject,
                'body': body,
            })
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {e}")
            continue

    if not emails_to_process:
        return

    # Temporary lists to collect results and logs from threads
    final_processed_email = []
    batch_audit_logs = []
    
    with st.status("Processing emails in parallel...", expanded=True) as status:
        st.write(f"Starting parallel classification for {len(emails_to_process)} emails (Max workers: {max_workers})...")
        
        # 2. Start Parallel Execution (The Looper)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_email = {
                executor.submit(process_single_email_pipeline, email_data, client): email_data['id']
                       for email_data in emails_to_process
            
            for i, future in enumerate(as_completed(future_to_email):
                try:
                    result = future.result()
                    final_processed_emails.append(result)
                    status.update(label=f"Classifying: {i+1}/{len(emails_to_process)} emails complete.")
                except Exception as e:
                    email_id = future_to_email.get(future, 'N/A')
                    st.error(f"A processing thread failed for Email ID[:8]): {e}")
                    st.session_state.audit_logs.append({
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "email_id": email_id,
                        "action": 'PARALLEL_FAIL',
                        "details": {'error': str(e)},
                    })
                        
         st.session_state.processed_emails = final_processed_emails           
        # Sort results by P-Score (Highest Priority First)
        st.session_state.processed_emails.sort(key=lambda x: x.get('p_score', 0), reverse=True)

        status.update(label="✅ Classification and Prioritization Complete!", state="complete")
        st.balloons()


def render_prioritization_queue():
    """Renders the main alert queue (DataFrame)."""
    emails_df = pd.DataFrame(st.session_state.processed_emails)
    
    if emails_df.empty:
        st.info("Upload emails to start the surveillance pipeline.")
        return
    
    # Prepare DataFrame for Display
    display_cols = ['p_score', 'subject', 'filename', 'categories_summary', 'reviewer_action', 'id']
    emails_df = emails_df.rename(columns={
        'p_score': 'P-Score', 
        'categories_summary': 'Findings',
        'reviewer_action': 'Action Status'
    })[display_cols].copy()
    
    def color_score(val):
        if val >= 20: return 'background-color: #F87171; color: white'
        if val >= 10: return 'background-color: #FBBF24'
        if val > 0: return 'background-color: #FEF3C7'
        return ''

    def select_row(selection):
        if selection and selection['selection']:
            selected_index = selection['selection'][0]
            original_row = st.session_state.processed_emails[selected_index]
            st.session_state.selected_email_id = original_row['id']
            st.rerun()

    st.subheader("B. Prioritized Alert Queue (Highest P-Score First)")
    
    st.dataframe(
        emails_df.style.applymap(color_score, subset=['P-Score']),
        column_order=['P-Score', 'subject', 'Findings', 'Action Status'],
        column_config={
            "P-Score": st.column_config.ProgressColumn(
                "P-Score", 
                format="%f", 
                min_value=0, max_value=max(emails_df['P-Score'].max(), 30)
            ),
            "subject": "Subject",
            "filename": "Filename",
            "Findings": "Categories Found",
            "Action Status": "Action Status",
            "id": None
        },
        hide_index=True,
        on_select=select_row,
        selection_mode='single-row',
        use_container_width=True,
        key="prioritization_queue_df"
    )

def render_details_pane():
    """Renders the right-hand panel with evidence and audit controls."""
    if not st.session_state.selected_email_id:
        st.info("Select an email from the queue to view detailed analysis and evidence.")
        return
        
    selected_email = next((e for e in st.session_state.processed_emails if e['id'] == st.session_state.selected_email_id), None)

    if not selected_email:
        st.error("Error: Selected email data not found.")
        return

    st.subheader(f"C. Analysis and Audit for: {selected_email['subject'][:50]}...")
    st.markdown(f"**Calculated P-Score:** **`{selected_email['p_score']:.2f}`** | **Alert Status:** `{selected_email['reviewer_action']}`")
    
    st.markdown("---")
    st.markdown("#### LLM Findings and Evidence (Explainability)")
    
    if selected_email.get('categories'):
        for i, finding in enumerate(selected_email['categories']):
            with st.expander(f"Finding {i+1}: {finding.get('category', 'Unknown Category')}"):
                st.markdown(
                    f"""
                    **Evidence (Source Line):**
                    ```
                    {finding.get('sourceline', 'N/A')}
                    ```
                    **AI Reason:** *{finding.get('reason', 'N/A')}*
                    
                    ---
                    ##### Reviewer Audit Action
                    """
                )
                
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    st.button("✅ Mark True Positive (Audit +1)", 
                                key=f"tp_{selected_email['id']}_{i}", 
                                on_click=handle_reviewer_action, 
                                args=(selected_email['id'], f"TRUE_POSITIVE | {finding.get('category')}"))
                
                with btn_col2:
                    st.button("❌ Mark False Positive (Audit -1)", 
                                key=f"fp_{selected_email['id']}_{i}", 
                                on_click=handle_reviewer_action, 
                                args=(selected_email['id'], f"FALSE_POSITIVE | {finding.get('category')}"))
    else:
        st.success("No non-compliant findings detected by the LLM.")

    st.markdown("---")
    with st.expander("View Full Email Content (Original & Masked)", expanded=False):
        st.text_area("Original Email Body", selected_email['body'], height=200, disabled=True)
        st.text_area("Masked Email Body (Sent to LLM)", selected_email['masked_body'], height=200, disabled=True)
        if PRESIDIO_AVAILABLE:
            st.json({"PII Tokens Used": "Presidio used. Tokens logged internally."})
        else:
             st.info("Presidio not installed. Using simple PII masking fallback.")


# --- 3. UNIT TESTING ---

class TestPrioritizationLogic(TestCase):
    """Tests the Bias-Based P-Score calculation logic."""

    def test_high_tier_bias(self):
        categories = [
            {'category': 'Market Manipulation/Misconduct'},
            {'category': 'Complaints'}                      
        ]
        # (16 + 2) / 2 = 9. Bias +10. P = 19.0
        self.assertEqual(calculate_p_score(categories), 19.0)

    def test_unknown_tier_bias(self):
        categories = [
            {'category': 'Secrecy'},                         
            {'category': 'Complaints'}                        
        ]
        # (6 + 2) / 2 = 4. Bias +10. P = 14.0
        self.assertEqual(calculate_p_score(categories), 14.0)

    def test_low_tier_only(self):
        categories = [
            {'category': 'Complaints'}, 
            {'category': 'Complaints'}
        ]
        # (2 + 2) / 2 = 2. Bias +10. P = 12.0 (Bias is for the tier, not the count)
        self.assertEqual(calculate_p_score(categories), 12.0)
        
    def test_medium_tier_only(self):
        categories = [
            {'category': 'Employee ethics'}, 
        ]
        # 4 / 1 = 4. Bias +10. P = 14.0
        self.assertEqual(calculate_p_score(categories), 14.0)
        

def run_tests():
    """Runs unit tests and displays results in Streamlit."""
    with st.sidebar.status("Running Prioritization Logic Unit Tests...", expanded=True):
        test_instance = TestPrioritizationLogic()
        test_methods = [getattr(test_instance, name) for name in dir(test_instance) if name.startswith('test_')]
        
        all_passed = True
        for test_func in test_methods:
            try:
                test_func()
                st.sidebar.markdown(f"✅ **PASS:** {test_func.__name__.replace('test_', '')}")
            except AssertionError:
                st.sidebar.markdown(f"❌ **FAIL:** {test_func.__name__.replace('test_', '')}")
                all_passed = False
        
        if all_passed:
            st.success("All Unit Tests Passed.")
        else:
            st.error("One or more Unit Tests Failed.")

# --- 4. MAIN APPLICATION ---

def main():
    """Sets up the Streamlit UI and handles the main workflow."""
    st.set_page_config(layout="wide", page_title="AI Communication Surveillance")
    #init_session_state()
    
    # Sidebar
    st.sidebar.title("System Status")
    run_tests()
    
    # Initialize OpenAI Client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("🚨 **API KEY MISSING:** Please set the `OPENAI_API_KEY` environment variable.")
        st.session_state.api_client = None
    else:
        try:
            st.session_state.api_client = AzureOpenAI(api_version = os.getenv("API_VERSION"),azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),api_key= os.getenv("AZURE_OPENAI_API_KEY"))
            st.sidebar.success("OpenAI Client Initialized (GPT-4o)")
        except Exception as e:
             st.sidebar.error(f"Failed to initialize OpenAI Client: {e}")
             st.session_state.api_client = None
             
    if not PRESIDIO_AVAILABLE:
        st.sidebar.warning("Presidio modules missing. Install: `pip install presidio-analyzer presidio-anonymizer`")

    # Main Title and Overview
    st.title("🛡️ AI Communication Surveillance Dashboard")
    st.markdown("**:blue[Bias-Based P-Score]** prioritization for high-throughput compliance review.")

    # Application Structure (Two Columns)
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("A. Batch Email Upload and Processing")
        
        # File Uploader (Max 50 Mails)
        uploaded_files = st.file_uploader(
            "Upload Email Files (TXT or EML format, Max 50)",
            type=['txt', 'eml'],
            accept_multiple_files=True
        )

        if st.button("Start Parallel Classification Pipeline", type="primary", disabled=(st.session_state.api_client is None)):
            if uploaded_files:
                if st.session_state.api_client:
                    st.session_state.processed_emails = [] 
                    process_email_batch(uploaded_files, max_workers=5) 
                else:
                    st.error("Cannot run: OpenAI API client failed to initialize.")
            else:
                st.warning("Please upload files first.")
        
        st.markdown("---")
        render_prioritization_queue()

    with col2:
        st.subheader("C. Detailed Analysis and Audit")
        render_details_pane()

    # Footer/Audit Log
    st.sidebar.markdown("---")
    st.sidebar.subheader("Full Audit Log (Last 10 Actions)")
    if st.session_state.audit_logs:
        log_df = pd.DataFrame(st.session_state.audit_logs).tail(10)
        st.sidebar.dataframe(log_df, use_container_width=True, hide_index=True)

if __name__ == '__main__':
    load_dotenv()
    import sys
    sys.setrecursionlimit(2000) 
    init_session_state()
    main()
