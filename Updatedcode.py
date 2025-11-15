import streamlit as st
import pandas as pd
import os
import json
import math
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import TestCase, main, TestLoader, TextTestRunner 
from io import StringIO
from dotenv import load_dotenv

# --- LLM Client Imports and Setup ---
try:
    from openai import AzureOpenAI
except ImportError:
    st.error("The 'openai' library is not installed. Please run: pip install openai")
    AzureOpenAI = None 

# --- PRESIDIO IMPORTS (CLEANED) ---
try:
    # Use explicit imports only from the known correct packages
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.operators import Replace
    from presidio_anonymizer.entities import OperatorConfig 
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    # Mock classes to prevent runtime errors if library is missing
    class MockAnalyzerEngine:
        def analyze(self, *args, **kwargs):
            return []
    class MockAnonymizerEngine:
        def anonymize(self, text, *args, **kwargs):
            return text
    AnalyzerEngine = MockAnalyzerEngine
    AnonymizerEngine = MockAnonymizerEngine
    OperatorConfig = object 

# --- CORE LOGIC FUNCTION: P-Score Calculation ---

# (CATEGORY_MAP and THRESHOLD_T must be defined before calculate_p_score)
CATEGORY_MAP = {
    "Market Manipulation/Misconduct": {"f1": 4, "f2": 4, "tier": "H"},
    "Market Bribery":                 {"f1": 4, "f2": 4, "tier": "H"},
    "Change in communication":        {"f1": 3, "f2": 2, "tier": "U"},
    "Secrecy":                        {"f1": 3, "f2": 2, "tier": "U"},
    "Employee ethics":                {"f1": 1, "f2": 4, "tier": "M"},
    "Complaints":                     {"f1": 2, "f2": 1, "tier": "L"},
}
THRESHOLD_T = 10

def calculate_p_score(categories):
    """Calculates the P-Score based on the non-compliance matrix formula."""
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


# --- 3. UNIT TESTING (MOVED TO TOP) ---

class TestPrioritizationLogic(TestCase):
    # ... (Tests remain the same)
    def test_high_tier_bias(self):
        categories = [{'category': 'Market Manipulation/Misconduct'}, {'category': 'Complaints'}]
        self.assertEqual(calculate_p_score(categories), 19.0)
    def test_unknown_tier_bias(self):
        categories = [{'category': 'Secrecy'}, {'category': 'Complaints'}]
        self.assertEqual(calculate_p_score(categories), 14.0)
    def test_low_tier_only(self):
        categories = [{'category': 'Complaints'}, {'category': 'Complaints'}]
        self.assertEqual(calculate_p_score(categories), 22.0)
    def test_medium_tier_only(self):
        categories = [{'category': 'Employee ethics'}]
        self.assertEqual(calculate_p_score(categories), 14.0)
        

def run_tests():
    """Runs unit tests and displays results in Streamlit."""
    stream = StringIO()
    loader = TestLoader()
    suite = loader.loadTestsFromTestCase(TestPrioritizationLogic)
    runner = TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    test_output = stream.getvalue()
    summary = f"Tests Run: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}"
    return summary, test_output

# --- 1. CONFIGURATION (Schema moved here) ---
CATEGORY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "category": { "type": "string", "enum": list(CATEGORY_MAP.keys()), "description": "The specific non-compliant category identified in the text.", },
            "sourceline": { "type": "string", "description": "The exact sentence or phrase from the email body that triggers the non-compliance alert. Must be a direct quote.", },
            "reason": { "type": "string", "description": "A concise explanation of why this source line is non-compliant under this category.", },
        },
        "required": ["category", "sourceline", "reason"],
    },
}

# --- DATABASE PERSISTENCE (THREAD-SAFE SIMULATION) ---
def simulate_db_log_action(email_id, action, details):
    log_entry = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email_id": email_id,
        "action": action,
        "details": details.get('status', str(details)),
    }
    print(f"DB AUDIT LOG: {email_id[:8]} | Action: {action} | Details: {details.get('status', 'N/A')}")
    return log_entry

def simulate_db_store_processed_email(email_data):
    print(f"DB WRITE: Email {email_data['id'][:8]} ready for final storage. P-Score: {email_data['p_score']:.2f}")
    return email_data 


# --- PII MASKING ---
def mask_pii_with_presidio(email_body):
    pii_map = {}
    if not PRESIDIO_AVAILABLE:
        replacements = { "John Smith": "[PERSON_A]", "Jane Doe": "[PERSON_B]", "account_1234567890": "[ACC_NUM]", "ext-party@corp.com": "[EMAIL_EXT]", }
        masked_body = email_body
        for original, token in replacements.items():
            masked_body = masked_body.replace(original, token)
        return masked_body, pii_map
    
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(
        text=email_body, 
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE", "CREDIT_CARD", "US_DRIVER_LICENSE"],
        language='en'
    )
    # FIX: Correct configuration for the replace operator using string literal
    anonymized_results = anonymizer.anonymize(
        text=email_body,
        analyzer_results=results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<" + "[ENTITY_TYPE]" + ">"})}
    )
    masked_body = anonymized_results.text
    for res in results:
        entity_text = email_body[res.start:res.end]
        pii_map[f"<[[{res.entity_type}]]>"] = entity_text 
    return masked_body, pii_map


# --- LLM CLASSIFICATION (THREAD-SAFE) ---
def classify_email_llm(email_data, client: AzureOpenAI):
    email_id = email_data['id']
    masked_body = email_data['masked_body']
    deployment_id = os.getenv("AZURE_DEPLOYMENT_ID") 
    if not deployment_id:
        raise ValueError("AZURE_DEPLOYMENT_ID environment variable not set.")
    
    system_prompt = (
        "You are a world-class Financial Communication Surveillance Analyst. Your task is to review the following masked email and identify ALL instances of non-compliance "
        f"related to the following financial categories: {list(CATEGORY_MAP.keys())}. "
        "You MUST respond ONLY with a JSON array that strictly adheres to the provided schema. If the email is fully compliant, return an empty JSON array []."
    )
    user_prompt = f"Analyze the communication:\n\n{masked_body}"
    
    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object", "schema": CATEGORY_SCHEMA},
            temperature=0.0
        )
        findings_json = response.choices[0].message.content
        parsed_data = json.loads(findings_json)
        findings = parsed_data.get('array', parsed_data) if isinstance(parsed_data, dict) else parsed_data
        
        simulate_db_log_action(email_id=email_id, action='CLASSIFIED', details={'status': 'SUCCESS', 'findings_count': len(findings)})
        return findings

    except Exception as e:
        simulate_db_log_action(email_id=email_id, action='CLASSIFIED', details={'status': 'FAILURE', 'error': str(e)})
        return []


def process_single_email_pipeline(email_data_dict, client: AzureOpenAI):
    """Executes the full pipeline for one email in a thread."""
    email_id = email_data_dict['id']
    
    # 1. Initialize ALL required keys to safe defaults
    email_data_dict['masked_body'] = email_data_dict.get('masked_body', email_data_dict['body'])
    email_data_dict['pii_map'] = email_data_dict.get('pii_map', {})
    email_data_dict['categories'] = []
    email_data_dict['p_score'] = 0.0
    email_data_dict['categories_summary'] = "INIT FAILED"
    email_data_dict['reviewer_action'] = 'ERROR'

    try:
        masked_body, pii_map = mask_pii_with_presidio(email_data_dict['body'])
        email_data_dict['masked_body'] = masked_body
        email_data_dict['pii_map'] = pii_map

        findings = classify_email_llm(email_data_dict, client)

        p_score = calculate_p_score(findings)
        
        # Finalize Data Structure (SUCCESS PATH)
        email_data_dict['categories'] = findings
        email_data_dict['p_score'] = p_score
        email_data_dict['categories_summary'] = ", ".join(f['category'] for f in findings) if findings else "CLEAN"
        email_data_dict['reviewer_action'] = 'PENDING'

    except Exception as e:
        # Catch critical error and update failure summary
        email_data_dict['categories'] = []
        email_data_dict['p_score'] = 0.0
        email_data_dict['categories_summary'] = f"CRASH: {str(e)[:40]}..."
        email_data_dict['reviewer_action'] = 'CRASHED'
        print(f"THREAD CRITICAL EXCEPTION for {email_id[:8]}: {e}")

    return simulate_db_store_processed_email(email_data_dict)


# --- STREAMLIT STATE AND UI LOGIC ---

def init_session_state():
    """Initializes all necessary session variables."""
    if 'processed_emails' not in st.session_state: st.session_state.processed_emails = [] 
    if 'audit_logs' not in st.session_state: st.session_state.audit_logs = [] 
    if 'selected_email_id' not in st.session_state: st.session_state.selected_email_id = None
    if 'api_client' not in st.session_state: st.session_state.api_client = None


def handle_reviewer_action(email_id, action_type):
    """Handles reviewer feedback (True/False Positive) and updates audit."""
    log_entry = simulate_db_log_action(
        email_id=email_id, 
        action=action_type, 
        details={"status": "Reviewer Confirmation"}
    )
    st.session_state.audit_logs.append(log_entry)
    
    for email in st.session_state.processed_emails:
        if email['id'] == email_id:
            email['reviewer_action'] = action_type
            break
            
    st.success(f"Audit log updated: Email {email_id[:8]} marked as {action_type}.")
    st.session_state.selected_email_id = None 
    st.rerun()

def process_email_batch(uploaded_files, max_workers=5):
    """The main looper function. Executes the parallel pipeline."""
    if not uploaded_files:
        st.warning("Please upload at least one email file to begin processing.")
        return

    client = st.session_state.api_client
    if client is None:
        st.error("Cannot run: Azure OpenAI client is not initialized.")
        return

    emails_to_process = []
    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.read().decode("utf-8", errors='ignore')
            
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

    final_processed_emails = []
    
    with st.status("Processing emails in parallel...", expanded=True) as status:
        st.write(f"Starting parallel classification for {len(emails_to_process)} emails (Max workers: {max_workers})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_email = {
                executor.submit(process_single_email_pipeline, email_data, client): email_data['id']
                for email_data in emails_to_process
            }
            
            for i, future in enumerate(as_completed(future_to_email)):
                email_id = future_to_email.get(future, 'N/A')
                
                try:
                    result = future.result() 
                    final_processed_emails.append(result) 
                    
                    status.update(label=f"Classifying: {i+1}/{len(emails_to_process)} emails complete.")
                
                except Exception as e:
                    st.error(f"A processing thread crashed for Email ID {email_id[:8]}: {e}")
                    
                    # Crash safety net: Create a GUARANTEED safe dictionary
                    safe_fail_dict = {
                        'id': email_id, 'filename': future_to_email.get(future, 'N/A'),
                        'subject': f"FATAL ERROR: {str(e)[:30]}...", 'body': f"Thread crashed: {str(e)}",
                        'categories': [], 'p_score': 0.0,
                        'categories_summary': "THREAD CRASHED", 'reviewer_action': 'CRASHED',
                        'pii_map': {}, 'masked_body': "N/A"
                    }
                    final_processed_emails.append(safe_fail_dict)
                    st.session_state.audit_logs.append(simulate_db_log_action(
                        email_id=email_id, action='PARALLEL_FAIL', details={'status': 'CRASH'}
                    ))
                    
        st.session_state.processed_emails = final_processed_emails
        st.session_state.processed_emails.sort(key=lambda x: x.get('p_score', 0), reverse=True)

        status.update(label="✅ Classification and Prioritization Complete!", state="complete")
        st.balloons()


def render_prioritization_queue():
    """Renders the main alert queue (DataFrame)."""
    if not st.session_state.processed_emails:
        st.info("Upload emails to start the surveillance pipeline.")
        return
    
    # CRITICAL FIX: Ensure ALL items are dictionaries before creating DataFrame
    clean_processed_emails = [
        email_dict for email_dict in st.session_state.processed_emails 
        if isinstance(email_dict, dict)
    ]
    
    if not clean_processed_emails:
        st.warning("All processed results were invalid. Please check logs for errors.")
        return
        
    emails_df = pd.DataFrame(clean_processed_emails)
    
    display_cols = ['p_score', 'subject', 'filename', 'categories_summary', 'reviewer_action', 'id']
    
    # Final check for missing columns (should be redundant now, but good defense)
    missing_cols = [col for col in display_cols if col not in emails_df.columns]
    if missing_cols:
        st.error(f"Internal Data Error: Missing columns {missing_cols} in DataFrame source. Check thread processing stability.")
        return

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
            # We must use the index relative to the *clean* list for look-up
            selected_index = selection['selection'][0]
            original_row = clean_processed_emails[selected_index]
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
        width='stretch',
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

    subject = selected_email.get('subject', 'N/A')
    p_score = selected_email.get('p_score', 0.0)
    action_status = selected_email.get('reviewer_action', 'N/A')
    categories = selected_email.get('categories', [])

    st.subheader(f"C. Analysis and Audit for: {subject[:50]}...")
    st.markdown(f"**Calculated P-Score:** **`{p_score:.2f}`** | **Alert Status:** `{action_status}`")
    
    st.markdown("---")
    st.markdown("#### LLM Findings and Evidence (Explainability)")
    
    if categories:
        for i, finding in enumerate(categories):
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
        if action_status != 'CLEAN':
             st.warning(f"⚠️ {action_status} Status: Classification failed to run or crashed.")


    st.markdown("---")
    with st.expander("View Full Email Content (Original & Masked)", expanded=False):
        st.text_area("Original Email Body", selected_email.get('body', 'N/A'), height=200, disabled=True)
        st.text_area("Masked Email Body (Sent to LLM)", selected_email.get('masked_body', 'N/A'), height=200, disabled=True)
        
        if PRESIDIO_AVAILABLE:
            st.json({"PII Tokens Used": "Presidio used. Tokens logged internally."})
        else:
             st.info("Presidio not installed. Using simple PII masking fallback.")


def main():
    """Sets up the Streamlit UI and handles the main workflow."""
    st.set_page_config(layout="wide", page_title="AI Communication Surveillance")
    
    # Sidebar
    st.sidebar.title("System Status")
    
    # Run tests and display results in the sidebar
    with st.sidebar.status("Running Prioritization Unit Tests...", expanded=True) as status:
        summary, output = run_tests()
        st.sidebar.markdown(f"**{summary}**")
        with st.sidebar.expander("Show Test Execution Details"):
            st.code(output, language='text')

        if "Failures: 0, Errors: 0" in summary:
            status.update(label="✅ All Prioritization Tests Passed!", state="complete", expanded=False)
        else:
            status.update(label="❌ Tests Failed! Check Details.", state="error", expanded=True)
            
    # Initialize OpenAI Client
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") 
    
    if not api_key:
        st.error("🚨 **API KEY MISSING:** Please set the `AZURE_OPENAI_API_KEY` environment variable.")
        st.session_state.api_client = None
    else:
        try:
            st.session_state.api_client = AzureOpenAI(
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key
            )
            st.sidebar.success("Azure OpenAI Client Initialized")
        except Exception as e:
             st.sidebar.error(f"Failed to initialize Azure OpenAI Client: {e}")
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
                    st.error("Cannot run: Azure OpenAI client failed to initialize.")
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
        st.sidebar.dataframe(log_df, width='stretch', hide_index=True)


# --- EXECUTION ---
if __name__ == '__main__':
    load_dotenv()
    import sys
    sys.setrecursionlimit(2000) 
    init_session_state() 
    main()
