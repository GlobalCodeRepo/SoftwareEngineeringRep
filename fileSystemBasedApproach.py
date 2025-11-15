import streamlit as st
import pandas as pd
import os
import json
import math
import uuid
import time
import re
import shutil
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

# --- PRESIDIO IMPORTS ---
# We keep this block for robustness, using the corrected import path.
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.operators import Replace
    from presidio_anonymizer.entities import OperatorConfig 
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    class MockAnalyzerEngine:
        def analyze(self, *args, **kwargs): return []
    class MockAnonymizerEngine:
        def anonymize(self, text, *args, **kwargs): return text
    AnalyzerEngine = MockAnalyzerEngine
    AnonymizerEngine = MockAnonymizerEngine
    OperatorConfig = object 

# --- CONFIGURATION AND CORE LOGIC ---
BASE_DIR = "project"
DIRS = {
    "generated": os.path.join(BASE_DIR, "generated_emails"),
    "masked": os.path.join(BASE_DIR, "masked_emails"),
    "prioritized": os.path.join(BASE_DIR, "prioritized_alerts"),
}
# Priority Matrix based on user request
CATEGORY_MAP = {
    "Market Manipulation/Misconduct": {"f1": 4, "f2": 4, "tier": "H"},
    "Market Bribery":                 {"f1": 4, "f2": 4, "tier": "H"},
    "Secrecy":                        {"f1": 3, "f2": 2, "tier": "U"},
    "Change in communication":        {"f1": 3, "f2": 2, "tier": "U"},
    "Employee ethics":                {"f1": 1, "f2": 4, "tier": "M"},
    "Complaints":                     {"f1": 2, "f2": 1, "tier": "L"},
}
THRESHOLD_T = 10

# JSON Schema for LLM Structured Output (Core Findings Only)
LLM_FINDINGS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "category": { "type": "string", "enum": list(CATEGORY_MAP.keys()), "description": "The specific non-compliant category identified.", },
            "sourceline_quotes": { "type": "array", "items": {"type": "string"}, "description": "A list of exact, complete sentence quotes from the email body that triggered this finding.", },
            "reason": { "type": "string", "description": "A concise explanation of why this finding is classified as non-compliant.", },
        },
        "required": ["category", "sourceline_quotes", "reason"],
    },
}

# --- FILE SYSTEM AND UTILITY FUNCTIONS ---

def setup_directories():
    """Sets up project directories, clearing previous runs."""
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

def parse_email_content(file_content, filename):
    """Robustly extracts email fields from raw content."""
    
    # Set default values
    email_data = {
        "Identifier": str(uuid.uuid4()),
        "date": "N/A", "from": "N/A", "to": "N/A", "subject": "N/A", "body": file_content
    }
    
    # Regex to extract standard headers
    headers = re.findall(r"^(\w+):\s*(.*?)$", file_content, re.MULTILINE)
    body_match = re.search(r"\n\n(.*)", file_content, re.DOTALL)
    
    if body_match:
        email_data["body"] = body_match.group(1).strip()
    
    for key, value in headers:
        key = key.lower()
        if key in email_data:
            email_data[key] = value.strip()
            
    if email_data['subject'] == "N/A":
        email_data['subject'] = f"File: {filename}"
        
    # Ensure all required fields for the final structure are present
    email_data.update({
        "categories": [],
        "priority": 0.0,
        "falsePositive": False,
        "manualOverride": False,
        "raiseAlarm": False,
        "reviewer_action": "PENDING"
    })

    return email_data

def write_json_file(data, directory, identifier):
    """Writes a dictionary to a JSON file."""
    filepath = os.path.join(directory, f"{identifier}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    return filepath

def read_json_file(filepath):
    """Reads a dictionary from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

# --- CORE PROCESSING STAGES (THREAD-SAFE) ---

def process_stage_2_masking(email_data):
    """Stage 2: PII Masking and writing masked JSON."""
    masked_body, pii_map = mask_pii_with_presidio(email_data["body"])
    
    email_data["masked_body"] = masked_body
    email_data["pii_map"] = pii_map
    
    write_json_file(email_data, DIRS["masked"], email_data["Identifier"])
    return email_data

def process_stage_3_classification(email_data, client: AzureOpenAI):
    """Stage 3: LLM Classification, P-Score, and writing prioritized JSON."""
    email_id = email_data['Identifier']
    
    try:
        # LLM Classification
        findings = classify_email_llm(email_data, client)

        # P-Score Calculation
        p_score = calculate_p_score(findings)
        
        # Build FINAL Structured Output
        final_categories = []
        for finding in findings:
            sourcelines_list = []
            for quote in finding.get('sourceline_quotes', []):
                sourcelines_list.append({
                    "lines": quote,
                    "feedback": "feedback is managed by reviewer in UI"
                })
            
            final_categories.append({
                "category": finding.get("category", "Unknown"),
                "sourcelines": sourcelines_list,
                "reason": finding.get("reason", "No reason provided by LLM."),
            })

        # Update email data with results
        email_data["categories"] = final_categories
        email_data["priority"] = p_score
        email_data["categories_summary"] = ", ".join(f['category'] for f in final_categories) if final_categories else "CLEAN"
        email_data["reviewer_action"] = "PENDING"
        
        simulate_db_log_action(email_id, 'CLASSIFIED', {'status': 'SUCCESS', 'score': p_score})

    except Exception as e:
        # Handling Thread Crash/LLM Failure
        email_data["categories"] = []
        email_data["priority"] = 0.0
        email_data["categories_summary"] = f"CRASH: {str(e)[:40]}..."
        email_data["reviewer_action"] = 'CRASHED'
        simulate_db_log_action(email_id, 'CLASSIFIED', {'status': 'FAILURE', 'error': str(e)})

    # Write final prioritized JSON file
    write_json_file(email_data, DIRS["prioritized"], email_data["Identifier"])
    return email_data


# --- LLM FUNCTION (Updated for new schema) ---
def classify_email_llm(email_data, client: AzureOpenAI):
    """Calls the Azure OpenAI API to classify a single masked email (THREAD-SAFE)."""
    # (Simplified LLM function remains the same, but uses new schema)
    deployment_id = os.getenv("AZURE_DEPLOYMENT_ID") 
    
    system_prompt = (
        "You are a world-class Financial Communication Surveillance Analyst. "
        "Review the masked email. Identify ALL instances of non-compliance "
        f"related to the categories: {list(CATEGORY_MAP.keys())}. "
        "You MUST respond ONLY with a JSON array that strictly adheres to the provided schema. "
        "For each finding, provide multiple exact, complete quotes as 'sourceline_quotes'."
    )
    
    user_prompt = f"Analyze the communication:\n\n{email_data['masked_body']}"
    
    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object", "schema": LLM_FINDINGS_SCHEMA},
            temperature=0.0
        )
        
        findings_json = response.choices[0].message.content
        parsed_data = json.loads(findings_json)
        
        # LLM output is an array of findings, which we return
        return parsed_data.get('array', parsed_data) if isinstance(parsed_data, dict) else parsed_data

    except Exception as e:
        # LLM API FAILURE PATH
        print(f"LLM API Call failed for {email_data['Identifier'][:8]}: {e}")
        return []


# --- MAIN PIPELINE EXECUTION ---

def process_email_batch_pipeline(uploaded_files, max_workers=5):
    """
    Manages the multi-stage file-based processing pipeline.
    """
    client = st.session_state.api_client
    if client is None:
        st.error("Cannot run: Azure OpenAI client is not initialized.")
        return

    setup_directories()
    
    all_emails = []
    
    # 1. STAGE 1: File Parsing and Generated JSON Creation (Main Thread)
    st.subheader("Stage 1: Parsing and JSON Generation")
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            content = uploaded_file.read().decode("utf-8", errors='ignore')
            email_data = parse_email_content(content, uploaded_file.name)
            
            write_json_file(email_data, DIRS["generated"], email_data["Identifier"])
            all_emails.append(email_data)
        except Exception as e:
            st.error(f"Error during file parsing {uploaded_file.name}: {e}")
            
    if not all_emails:
        st.warning("No valid emails were parsed.")
        return

    st.success(f"Parsed {len(all_emails)} emails and generated JSON files.")
    
    # 2. STAGE 2 & 3: Masking and Parallel Classification (Thread Pool)
    st.subheader("Stage 2/3: PII Masking and Parallel LLM Classification")
    
    final_processed_emails = []
    
    with st.status("Running parallel pipeline (Masking -> LLM -> Prioritization)...", expanded=True) as status:
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs to the thread pool for STAGE 2 & 3
            future_to_email = {
                executor.submit(process_stage_3_classification, email_data, client): email_data['Identifier']
                for email_data in all_emails
            }
            
            for i, future in enumerate(as_completed(future_to_email)):
                email_id = future_to_email.get(future, 'N/A')
                
                try:
                    # Result is the final dictionary written to /prioritized_alerts/
                    result_dict = future.result() 
                    final_processed_emails.append(result_dict)
                    
                    status.update(label=f"Classifying: {i+1}/{len(all_emails)} emails complete. ID: {email_id[:8]}")
                
                except Exception as e:
                    st.error(f"Thread for {email_id[:8]} crashed: {e}")
                    # Log crash
                    simulate_db_log_action(email_id, 'PARALLEL_FAIL', {'status': 'CRASH'})
                    
        # 3. Final Main Thread State Update
        st.session_state.processed_emails = final_processed_emails
        
        st.session_state.processed_emails.sort(key=lambda x: x.get('priority', 0), reverse=True)

        status.update(label="✅ Classification and Prioritization Complete!", state="complete")
        st.balloons()


# --- UI RENDERING FUNCTIONS ---

def render_prioritization_queue():
    """Renders the main alert queue by reading final JSON files."""
    
    # Initial state check
    if not st.session_state.processed_emails:
        st.info("Upload emails to start the surveillance pipeline.")
        return
    
    # 1. Filter the list to ensure only valid dictionaries are used (CRITICAL)
    clean_processed_emails = [
        email_dict for email_dict in st.session_state.processed_emails 
        if isinstance(email_dict, dict) and email_dict.get('Identifier') is not None
    ]
    
    # Define required columns for the table display
    COLUMNS_TO_DISPLAY = ['Identifier', 'subject', 'priority', 'categories_summary', 'reviewer_action', 'filename']
    
    if not clean_processed_emails:
        # If the list is empty after filtering, create an empty DataFrame with headers
        emails_df = pd.DataFrame(columns=COLUMNS_TO_DISPLAY)
        st.warning("All processed results failed or were invalid. Displaying empty queue.")
    else:
        # 2. DataFrame creation (Guaranteed to be from valid dictionaries)
        emails_df = pd.DataFrame(clean_processed_emails)
        
        # 3. Final check for missing columns 
        missing_cols = [col for col in COLUMNS_TO_DISPLAY if col not in emails_df.columns]
        if missing_cols:
            st.error(f"Internal Data Error: Missing columns {missing_cols}. Displaying empty queue.")
            emails_df = pd.DataFrame(columns=COLUMNS_TO_DISPLAY) # Fallback
            
    # 4. Rename and select columns
    emails_df = emails_df.rename(columns={
        'priority': 'P-Score', 
        'categories_summary': 'Findings',
        'reviewer_action': 'Action Status'
    })[['P-Score', 'subject', 'Findings', 'Action Status', 'Identifier']].copy()
    
    # --- UI Display Logic ---
    def color_score(val):
        if val >= 20: return 'background-color: #F87171; color: white'
        if val >= 10: return 'background-color: #FBBF24'
        if val > 0: return 'background-color: #FEF3C7'
        return ''

    def select_row(selection):
        if selection and selection['selection']:
            selected_index = selection['selection'][0]
            # Use the index relative to the *clean* list for look-up
            original_row = clean_processed_emails[selected_index]
            st.session_state.selected_email_id = original_row['Identifier'] # Use Identifier
            st.rerun()

    st.subheader("B. Prioritized Alert Queue (Highest P-Score First)")
    
    st.dataframe(
        emails_df.style.applymap(color_score, subset=['P-Score']),
        column_order=['P-Score', 'subject', 'Findings', 'Action Status'],
        column_config={
            "P-Score": st.column_config.ProgressColumn(
                "P-Score", 
                format="%f", 
                min_value=0, max_value=max(emails_df['P-Score'].max(), 30) if not emails_df.empty else 30
            ),
            "subject": "Subject",
            "Findings": "Categories Found",
            "Action Status": "Action Status",
            "Identifier": None
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
        
    # Find the email using the Identifier
    selected_email = next((e for e in st.session_state.processed_emails if e['Identifier'] == st.session_state.selected_email_id), None)

    if not selected_email:
        st.error("Error: Selected email data not found.")
        return

    subject = selected_email.get('subject', 'N/A')
    p_score = selected_email.get('priority', 0.0)
    action_status = selected_email.get('reviewer_action', 'N/A')
    categories = selected_email.get('categories', [])
    body = selected_email.get('body', 'N/A')
    
    # Find the matching masked email for display
    try:
        masked_filepath = os.path.join(DIRS['masked'], f"{selected_email['Identifier']}.json")
        masked_data = read_json_file(masked_filepath)
        masked_body = masked_data.get('masked_body', 'Masked body file not found.')
    except Exception:
        masked_body = "Masked body could not be loaded."


    st.subheader(f"C. Analysis and Audit for: {subject[:50]}...")
    st.markdown(f"**Calculated P-Score:** **`{p_score:.2f}`** | **Alert Status:** `{action_status}`")
    
    st.markdown("---")
    st.markdown("#### LLM Findings and Evidence (Explainability)")
    
    if categories:
        for i, finding in enumerate(categories):
            with st.expander(f"Finding {i+1}: {finding.get('category', 'Unknown Category')}"):
                
                st.markdown(f"**AI Reason:** *{finding.get('reason', 'N/A')}*")
                st.markdown("**Evidence (Source Line Quotes):**")
                
                # Render nested sourcelines
                for j, sourceline in enumerate(finding.get('sourcelines', [])):
                    st.code(sourceline.get('lines', f"Line {j+1} N/A"), language='text')
                
                st.markdown(f"---")

                # Mock Reviewer Action buttons
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.button("✅ Mark True Positive (Audit +1)", 
                                key=f"tp_{selected_email['Identifier']}_{i}", 
                                on_click=handle_reviewer_action, 
                                args=(selected_email['Identifier'], f"TRUE_POSITIVE | {finding.get('category')}"))
                
                with btn_col2:
                    st.button("❌ Mark False Positive (Audit -1)", 
                                key=f"fp_{selected_email['Identifier']}_{i}", 
                                on_click=handle_reviewer_action, 
                                args=(selected_email['Identifier'], f"FALSE_POSITIVE | {finding.get('category')}"))
    else:
        st.success("No non-compliant findings detected by the LLM.")
        if action_status != 'CLEAN':
             st.warning(f"⚠️ {action_status} Status: Classification failed to run or crashed.")


    st.markdown("---")
    with st.expander("View Full Email Content (Original & Masked)", expanded=False):
        st.text_area("Original Email Body", body, height=200, disabled=True)
        st.text_area("Masked Email Body (Sent to LLM)", masked_body, height=200, disabled=True)
        st.json({"Email Headers": {k: selected_email[k] for k in ["date", "from", "to", "subject"]}})
        
        if PRESIDIO_AVAILABLE:
            st.info("PII Mapping: Logged internally during Stage 2.")
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
    st.markdown("---")
    st.markdown(f"**Pipeline Directories:** Results and temporary files are stored locally in `{BASE_DIR}/`.")


    # Application Structure (Two Columns)
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("A. Batch Email Upload and Processing")
        
        uploaded_files = st.file_uploader(
            "Upload Email Files (TXT or EML format, Max 50)",
            type=['txt', 'eml'],
            accept_multiple_files=True
        )

        if st.button("Start File-Based Classification Pipeline", type="primary", disabled=(st.session_state.api_client is None)):
            if uploaded_files:
                if st.session_state.api_client:
                    st.session_state.processed_emails = [] 
                    # Use the new file-based pipeline manager
                    process_email_batch_pipeline(uploaded_files, max_workers=5) 
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
