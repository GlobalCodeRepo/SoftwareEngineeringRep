# compliance4_app.py
"""
Streamlit Email Surveillance — AzureOpenAI + validation + reviewer export
Includes:
 - per-category reviewer actions (TP/FP)
 - deduplicated reviewer feedback storage
 - immediate update of categorized json on review
 - UI highlights and per-category buttons
"""

import os
import re
import uuid
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import AzureOpenAI

import streamlit as st

# AzureOpenAI import if available (optional)
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except Exception:
    AZURE_OPENAI_AVAILABLE = False

# Optional PII libs (if installed)
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except Exception:
    PRESIDIO_AVAILABLE = False

# Optional JSON schema validator
try:
    from jsonschema import validate, ValidationError  # type: ignore
    JSONSCHEMA_AVAILABLE = True
except Exception:
    JSONSCHEMA_AVAILABLE = False

# --------------------
# Config / directories
# --------------------
BASE_DIR = Path.cwd() / "project"
GENERATED_DIR = BASE_DIR / "generated_Email"
MASKED_DIR = BASE_DIR / "masked_Email"
CATEGORIZED_DIR = BASE_DIR / "categorized_Email"
FEEDBACK_DIR = BASE_DIR / "reviewer_feedback"

for d in (GENERATED_DIR, MASKED_DIR, CATEGORIZED_DIR, FEEDBACK_DIR):
    d.mkdir(parents=True, exist_ok=True)

# concurrency
LLM_CONCURRENCY = 4

# default weights (user adjustable in UI)
DEFAULT_CATEGORY_WEIGHTS = {
    "secrecy": 9,
    "Market Manipulation/Misconduct": 10,
    "Market Bribery": 10,
    "Change in communication": 4,
    "complaints": 3,
    "Employee ethics": 6,
}

ALLOWED_CATEGORIES = set(DEFAULT_CATEGORY_WEIGHTS.keys())

# regexes for fallback masking
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s\-\.]*)?(?:\d{10}|\d{3}[\s\-\.]\d{3}[\s\-\.]\d{4}))")
CARD_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")
ACCOUNT_RE = re.compile(r"\b\d{6,18}\b")

# expected keys guard
LLM_EXPECTED_KEYS = {"Identifier", "date", "from", "to", "subject", "body", "categories", "priority", "falsePositive", "manualOverride", "raiseAlarm"}

# --------------------
# Utilities: parse / mask
# --------------------
def gen_id() -> str:
    return str(uuid.uuid4())

EMAIL_SPLIT_PATTERN = re.compile(r"(?m)^From:\s*", re.IGNORECASE)

def _extract_header_field(text: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""

def _extract_body(text: str) -> str:
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    if len(parts) > 1:
        return parts[1].strip()
    m = re.search(r"Subject:.*\n", text, re.IGNORECASE)
    if m:
        return text[m.end():].strip()
    return ""

def parse_emails_txt(raw_text: str) -> List[Dict[str,Any]]:
    parts = EMAIL_SPLIT_PATTERN.split(raw_text)
    out = []
    for p in parts:
        txt = p.strip()
        if not txt:
            continue
        if not txt.lower().startswith('from:'):
            txt = 'From: ' + txt
        out.append({
            "Identifier": gen_id(),
            "date": _extract_header_field(txt, "Date"),
            "from": _extract_header_field(txt, "From"),
            "to": _extract_header_field(txt, "To"),
            "subject": _extract_header_field(txt, "Subject") or "(No Subject)",
            "body": _extract_body(txt),
            "raw": txt
        })
    return out

def mask_text_regex(text: str) -> Tuple[str, List[Dict[str,Any]]]:
    pii_map = []
    for pattern, label in [(EMAIL_RE, "EMAIL_MASKED"), (PHONE_RE, "PHONE_MASKED"), (CARD_RE, "CARD_MASKED"), (ACCOUNT_RE, "ACCOUNT_MASKED")]:
        for m in pattern.finditer(text):
            pii_map.append({"label": label, "start": m.start(), "end": m.end(), "text": m.group(0)})
        text = pattern.sub(f"[{label}]", text)
    return text, pii_map

def mask_pii(text: str) -> Tuple[str, List[Dict[str,Any]]]:
    if PRESIDIO_AVAILABLE:
        try:
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()
            results = analyzer.analyze(text=text, entities=None, language='en')
            if not results:
                return text, []
            ops = {r.entity_type: OperatorConfig("replace", {"new_value": f"[{r.entity_type}_MASKED]"}) for r in results}
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=ops)
            pii_map = [{"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results]
            return anonymized.text, pii_map
        except Exception:
            return mask_text_regex(text)
    else:
        return mask_text_regex(text)

# --------------------
# Azure OpenAI client
# --------------------
def init_azure_client():
    if not AZURE_OPENAI_AVAILABLE:
        return None
    key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("API_VERSION")
    if not key or not endpoint:
        return None
    try:
        client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)
        return client
    except Exception:
        return None

# --------------------
# LLM prompt/call/validation
# --------------------
LLM_PROMPT_SYSTEM = (
    "You are a strict compliance classifier. Return ONLY a single JSON object exactly matching the described structure. "
    "If you're uncertain, omit the finding. Do not include any explanation text."
)

LLM_PROMPT_TEMPLATE = (
    "Allowed categories: {categories_list}.\n\n"
    "Return a JSON object with these fields: Identifier, date, from, to, subject, body, categories (array), priority (0-10 int), "
    "falsePositive (bool), manualOverride (bool), raiseAlarm (bool).\n\n"
    "Each category object MUST include: category (string), sourceline_quotes (array of strings), reason (short string).\n\n"
    "Email:\n{email_text}\n"
)

def validate_llm_output(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Top-level JSON is not an object."
    missing = LLM_EXPECTED_KEYS - set(obj.keys())
    if missing:
        return False, f"Missing keys: {missing}"
    if not isinstance(obj.get("categories"), list):
        return False, "categories must be an array."
    for c in obj.get("categories", []):
        if not isinstance(c, dict):
            return False, "category entry must be an object"
        if "category" not in c or "sourceline_quotes" not in c or "reason" not in c:
            return False, "each category entry must contain category, sourceline_quotes and reason"
        if c.get("category") not in ALLOWED_CATEGORIES:
            return False, f"Category '{c.get('category')}' not in allowed list"
    return True, "OK"

def call_azure_llm(client, email_obj: Dict[str,Any], categories_list: List[str]) -> Dict[str,Any]:
    deployment = os.getenv("AZURE_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("Azure deployment/model env var not set (AZURE_DEPLOYMENT or AZURE_OPENAI_MODEL)")
    email_text = f"""Subject: {email_obj.get('subject','')}
Body:
{email_obj.get('body','')}
"""
    prompt = LLM_PROMPT_TEMPLATE.format(categories_list=", ".join(categories_list), email_text=email_text)
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role":"system","content":LLM_PROMPT_SYSTEM}, {"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=3000
    )
    raw = resp.choices[0].message.content
    parsed = json.loads(raw)
    return parsed

# --------------------
# Classification / fallback
# --------------------
def _compute_priority_from_categories(categories: List[Dict[str,Any]], weights: Dict[str,int]) -> int:
    if not categories:
        return 0
    maxw = 0
    for c in categories:
        name = c.get("category","")
        if not name:
            continue
        w = weights.get(name, weights.get(name.lower(), 1))
        if w > maxw:
            maxw = w
    return min(10, int(maxw))

def classify_email_with_validation(client, masked_email: Dict[str,Any], weights: Dict[str,int]) -> Dict[str,Any]:
    ident = masked_email["Identifier"]
    categories_list = list(weights.keys())

    if client is not None:
        try:
            parsed = call_azure_llm(client, masked_email, categories_list)
            ok, msg = validate_llm_output(parsed)
            if not ok:
                simulate_db_log_action(ident, "LLM_INVALID", {"reason": msg})
                raise ValueError(f"LLM returned invalid structure: {msg}")
            parsed["priority"] = _compute_priority_from_categories(parsed.get("categories", []), weights)
            parsed.setdefault("falsePositive", False)
            parsed.setdefault("manualOverride", False)
            parsed.setdefault("raiseAlarm", parsed["priority"] >= 8)
            return parsed
        except Exception as e:
            simulate_db_log_action(ident, "LLM_FAIL", {"error": str(e)})

    # deterministic fallback
    body = masked_email.get("body","").lower()
    findings = []
    def rule_add(cat, kws, reason):
        for kw in kws:
            if kw in body:
                i = body.find(kw)
                start = max(0, i-120)
                end = min(len(body), i+120)
                quote = masked_email.get("body","")[start:end].strip()
                findings.append({"category": cat, "sourceline_quotes": [quote], "reason": reason})
                return True
        return False

    rule_add("Market Bribery", ["bribe", "kickback", "payoff", "commission"], "Bribery-like language")
    rule_add("Market Manipulation/Misconduct", ["insider", "non-public", "manipulate", "pump", "dump"], "Insider/manipulation")
    rule_add("secrecy", ["confidential", "secret", "internal use only", "do not share"], "Secrecy/confidential")

    categories = []
    for f in findings:
        categories.append({"category": f["category"], "sourcelines": [{"lines": s, "feedback": ""} for s in f.get("sourceline_quotes", [])], "reason": f.get("reason","")})
    priority = _compute_priority_from_categories(categories, weights)
    out = {
        "Identifier": ident,
        "date": masked_email.get("date",""),
        "from": masked_email.get("from",""),
        "to": masked_email.get("to",""),
        "subject": masked_email.get("subject",""),
        "body": masked_email.get("body",""),
        "categories": categories,
        "priority": priority,
        "falsePositive": False,
        "manualOverride": False,
        "raiseAlarm": priority >= 8
    }
    return out

# --------------------
# Pipeline worker
# --------------------
def save_json(obj: Dict[str,Any], directory: Path, identifier: str):
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{identifier}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p

def pipeline_worker(email_obj: Dict[str,Any], client, weights: Dict[str,int], sem: threading.Semaphore) -> Dict[str,Any]:
    ident = email_obj["Identifier"]
    masked_body, pii_map = mask_pii(email_obj.get("body",""))
    generated = {k: email_obj.get(k,"") for k in ("Identifier","date","from","to","subject","body")}
    generated["filename"] = email_obj.get("filename")
    save_json(generated, GENERATED_DIR, ident)
    masked_save = dict(generated)
    masked_save["masked_body"] = masked_body
    masked_save["pii_map"] = pii_map
    save_json(masked_save, MASKED_DIR, ident)

    with sem:
        result = classify_email_with_validation(client, {**generated, "body": masked_body}, weights)

    # normalize
    normalized = []
    for c in result.get("categories", []):
        srcs = []
        quotes = c.get("sourceline_quotes", c.get("sourcelines", []))
        for q in quotes:
            if isinstance(q, dict):
                lines = q.get("lines","")
                feedback = q.get("feedback","")
            else:
                lines = q
                feedback = ""
            srcs.append({"lines": lines, "feedback": feedback})
        normalized.append({"category": c.get("category",""), "sourcelines": srcs, "reason": c.get("reason","")})

    final = {
        "Identifier": ident,
        "date": generated["date"],
        "from": generated["from"],
        "to": generated["to"],
        "subject": generated["subject"],
        "body": generated["body"],
        "categories": normalized,
        "priority": _compute_priority_from_categories(normalized, weights),
        "falsePositive": result.get("falsePositive", False),
        "manualOverride": result.get("manualOverride", False),
        "raiseAlarm": result.get("raiseAlarm", False),
        "filename": generated.get("filename")
    }
    save_json(final, CATEGORIZED_DIR, ident)
    simulate_db_log_action(ident, "PIPELINE_COMPLETE", {"priority": final["priority"]})
    return final

# --------------------
# Audit / reviewer feedback helpers (NEW)
# --------------------
def simulate_db_log_action(identifier: str, action: str, meta: Dict[str,Any]):
    if "audit_logs" not in st.session_state:
        st.session_state.audit_logs = []
    st.session_state.audit_logs.append({"id": identifier, "action": action, "meta": meta, "ts": time.time()})

def _write_categorized_file_and_state(identifier: str, updated_obj: Dict[str, Any]):
    """
    Write updated classified email to disk and update st.session_state.processed_emails in-place (if present).
    """
    CATEGORIZED_DIR.mkdir(parents=True, exist_ok=True)
    p = CATEGORIZED_DIR / f"{identifier}.json"
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(updated_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        try:
            simulate_db_log_action(identifier, "WRITE_CATEGORIZED_ERR", {"error": str(e)})
        except Exception:
            pass
        raise

    # update in-memory if exists
    if "processed_emails" in st.session_state:
        for idx, e in enumerate(st.session_state.processed_emails):
            if e.get("Identifier") == identifier:
                st.session_state.processed_emails[idx] = updated_obj
                break

def add_reviewer_feedback(item: Dict[str, Any]):
    """
    Add or update a reviewer feedback record in session and on disk (dedupe by Identifier+category).
    item should contain: Identifier, category, action ('TP'|'FP'), timestamp, subject, sourcelines
    """
    if 'reviewer_feedback' not in st.session_state:
        st.session_state.reviewer_feedback = []

    existing = None
    for fb in st.session_state.reviewer_feedback:
        if fb.get('Identifier') == item.get('Identifier') and fb.get('category') == item.get('category'):
            existing = fb
            break

    if existing:
        existing['action'] = item.get('action')
        existing['timestamp'] = item.get('timestamp', time.time())
        existing['subject'] = item.get('subject', existing.get('subject', ''))
        existing['sourcelines'] = item.get('sourcelines', existing.get('sourcelines', []))
    else:
        st.session_state.reviewer_feedback.append(item)

    # persist
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    path = FEEDBACK_DIR / 'reviewer_feedback.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.reviewer_feedback, f, ensure_ascii=False, indent=2)

def handle_reviewer_action(identifier: str, category: str, action_label: str):
    """
    Handle reviewer click. action_label is 'TP' or 'FP'.
    Updates per-category review state, updates sourcelines feedback, writes reviewer feedback,
    updates categorized json and session.
    """

    # Feedback text mapping
    feedback_map = {
        "TP": "True positive / Correct prediction",
        "FP": "False positive / Manual review required"
    }

    # Find the email record in session_state or disk
    target = None
    if 'processed_emails' in st.session_state:
        for e in st.session_state.processed_emails:
            if e.get('Identifier') == identifier:
                target = e
                break

    # Fallback: load from disk
    if target is None:
        p = CATEGORIZED_DIR / f"{identifier}.json"
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    target = json.load(f)
            except Exception:
                target = None

    if target is None:
        st.error(f"Could not find email record for Identifier {identifier}")
        return

    # Ensure per-category review structure exists
    if 'category_reviews' not in target:
        target['category_reviews'] = {}

    # Normalize label
    action_label = action_label.upper()
    if action_label not in ('TP', 'FP'):
        st.error("Invalid reviewer action")
        return

    # Update per-category review
    target['category_reviews'][category] = action_label
    target['manualOverride'] = True

    # ---------------------------
    # UPDATE FEEDBACK IN SOURCELINES
    # ---------------------------
    for c in target.get("categories", []):
        if c.get("category") == category:
            for s in c.get("sourcelines", []):
                s["feedback"] = feedback_map[action_label]
            break

    # ---------------------------
    # PREPARE REVIEWER FEEDBACK OBJECT
    # ---------------------------
    fb = {
        "Identifier": identifier,
        "category": category,
        "action": action_label,
        "feedback": feedback_map[action_label],
        "timestamp": datetime.now().astimezone().isoformat(),
        "subject": target.get("subject", ""),
        "sourcelines": []
    }

    # Copy updated sourcelines
    for c in target.get("categories", []):
        if c.get("category") == category:
            lines_list = []
            for s in c.get("sourcelines", []):
                line_text = s.get("lines", "") if isinstance(s, dict) else str(s)
                lines_list.append({
                    "lines": line_text,
                    "feedback": feedback_map[action_label]
                })
            fb["sourcelines"] = lines_list
            break

    # Save reviewer feedback (deduplicates automatically)
    add_reviewer_feedback(fb)

    # Save audit log
    simulate_db_log_action(identifier, "REVIEWER_ACTION", {
        "category": category,
        "action": action_label
    })

    # ---------------------------
    # COMPUTE OVERALL EMAIL RESULT
    # ---------------------------
    cat_reviews = target.get("category_reviews", {})
    categories_list = [c.get("category") for c in target.get("categories", []) if c.get("category")]
    if not categories_list:
        categories_list = list(cat_reviews.keys())

    any_fp = any(cat_reviews.get(c) == 'FP' for c in categories_list if c in cat_reviews)
    all_tp = categories_list and all(cat_reviews.get(c) == 'TP' for c in categories_list)

    if any_fp:
        target['falsePositive'] = True
        target['reviewer_action'] = "FALSE_POSITIVE"
    elif all_tp:
        target['falsePositive'] = False
        target['reviewer_action'] = "TRUE_POSITIVE"
    else:
        target['falsePositive'] = False
        target['reviewer_action'] = "PENDING"

    target["manualOverride"] = True

    # ---------------------------
    # SAVE UPDATED CATEGORIZED FILE + SESSION UPDATE
    # ---------------------------
    try:
        _write_categorized_file_and_state(identifier, target)
    except Exception as e:
        simulate_db_log_action(identifier, "WRITE_UPDATE_ERR", {"error": str(e)})

    # ---------------------------
    # SAFE UI REFRESH (no rerun errors)
    # ---------------------------
    st.success(f"Marked {category} as {action_label}")
    st.session_state["_force_refresh"] = time.time()

def export_reviewer_feedback_download():
    fb = st.session_state.get("reviewer_feedback", [])
    if not fb:
        return None, None
    jb = json.dumps(fb, ensure_ascii=False, indent=2).encode("utf-8")
    rows = [["Identifier","category","action","timestamp","subject","sourcelines"]]
    for r in fb:
        rows.append([r.get("Identifier",""), r.get("category",""), r.get("action",""), str(r.get("timestamp","")), r.get("subject","").replace('"','""'), ";".join(r.get("sourcelines",[])).replace('"','""')])
    csv_lines = []
    for row in rows:
        csv_lines.append(",".join(f'"{col}"' if "," in str(col) or '"' in str(col) else str(col) for col in row))
    cb = ("\n".join(csv_lines)).encode("utf-8")
    return jb, cb

# --------------------
# Parallel runner
# --------------------
def run_pipeline_parallel(email_objs: List[Dict[str,Any]], client, weights: Dict[str,int], max_workers: int=6) -> List[Dict[str,Any]]:
    sem = threading.Semaphore(LLM_CONCURRENCY)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(pipeline_worker, e, client, weights, sem): e for e in email_objs}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                eid = futures[fut]["Identifier"]
                simulate_db_log_action(eid, "PIPELINE_THREAD_ERR", {"error": str(exc)})
    results.sort(key=lambda x: x.get("priority", 0), reverse=True)
    return results

def read_masked_body(identifier: str) -> str:
    p = MASKED_DIR / f"{identifier}.json"
    if not p.exists():
        return "Masked file not present"
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        return j.get("masked_body","")
    except Exception:
        return "Could not read masked body"

# --------------------
# Streamlit UI
# --------------------
def main():
    _force = st.session_state.get("_force_refresh",None)
    st.set_page_config(page_title="Email Surveillance (Azure)", layout="wide")
    st.title("AI Email Surveillance — Azure + Validation")

    if "processed_emails" not in st.session_state:
        st.session_state.processed_emails = []
    if "audit_logs" not in st.session_state:
        st.session_state.audit_logs = []
    if "reviewer_feedback" not in st.session_state:
        st.session_state.reviewer_feedback = []

    st.sidebar.header("System")
    client = init_azure_client()
    if client:
        st.sidebar.success("AzureOpenAI client available")
    else:
        st.sidebar.warning("AzureOpenAI not configured — falling back to conservative rules")

    st.sidebar.markdown("---")
    weights = DEFAULT_CATEGORY_WEIGHTS.copy()
    for k in list(weights.keys()):
        weights[k] = st.sidebar.slider(k, 0, 10, weights[k])

    global LLM_CONCURRENCY
    LLM_CONCURRENCY = st.sidebar.slider("LLM Concurrency", 1, 8, LLM_CONCURRENCY)

    uploaded = st.file_uploader("Upload emails (.txt/.eml). A single txt can contain multiple emails.", type=["txt","eml"], accept_multiple_files=True)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("Upload & Run")
        if st.button("Run Pipeline"):
            if not uploaded:
                st.warning("Please upload files")
            else:
                emails = []
                for f in uploaded:
                    raw = f.read().decode("utf-8", errors="ignore")
                    parsed = parse_emails_txt(raw)
                    for p in parsed:
                        p["filename"] = f.name
                        emails.append(p)

                st.info(f"Parsed {len(emails)} emails. Starting...")
                with st.spinner("Processing..."):
                    processed = run_pipeline_parallel(emails, client, weights, max_workers=min(12, max(2, len(emails))))
                    st.session_state.processed_emails = processed
                st.success("Pipeline finished")

        st.markdown("---")
        st.subheader("Prioritized Queue")
        if not st.session_state.processed_emails:
            st.info("No processed emails yet")
        else:
            rows = []
            for e in st.session_state.processed_emails:
                rows.append({
                    "Identifier": e["Identifier"],
                    "subject": e.get("subject","")[:80],
                    "priority": e.get("priority",0),
                    "categories": ",".join([c["category"] for c in e.get("categories",[])]),
                    "reviewer_action": e.get("reviewer_action","PENDING")
                })
            st.dataframe(rows)
            if st.button("Download categorized JSONs ZIP"):
                import io, zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    for p in CATEGORIZED_DIR.glob("*.json"):
                        z.write(p, arcname=p.name)
                buf.seek(0)
                st.download_button("Download ZIP", data=buf, file_name="categorized_emails.zip")

    with col2:
        st.subheader("Details & Review")
        sel = st.text_input("Enter Identifier to view details")
        if sel:
            selected = next((x for x in st.session_state.processed_emails if x["Identifier"] == sel), None)
            if not selected:
                st.error("Identifier not found")
            else:
                # Email header + score
                st.markdown(f"### {selected.get('subject','(no subject)')}")
                st.markdown(f"**P-Score:** {selected.get('priority')}")
                # EXPANDER (wraps categories)
                with st.expander("Categories / Evidence", expanded=True):
                    # If there are no categories, show helpful message
                    if not selected.get("categories"):
                        st.info("No categories predicted for this email.")
                    for c in selected.get("categories", []):
                        cat_name = c.get("category","")
                        review_status = selected.get("category_reviews", {}).get(cat_name, "PENDING")

                        def style_review(action):
                            if action == "TP":
                                return "display:inline-block;background:#ccffcc;color:#035f2a;padding:6px;border-radius:6px;font-weight:600;"
                            if action == "FP":
                                return "display:inline-block;background:#ffcccc;color:#7a0606;padding:6px;border-radius:6px;font-weight:600;"
                            return "display:inline-block;background:#f0f0f0;color:#333;padding:6px;border-radius:6px;"

                        st.markdown(
                            f"<div style='margin-bottom:6px'>"
                            f"<b>{cat_name}</b> — <span style='{style_review(review_status)}'>{review_status}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        col_tp, col_fp = st.columns([1,1])
                        with col_tp:
                            st.button(
                                "✅ Mark TP",
                                key=f"tp_{sel}_{cat_name}",
                                on_click=handle_reviewer_action,
                                args=(sel, cat_name, "TP"),
                            )
                        with col_fp:
                            st.button(
                                "❌ Mark FP",
                                key=f"fp_{sel}_{cat_name}",
                                on_click=handle_reviewer_action,
                                args=(sel, cat_name, "FP"),
                            )

                        # Evidence
                        for s in c.get("sourcelines", []):
                            if isinstance(s, dict):
                                st.code(s.get("lines",""))
                            else:
                                st.code(str(s))
                        st.markdown("---")

                st.markdown("---")
                st.expander("View Masked & Original (read-only)", expanded=False).write({"original": selected.get("body",""), "masked": read_masked_body(selected["Identifier"])})

    st.sidebar.header("Audit & Feedback")
    st.sidebar.markdown("Recent actions")
    st.sidebar.dataframe(st.session_state.audit_logs[-10:] if st.session_state.audit_logs else [])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Reviewer Feedback Export")
    if st.sidebar.button("Export reviewer feedback (JSON + CSV)"):
        jb, cb = export_reviewer_feedback_download()
        if jb is None:
            st.sidebar.info("No reviewer feedback yet")
        else:
            st.sidebar.download_button("Download JSON", data=jb, file_name="reviewer_feedback.json")
            st.sidebar.download_button("Download CSV", data=cb, file_name="reviewer_feedback.csv")

if __name__ == "__main__":
    load_dotenv()
    main()
