def add_reviewer_feedback(item: Dict[str, Any]):
    \"\"\"Add or update a reviewer feedback record in session and on disk (dedupe by Identifier+category).\n    item should contain: Identifier, category, action ('TP'|'FP'), timestamp, subject, sourcelines\n    \"\"\"
    if 'reviewer_feedback' not in st.session_state:
        st.session_state.reviewer_feedback = []

    # Find existing feedback for same Identifier+category
    existing = None
    for fb in st.session_state.reviewer_feedback:
        if fb.get('Identifier') == item.get('Identifier') and fb.get('category') == item.get('category'):
            existing = fb
            break

    if existing:
        # update fields
        existing['action'] = item.get('action')
        existing['timestamp'] = item.get('timestamp')
        existing['subject'] = item.get('subject', existing.get('subject', ''))
        existing['sourcelines'] = item.get('sourcelines', existing.get('sourcelines', []))
    else:
        st.session_state.reviewer_feedback.append(item)

    # persist to disk
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    path = FEEDBACK_DIR / 'reviewer_feedback.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.reviewer_feedback, f, ensure_ascii=False, indent=2)




==========

def _write_categorized_file_and_state(identifier: str, updated_obj: Dict[str, Any]):
    \"\"\"Write updated classified email to disk and update st.session_state.processed_emails in-place (if present).\"\"\"
    # write to disk
    CATEGORIZED_DIR.mkdir(parents=True, exist_ok=True)
    p = CATEGORIZED_DIR / f\"{identifier}.json\"
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(updated_obj, f, ensure_ascii=False, indent=2)

    # update in-memory processed_emails if loaded
    if 'processed_emails' in st.session_state:
        for idx, e in enumerate(st.session_state.processed_emails):
            if e.get('Identifier') == identifier:
                # replace object (keep other keys if any)
                st.session_state.processed_emails[idx] = updated_obj
                break


==========

def handle_reviewer_action(identifier: str, category: str, action_label: str):
    \"\"\"Handle reviewer click. action_label is 'TP' or 'FP'.\n    Updates per-category review state, writes reviewer feedback, updates categorized json and session.\n    \"\"\"
    # 1) Update the in-memory processed_emails object for that email (attach per-category reviews)
    target = None
    if 'processed_emails' in st.session_state:
        for e in st.session_state.processed_emails:
            if e.get('Identifier') == identifier:
                target = e
                break

    # If not found in memory, try to load from disk
    if target is None:
        p = CATEGORIZED_DIR / f\"{identifier}.json\"
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    target = json.load(f)
            except Exception:
                target = None

    if target is None:
        st.error(f\"Could not find email record for Identifier {identifier}\")
        return

    # Ensure category_reviews map exists
    if 'category_reviews' not in target:
        target['category_reviews'] = {}

    # Set or update review for this category
    # canonicalize action_label
    action_label = action_label.upper()
    if action_label not in ('TP', 'FP'):
        st.error('Invalid reviewer action')
        return

    target['category_reviews'][category] = action_label
    target['manualOverride'] = True  # mark overridden

    # Build reviewer_feedback record and persist (dedupe handled in add_reviewer_feedback)
    fb = {
        'Identifier': identifier,
        'category': category,
        'action': action_label,
        'timestamp': time.time(),
        'subject': target.get('subject',''),
        # gather sourcelines for this category if available
        'sourcelines': []
    }
    # find sourcelines from categories array if present
    for c in target.get('categories', []):
        if c.get('category') == category:
            # sourcelines may be list of dicts or strings
            sls = []
            for s in c.get('sourcelines', []) if isinstance(c.get('sourcelines', []), list) else c.get('sourcelines', []):
                if isinstance(s, dict):
                    sls.append(s.get('lines',''))
                else:
                    sls.append(str(s))
            fb['sourcelines'] = sls
            break

    add_reviewer_feedback(fb)
    simulate_db_log_action(identifier, 'REVIEWER_ACTION', {'category': category, 'action': action_label})

    # 2) Determine overall email-level result:
    # If any category reviewed as FP -> overall falsePositive True
    # Else if all categories (that exist in categories list) are TP -> overall true positive
    # Else Pending/partial -> keep falsePositive False, manualOverride True
    cat_reviews = target.get('category_reviews', {})
    categories_list = [c.get('category') for c in target.get('categories', []) if c.get('category')]
    # If there are categories predicted by LLM, only consider those; if none predicted, consider reviews keys
    if not categories_list:
        categories_list = list(cat_reviews.keys())

    any_fp = any(cat_reviews.get(c) == 'FP' for c in categories_list if c in cat_reviews)
    all_tp = categories_list and all(cat_reviews.get(c) == 'TP' for c in categories_list)

    if any_fp:
        target['falsePositive'] = True
        overall_status = 'FALSE_POSITIVE'
    elif all_tp:
        target['falsePositive'] = False
        overall_status = 'TRUE_POSITIVE'
    else:
        # partially reviewed or pending
        target['falsePositive'] = False
        overall_status = 'PENDING'

    target['reviewer_action'] = overall_status
    target['manualOverride'] = True

    # write back categorized file and update state
    _write_categorized_file_and_state(identifier, target)

    # Finally, refresh UI
    try:
        st.success(f\"Marked {category} as {action_label}\")  # small feedback
        st.rerun()
    except Exception:
        # fallback: just return (some streamlit versions might not allow immediate rerun in callback context)
        return

==============
# inside the selected email details rendering block, for each category c:
review_status = selected.get('category_reviews', {}).get(c.get('category'), 'PENDING')

def style_review(action):
    if action == "TP":
        return "display:inline-block;background:#e6ffec;color:#035f2a;padding:6px;border-radius:6px;font-weight:600;"
    if action == "FP":
        return "display:inline-block;background:#ffecec;color:#7a0606;padding:6px;border-radius:6px;font-weight:600;"
    return "display:inline-block;background:#f0f0f0;color:#333;padding:6px;border-radius:6px;"

st.markdown(f\"<div style='margin-bottom:6px'>{c.get('category')} — <span style='{style_review(review_status)}'>{review_status}</span></div>\", unsafe_allow_html=True)

# then the TP/FP buttons (pass category and id)
col_t, col_f = st.columns([1,1])
with col_t:
    st.button('✅ Mark TP', key=f\"tp_{sel}_{c.get('category')}\", on_click=handle_reviewer_action, args=(sel, c.get('category'), 'TP'))
with col_f:
    st.button('❌ Mark FP', key=f\"fp_{sel}_{c.get('category')}\", on_click=handle_reviewer_action, args=(sel, c.get('category'), 'FP'))

                   
