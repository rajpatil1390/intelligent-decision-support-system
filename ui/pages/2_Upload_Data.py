"""Upload Data: send a file to POST /datasets, pick an existing dataset, and inspect its profile."""

import hashlib

import streamlit as st

from ui import common
from ui import components as ui
from ui import styles

FILE_TYPES = ["csv", "xlsx", "json", "parquet"]  # what the backend accepts

common.shell()
common.page_title("Upload data", f"Upload a dataset (max {common.fmt_num(common.max_upload_mb())} MB) or pick one you "
                                 "uploaded before. IDSS profiles every column and suggests what to predict.", icon="📤")

api = common.client()

with st.container(border=True):
    st.markdown("**📁 Upload a new file**")
    upload = st.file_uploader("CSV, Excel, JSON or Parquet", type=FILE_TYPES, key="upload_file")
    if upload is not None:
        content = upload.getvalue()
        signature = hashlib.sha256(content).hexdigest()
        already = st.session_state.get("uploaded_signatures", {}).get(signature)
        if already:
            st.caption(f"This file was already uploaded in this session (dataset {already}).")
        if st.button("Upload and profile", type="primary", key="upload_button"):
            with st.spinner("Uploading and profiling…"):
                result = common.fetch(api.upload_dataset, upload.name, content)
            if result is not None:
                st.session_state.setdefault("uploaded_signatures", {})[signature] = result["id"]
                common.set_current_dataset(result["id"])
                st.success(f"Uploaded **{result['filename']}**: {result['rows']:,} rows × {result['cols']} columns.",
                           icon=":material/check_circle:")

datasets = common.fetch(api.list_datasets)
if datasets is None:
    st.stop()
if not datasets:
    common.empty_state("No datasets yet. Upload a file above, for example data/samples/customer_churn.csv.")
    st.stop()

styles.section("Dataset profile")
dataset_id = common.dataset_picker(datasets, key="upload_dataset_picker", label="Dataset")
profile = common.fetch(api.dataset_profile, dataset_id) if dataset_id else None
if profile is None:
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{profile['n_rows']:,}")
col2.metric("Columns", f"{profile['n_cols']:,}")
col3.metric("Duplicate rows", f"{profile.get('duplicate_rows', 0):,}")
col4.metric("Memory", f"{common.fmt_num(profile.get('memory_mb'))} MB")

with st.container(border=True):
    target = profile.get("suggested_target")
    task = profile.get("suggested_task_type")
    left, right = st.columns(2)
    left.markdown(f"🎯 **Suggested target:** `{target}`" if target else "🎯 **Suggested target:** none found")
    right.markdown(f"🧠 **Suggested task:** {common.task_label(task)}" if task else "🧠 **Suggested task:** —")
    if profile.get("time_column"):
        st.markdown(f"**Time column:** `{profile['time_column']}` (complete, unique and increasing)")
    st.caption("Your question decides the final target and task when an LLM is configured; "
               "otherwise these suggestions are used.")

for warning in profile.get("warnings") or []:
    st.warning(warning, icon=":material/warning:")

styles.section("Columns")
st.caption("🎯 = suggested target")
ui.profile_table(profile)

st.page_link(common.PAGE_NEW, label="Next: ask a question about this dataset", icon=":material/arrow_forward:")
