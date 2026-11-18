"""IDSS Streamlit frontend – entry point (start with scripts/run_ui.ps1 or `streamlit run ui/app.py`).

Registers the pages of ui/pages/ with st.navigation and runs the selected one. Every page starts with
`common.shell()` (wide layout, styling, banner, sidebar status). The UI talks only to the FastAPI backend
through ui/api_client.py (base URL: env IDSS_API_URL).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # `streamlit run ui/app.py` only puts ui/ on the path
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from ui import common  # noqa: E402

common.page_config()
navigation = st.navigation([st.Page(path, title=title, icon=icon, default=index == 0)
                            for index, (path, title, icon) in enumerate(common.PAGES)])
navigation.run()
