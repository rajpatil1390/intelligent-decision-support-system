"""Reports: download a run's PDF/Markdown report, preview the Markdown (with its charts), and list past reports."""

import re

import streamlit as st

from ui import common, styles
from ui import components as ui
from ui.api_client import ApiClientError

_IMAGE = re.compile(r"!\[([^\]]*)\]\(charts/([A-Za-z0-9][A-Za-z0-9_-]*\.png)\)")
_HEADING = re.compile(r"^(#{1,4}) ", flags=re.MULTILINE)


def _demote_headings(text: str) -> str:
    """'# Title' -> '### Title', so the preview's headings sit below the page's own."""
    return _HEADING.sub(lambda m: "#" * min(len(m.group(1)) + 2, 6) + " ", text)


def render_markdown(run_id: str, text: str) -> None:
    """The report's Markdown, with its chart images (relative links) fetched from the API."""
    api = common.client()
    position = 0
    for match in _IMAGE.finditer(text):
        if text[position:match.start()].strip():
            st.markdown(text[position:match.start()])
        try:
            st.image(api.chart(run_id, match.group(2)), caption=match.group(1) or None, width=640)
        except ApiClientError as exc:
            st.caption(f"Chart {match.group(2)} unavailable: {exc.message}")
        position = match.end()
    if text[position:].strip():
        st.markdown(text[position:])


common.shell()
common.page_title("Reports", "Download the PDF or Markdown report of a finished run and preview it here.",
                  icon="📄")

api = common.client()
runs = common.fetch(api.list_runs, limit=200)
if runs is None:
    st.stop()
with_reports = [run for run in runs if run.get("report_pdf_url") or run.get("report_md_url")]
if not with_reports:
    message = ("No reports yet: a report is created when a run finishes." if runs
               else "No runs yet. Start one from New Analysis.")
    common.empty_state(message, common.PAGE_NEW, "New analysis")
    st.stop()

run_id = common.run_picker(with_reports, key="reports_run_picker", label="Run with a report")
run = next(r for r in with_reports if r["run_id"] == run_id)

buttons = st.container(horizontal=True)
if run.get("report_pdf_url"):
    pdf = common.fetch(api.report_pdf, run_id)
    if pdf is not None:
        buttons.download_button("Download PDF", data=pdf, file_name=f"idss_report_{run_id}.pdf", mime="application/pdf",
                                type="primary", icon=":material/picture_as_pdf:")
markdown = common.fetch(api.report_markdown, run_id) if run.get("report_md_url") else None
if markdown is not None:
    buttons.download_button("Download Markdown", data=markdown, file_name=f"idss_report_{run_id}.md",
                            mime="text/markdown", icon=":material/article:")

styles.section("Preview")
if markdown is None:
    st.caption("No Markdown version of this report.")
else:
    with st.container(border=True, height=650):
        render_markdown(run_id, _demote_headings(markdown))

styles.section("All reports")
frame = ui.runs_frame(with_reports)[["Run", "Question", "State", "Created"]]
frame["PDF"] = [api.url(r["report_pdf_url"]) if r.get("report_pdf_url") else None for r in with_reports]
frame["Markdown"] = [api.url(r["report_md_url"]) if r.get("report_md_url") else None for r in with_reports]
st.dataframe(frame, hide_index=True, column_config={
    "Question": st.column_config.TextColumn(width="large"),
    "PDF": st.column_config.LinkColumn(display_text="Open PDF"),
    "Markdown": st.column_config.LinkColumn(display_text="Open .md"),
})
st.caption("Pick a run above to preview its report; the links open the files straight from the backend.")
