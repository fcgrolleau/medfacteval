import re
import html
from typing import Dict
import pandas as pd

# -------------------------------------------------------------
# HTML Generation Helpers (adapted from HTMLSummaryGenerator)
# -------------------------------------------------------------

# Constants for styling
_HEADING_PATTERNS = [
    r"One-Liner:",
    r"Brief Hospital Course:",
    r"Outstanding Issues / Follow-Up:",
    r"Relevant Medical History:",
    r"Relevant Surgical History:",
]
_PROBLEM_HEADING_RE = re.compile(r"^\s*Problem #\d+[^\n]*", re.MULTILINE)
_SUBFIELD_LABELS = [
    "Key Diagnostics & Results",
    "Therapeutic Procedures",
    "Current Status",
    "Discharge Plan & Goals",
    "Outstanding / Pending Issues",
]

_CSS = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.5;
        }

        .top-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        #copy-button {
            background-color: #0077aa;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }

        #copy-button:hover {
            background-color: #005f88;
        }

        .toggle-container {
            display: flex;
            align-items: center;
        }
        .label-text {
            margin-left: 10px;
            font-size: 16px;
        }
        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            -webkit-transition: .4s;
            transition: .4s;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            -webkit-transition: .4s;
            transition: .4s;
        }
        input:checked + .slider {
            background-color: #0077aa;
        }
        input:focus + .slider {
            box-shadow: 0 0 1px #0077aa;
        }
        input:checked + .slider:before {
            -webkit-transform: translateX(26px);
            -ms-transform: translateX(26px);
            transform: translateX(26px);
        }
        .slider.round {
            border-radius: 34px;
        }
        .slider.round:before {
            border-radius: 50%;
        }
        
        .notice {
            background-color: #fffbe6;
            border-left: 5px solid #ffc107;
            padding: 15px 20px;
            margin-bottom: 25px;
        }

        .notice strong {
            color: #b36e00;
        }

        .notice a {
            color: #0056b3;
            font-weight: bold;
            text-decoration: underline;
        }

        .citation {
            cursor: pointer;
            position: relative;
            color: #0077aa;
            text-decoration: none;
            vertical-align: baseline;
            font-size: 75%;
            top: -0.5em;
        }

        body.no-citations .citation {
            display: none;
        }

        .citation .tooltip {
            visibility: hidden;
            white-space: pre-wrap;
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            padding: 15px;
            width: 650px;
            max-width: 90vw;
            max-height: 70vh;
            overflow-y: auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            z-index: 9999;
            position: absolute;
            top: 25px;
            left: 0;
        }

        .citation:hover .tooltip {
            visibility: visible;
        }

        .citation .tooltip.tooltip-left {
            left: auto;
            right: 0;
        }

        .citation:hover::after {
            content: "";
        }
    """


def _bold_headings(text: str) -> str:
    for pattern in _HEADING_PATTERNS:
        text = re.sub(
            r"(?<!\n)(" + pattern + ")",
            lambda m: f"<br/><strong>{m.group(1)}</strong>",
            text,
        )
        text = re.sub(
            r"(\n)(" + pattern + ")",
            lambda m: f"{m.group(1)}<strong>{m.group(2)}</strong>",
            text,
        )

    def _bold_problem(match: re.Match) -> str:
        return f"<strong>{match.group(0)}</strong>"

    return _PROBLEM_HEADING_RE.sub(_bold_problem, text)


def _replace_citations(text: str, note_map: Dict[str, str]) -> str:
    tag_re = re.compile(r"\s*?(<(?:PROGRESS NOTE NO \d+)>)\s*?")

    def _sub(match: re.Match) -> str:
        tag = match.group(1)
        note_text = note_map.get(tag)

        if note_text is None:
            return ""  # Delete tag if not in the map

        if len(note_text) > 1000:
            note_text = note_text[:1000] + "..."
        escaped_note = html.escape(note_text)

        citation_html = f'<sup class="citation">†<span class="tooltip">{escaped_note}</span></sup>'

        full_match = match.group(0)
        end_of_match_pos = match.end()
        next_char = text[end_of_match_pos] if end_of_match_pos < len(text) else ""

        if next_char.isalnum():
            return citation_html + " "
        if full_match.endswith(" "):
            return citation_html + " "

        return citation_html

    parts = []
    last_end = 0
    for match in tag_re.finditer(text):
        parts.append(text[last_end : match.start()])
        parts.append(_sub(match))
        last_end = match.end()
    parts.append(text[last_end:])

    return "".join(parts)


def _style_subfields(text: str) -> str:
    joined = "|".join(map(re.escape, _SUBFIELD_LABELS))
    pattern = re.compile(rf"^\s*-\s*({joined}):", re.MULTILINE)

    def repl(m: re.Match) -> str:
        label = m.group(1)
        if label == "Key Diagnostics & Results":
            return f"• <em>{label}:</em>"
        return f"<br/>• <em>{label}:</em>"

    return pattern.sub(repl, text)


def _wrap_html(body: str) -> str:
    script = """
    <script>
        document.getElementById('citation-toggle').addEventListener('change', function() {
            if (this.checked) {
                document.body.classList.remove('no-citations');
            } else {
                document.body.classList.add('no-citations');
            }
        });

        // Helper function to provide visual feedback on the copy button.
        function showCopiedMessage() {
            const copyButton = document.getElementById('copy-button');
            if (!copyButton) return;
            const originalText = copyButton.innerText;
            copyButton.innerText = 'Copied!';
            setTimeout(function() {
                copyButton.innerText = originalText;
            }, 2000);
        }

        document.getElementById('copy-button').addEventListener('click', function() {
            const summaryEl = document.getElementById('summary-content');
            const tempEl = summaryEl.cloneNode(true);

            tempEl.querySelectorAll('.citation').forEach(function(citation) {
                citation.remove();
            });

            // Get HTML and plain text for clipboard
            const htmlToCopy = tempEl.innerHTML;
            const textToCopy = tempEl.innerText;

            // Use Clipboard API to copy both HTML and plain text.
            // The pasting application will choose the richest format it supports.
            if (navigator.clipboard && window.ClipboardItem) {
                const blobHtml = new Blob([htmlToCopy], { type: 'text/html' });
                const blobText = new Blob([textToCopy], { type: 'text/plain' });
                const clipboardItem = new ClipboardItem({
                    'text/html': blobHtml,
                    'text/plain': blobText
                });

                navigator.clipboard.write([clipboardItem]).then(showCopiedMessage).catch(err => {
                    // Fallback to plain text if the rich text copy fails
                    console.error('Rich text copy failed, falling back to plain text:', err);
                    navigator.clipboard.writeText(textToCopy).then(showCopiedMessage);
                });
            } else {
                // Fallback for older browsers: plain text only
                navigator.clipboard.writeText(textToCopy).then(showCopiedMessage);
            }
        });

        document.addEventListener('DOMContentLoaded', function() {
            const citations = document.querySelectorAll('.citation');
            citations.forEach(function(citation) {
                citation.addEventListener('mouseover', function() {
                    const tooltip = this.querySelector('.tooltip');
                    if (!tooltip) return;

                    const citationRect = this.getBoundingClientRect();
                    const viewportWidth = window.innerWidth;

                    if (citationRect.left > viewportWidth / 2) {
                        tooltip.classList.add('tooltip-left');
                    } else {
                        tooltip.classList.remove('tooltip-left');
                    }
                });
            });
        });
    </script>
""".strip()
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '    <meta charset="UTF-8"/>\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
        '    <title>MedAgentBrief Pilot</title>\n'
        f"    <style>\n{_CSS.strip()}\n    </style>\n"
        "</head>\n"
        "<body>\n"
        f"{body.strip()}\n"
        f"{script}\n"
        "</body>\n"
        "</html>"
    )

def generate_html_summary(summary: str, progress_notes_df: pd.DataFrame) -> str:
    """
    Generates a standalone HTML page from a summary and progress notes.

    Args:
        summary: The raw text summary containing citation tags like <PROGRESS NOTE NO 1>.
        progress_notes_df: A DataFrame with "content" and "note_index" columns.

    Returns:
        A string containing the full HTML document.
    """
    # Build mapping from citation tag -> note text
    note_map = {
        f"<PROGRESS NOTE NO {row['note_index']}>": row['content']
        for _, row in progress_notes_df.iterrows()
    }

    # Create the top controls and notice banner
    top_controls_html = (
        '<div class="top-controls">'
        '    <div class="toggle-container">'
        '        <label class="switch">'
        '            <input type="checkbox" id="citation-toggle" checked>'
        '            <span class="slider round"></span>'
        '        </label>'
        '        <span class="label-text">Show Citations</span>'
        '    </div>'
        '    <button id="copy-button">Copy to Clipboard</button>'
        '</div>'
    )
    
    patient_mrn = progress_notes_df.iloc[0]['mrn']
    patient_csn = progress_notes_df.iloc[0]['csn']
    
    feedback_url = f"https://docs.google.com/forms/d/e/1FAIpQLSflF0liN4yHaOhq7d0lw2JVt2DvKgo1EAWigoAysWYEmfbGxg/viewform?usp=pp_url&entry.560156158={patient_mrn}&entry.957787224={patient_csn}"
    notice_html = (
        '<div class="notice">'
        '<strong>Important Notice:</strong> This discharge summary was generated by MedAgentBrief from clinical notes. '
        'As part of our pilot study, you MUST provide feedback after using it to complete your final '
        'discharge summary or during hand-offs. Provide your feedback (<5 minutes) through '
        f'<a href="{feedback_url}" target="_blank" rel="noopener noreferrer">this link</a>. '
        'Your input is mandatory and crucial to this evaluation. Thank you!'
        '</div>'
    )

    # Process summary text
    processed = _bold_headings(summary.strip())
    processed = _style_subfields(processed)
    processed = _replace_citations(processed, note_map)
    processed = processed.replace("\n", "<br/>\n")

    summary_content_html = f'<div id="summary-content">{processed}</div>'
    html_body = f"{top_controls_html}\n{notice_html}\n{summary_content_html}"

    return _wrap_html(html_body)

def normalize_progress_note_tags(text: str) -> str:
    def replacer(match):
        tag = match.group(0)
        # Match <PROGRESS NOTE NO [number]> exactly
        m_exact = re.fullmatch(r"<PROGRESS NOTE NO (\d+)>", tag)
        if m_exact:
            return tag  # leave as is
        # Match <PROGRESS NOTE NO [number][anything else]>
        m_prefix = re.fullmatch(r"<PROGRESS NOTE NO (\d+)[^>]+>", tag)
        if m_prefix:
            number = m_prefix.group(1)
            return f"<PROGRESS NOTE NO {number}>"
        # Otherwise, remove
        return ""
    # Replace all <...> patterns
    return re.sub(r"<[^>]+>", replacer, text)

if __name__ == '__main__':
    # Minimal example to test the function
    sample_summary = """One-Liner: This is a summary.
Brief Hospital Course: The patient was admitted for observation <PROGRESS NOTE NO 1>.
Problem #1 - Acute Condition
- Key Diagnostics & Results: X-ray showed improvement <PROGRESS NOTE NO 2>.
- Discharge Plan & Goals: Follow up with specialist.
This is another part of summary <PROGRESS NOTE NO 2, NO 5>.
Relevant Medical History: None.
"""

    progress_notes_data = {
        "mrn": ["123", "123"],
        "csn": ["987", "987"],
        "note_index": [1, 2],
        "content": [
            "Note 1: Patient stable, vital signs normal. Admitted for overnight monitoring.",
            "Note 2: Chest X-ray on day 2 showed clearing of infiltrate.",
        ],
    }
    
    sample_df = pd.DataFrame(progress_notes_data)

    # Generate the HTML
    sample_summary = normalize_progress_note_tags(sample_summary)
    html_output = generate_html_summary(sample_summary, sample_df)

    # Save to a file to check the output
    with open("summary_output.html", "w", encoding="utf-8") as f:
        f.write(html_output)

    print("Generated summary_output.html with the example.")