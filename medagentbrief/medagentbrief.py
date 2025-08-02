import time 
import pandas as pd
from functools import partial
import json

hashes = "#####################"
next_note = "---NEXT NOTE---"
note_header = "UNJITTERED NOTE DATE"

def simplify_dates(text, note_type):
    real_start = text.find(note_header) + len(note_header)
    real_h_p = note_type + text[real_start:]
    return real_h_p.replace(hashes, "").strip()

def extract_h_p(text):
    # Find the first occurrence of multiple #
    start = text.find(hashes) + len(hashes)
    # Find the second occurrence
    end = text.find(hashes, start)
    # Return the content between them, stripped of whitespace
    full_h_p = text[start:end].strip()
    return simplify_dates(full_h_p, "H&P").strip()

def extract_last_progress_note(text):
    # Remember that the last note is the first one to be found in the text
    
    # Find first hash occurrence
    first_hash = text.find(hashes)
    # Find second hash occurrence
    start = text.find(hashes, first_hash + len(hashes)) + len(hashes)
    # Find next note marker
    end = text.find(next_note, start)
    full_last_note = text[start:end].strip()    
    return  simplify_dates(full_last_note, "LAST PROGRESS NOTE").strip()

def extract_other_progress_notes(text):    
    # Select all progress notes except the first that appears in the text (the last by date)
    other_progress_notes = text.split("---NEXT NOTE---")[1:]
    # reverse the list to get the first note (by date) first in the list
    other_progress_notes = other_progress_notes[::-1]
    return [simplify_dates(note, "PROGRESS NOTE NO " + str(i+1)) for i, note in enumerate(other_progress_notes)]

content = """Format for Hospital Course Summary
One-Liner: Provide a concise one-line summary describing the patient case:
Example: "Mr. XX is a YY-year-old M/F with [top 3 past medical history] admitted for [Reason for Admission]."

Section-Based Summary:
Organize the initial summary into clearly numbered sections:
1. Reason for Admission: Clearly state the primary documented reason for hospitalization.
2. Relevant Medical History: Summarize significant pre-existing medical conditions pertinent to this admission.
3. Relevant Surgical History: List any prior surgeries relevant to this hospital stay.

Problem-Based Summary:
After the Section-Based Summary, structure the summary by individual medical problems as documented, using this exact template:
Hospital Course/Significant Findings by Problem:

Problem #1: [Problem Name, e.g., Pneumonia, Heart Failure Exacerbation]
- Key Diagnostic Investigations and Results: List crucial tests and significant findings.
- Therapeutic Procedures Performed: Describe significant treatments or procedures performed.
- Current Clinical Status: Briefly summarize the patient's status regarding this problem at discharge.
- Discharge Plan and Goals: Clearly state the discharge instructions, medications, and follow-up related to this problem.
- Outstanding/Pending Issues: Mention any unresolved matters or pending results.

Problem #2: [Problem Name]
- Key Diagnostic Investigations and Results:
- Therapeutic Procedures Performed:
- Current Clinical Status: 
- Discharge Plan and Goals: 
- Outstanding/Pending Issues: 

(Repeat this structure for additional problems as needed.)

Conclusion:
End with a concise, clear paragraph summarizing the patient's overall hospital course, highlighting key outcomes, and include:
Patient's condition on their last day of hospitalization."""

requirements = f"""
Additional Requirements:
- Ensure the summary is concise yet comprehensive.
- Professional Tone: Employ language appropriate for a medical document.
- Medical Terminology: Use precise medical terminology while ensuring clarity.
- Acronyms: Avoid acronyms unless they are standard in medical documentation (e.g., ECG).
- Formatting: Do not use any formatting except for plain text, spaces, new lines, and hyphens (-) as specified. Do not use Markdown.
"""

def make_prompt_1(example_input):
    """Make the initial prompt for generating the first draft of the Hospital Course Summary from the H&P and last progress note."""    
    
    prompt = f"""
Role: You are an expert AI assistant specializing in internal medicine and medical documentation. Your task is to read the provided initial History & Physical (H&P) and the final Progress Note from a patient's hospital stay and generate a concise, clear, and professional "Hospital Course Summary."
Your output must adhere strictly to the format detailed below:
{content}
{requirements}

---
Patient History and Physical (H&P):
{extract_h_p(example_input)}
---
Patient last Progress Note:
{extract_last_progress_note(example_input)}
---
Now, generate your "Hospital Course Summary" for the patient below following the guidelines and output format requirements.
"""
    return prompt

def make_prompt_2(example_input, draft, note_no):
    """Make the prompt for generating the improved draft of the Hospital Course Summary from the previous draft and the progress notes.
    Input:
    - example_input: the input text containing the H&P and progress notes
    - draft: the previous draft of the Hospital Course Summary
    - note_no: the number of the progress note to be used for updating the draft int from 0 to (no of progress notes - 1)
    """
    
    # Extract the total number of progress notes
    no_of_progress_notes = len(extract_other_progress_notes(example_input)) + 1 # +1 because extract_other_progress_notes returns all progress notes except the last one
    
    # Human (or LLM) readable note number (starts from 1)
    human_note_no = note_no + 1 
    
    # If the note_no is 0, it means that the first progress note is being used to update the draft
    if note_no == 0:
        chronological_detail_part1 = ""
    
    # If the note_no is 1, it means that the second progress note is being used to update the draft
    elif note_no == 1:
        chronological_detail_part1 = f"- Iteratively incorporated details from Progress Note #1 into that draft summary."
        
    # If the note_no is greater than 1, it means that the third or later progress note is being used to update the draft
    else:
        chronological_detail_part1 = f"- Iteratively incorporated details from Progress Notes #1 through #{note_no} into that draft summary."
        
    chronological_detail_part2 = f" and" if note_no == 0 else ""
    
    if note_no == 0:
        chronological_detail_part3 = f"."
    elif note_no == 1:
        chronological_detail_part3 = f", and has been iteratively updated to incorporate information from Progress Note #1."
    else:
        chronological_detail_part3 = f", and has been iteratively updated to incorporate information from Progress Notes #1 through #{note_no}."
        
    prompt = f"""
Role: You are an expert AI assistant specializing in internal medicine and medical documentation.
Context: You have already performed the initial steps of creating a Hospital Course Summary:
- Written a first draft using the initial History & Physical and final progress note.
{chronological_detail_part1}
Your Current Task: Your specific task now is to carefully review only Progress Note #{human_note_no} (out of {no_of_progress_notes} total notes) and update the provided draft "Hospital Course Summary" by incorporating any new or updated relevant clinical information found exclusively within that specific note.

Inputs You Will Be Provided With:
1. The Current Draft "Hospital Course Summary": This summary has been constructed using the initial History & Physical,{chronological_detail_part2} the final progress note (Note #{no_of_progress_notes}){chronological_detail_part3}
2. The Text of Progress Note #{human_note_no}: This is the only new information source you should use for this update.

Instructions:
- Focus Exclusively on Progress Note #{human_note_no}: Analyze only the provided text of Progress Note #{human_note_no}. Do not incorporate information from any other source, prior notes (unless confirming a change mentioned in {human_note_no}), or general medical knowledge beyond interpreting Note #{human_note_no}'s content.
- Identify Relevant Additions/Changes: Determine if Progress Note #{human_note_no} contains any new findings, changes in status, modifications to the treatment plan, or new diagnostic results pertinent to the existing summary sections or problems.
- Integrate Information: If relevant new information is found, seamlessly integrate it into the appropriate sections (Reason for Admission, Histories, Problem list items) of the draft summary, maintaining the required structure. This might involve adding details to existing points or slightly modifying descriptions.
- Citation Rule (MUST follow exactly): Append the tag `<PROGRESS NOTE NO {human_note_no}>` immediately after every piece of content that comes from Progress Note #{human_note_no}. Example: `Creatinine increased to 2.1 mg/dL <PROGRESS NOTE NO {human_note_no}>.`
- Never alter or remove existing tags—reproduce them exactly as found.
- Formatting Rule for Citations: The citation tag must always come after a complete statement or fact, never as part of a clause. Not Allowed: ... as noted in <PROGRESS NOTE NO {human_note_no}>. Allowed: End your statement, then add the tag—always at the end.
- Maintain Strict Format: Adhere exactly to the detailed format specified below for the output.
- No New Relevant Information: If Progress Note #{human_note_no} contains no information that adds to or significantly modifies the existing draft summary, output the original draft summary unchanged, byte-for-byte.

Required Output Format:
{content}
{requirements}
Your Draft "Hospital Course Summary":
{draft}

Extra Information from Progress Note #{human_note_no} (out of {no_of_progress_notes} total notes):
---
{extract_other_progress_notes(example_input)[note_no]}
---
Provide Your Improved "Hospital Course Summary" for the patient below following the guidelines and output format requirements.
"""
    return prompt

def make_prompt_3(example_input, draft):
    prompt = f"""
Role: You are an expert AI assistant specializing in internal medicine and medical documentation.
You have written a first draft of the Hospital Course Summary using the initial History & Physical and other progress notes. Your task is to enhance this initial summary by integrating any relevant additional details found specifically in the patient's latest progress note provided below.
- Citation Rule (MUST follow exactly): Append the tag `<LAST PROGRESS NOTE>` immediately after every piece of content that comes from the patient's last progress note. Example: `Creatinine increased to 2.1 mg/dL <LAST PROGRESS NOTE>.`
- Never alter or remove existing tags—reproduce them exactly as found.
- Formatting Rule for Citations: The citation tag must always come after a complete statement or fact, never as part of a clause. Not Allowed: ... as noted in <LAST PROGRESS NOTE>. Allowed: End your statement, then add the tag—always at the end.

Your output must adhere strictly to the format detailed below:
{content}
{requirements}
Your Draft "Hospital Course Summary":
{draft}

Extra Information from the patient last Progress Note:
---
{extract_last_progress_note(example_input)}
---
Provide Your Improved "Hospital Course Summary" for the patient below following the guidelines and output format requirements.
"""
    return prompt

reorganized_content = f"""One-Liner: Provide a concise one-line summary describing the patient case:
Example: "Mr. XX is a YY-year-old M/F with [top 3 past medical history] admitted for [Reason for Admission]."

Brief Description of Hospital Course:
Copy the unchanged concluding paragraph from the draft into this section.

Outstanding Issues/Follow-Up:
List and highlight the most critical follow-up items from the Problem-Based Summary below.

Problem-Based Summary:
Organize the summary by documented medical problems using this exact template:
Hospital Course/Significant Findings by Problem:

Problem #1: [Problem Name, e.g., Pneumonia, Heart Failure Exacerbation]
- Key Diagnostic Investigations and Results: List crucial tests and significant findings.
- Therapeutic Procedures Performed: Describe significant treatments or procedures performed.
- Current Clinical Status: Briefly summarize the patient's status regarding this problem at discharge.
- Discharge Plan and Goals: Clearly state the discharge instructions, medications, and follow-up related to this problem.
- Outstanding/Pending Issues: Mention any unresolved matters or pending results.

Problem #2: [Problem Name]
- Key Diagnostic Investigations and Results:
- Therapeutic Procedures Performed:
- Current Clinical Status: 
- Discharge Plan and Goals: 
- Outstanding/Pending Issues: 

(Repeat this structure for additional problems as needed.)

Relevant Medical History: Summarize significant pre-existing medical conditions pertinent to this admission.
Relevant Surgical History: List any prior surgeries relevant to this hospital stay.
"""

def make_prompt_4(draft):
    prompt = f"""
Role:
You are an expert AI assistant specializing in internal medicine and medical documentation. You previously generated a structured initial draft of a patient's "Hospital Course Summary" based on their initial History & Physical (H&P) and all available progress notes. Your current task is to enhance the readability and coherence of this draft by reorganizing sections into the specific improved format detailed below.
Do not add, remove, or alter any clinical facts-your task is layout/formatting only.
Reproduce the headings exactly as shown below; do not rename, add, or reorder them.

Hard Rule:
- Tag Rule - STRICT COMPLIANCE:
1. Preserve all existing tags untouched. Any tag already in the draft (e.g., <PROGRESS NOTE NO 6>, <LAST PROGRESS NOTE>) must remain verbatim. Do not edit, move, or delete them.
2. Do NOT create any new tags. This is a layout-only task—do not introduce new clinical facts, and do not add any new <LAST PROGRESS NOTE> or other tags.
3. Keep tag spacing exactly as is. One space before each tag, no punctuation between text and tag. Example (as it might already exist): Creatinine increased to 2.1 mg/dL <PROGRESS NOTE NO 48>.

- Formatting: Do not use any formatting except for plain text, spaces, new lines, and hyphens (-) as specified. Do not use Markdown.

Your output must adhere strictly to the format detailed below:
{reorganized_content}

Provided Documentation:
Your Structured Draft Hospital Course Summary:
<<<CURRENT_DRAFT_START>>>  
{draft}
<<<CURRENT_DRAFT_END>>>

Now, reorganize and provide your improved "Hospital Course Summary" following these enhanced guidelines and the required format."""
    return prompt
       
final_content = f"""One-Liner: "<Name/Initials> is a <age>-year-old <male/female/other> with <top 3 past medical history> admitted for <Reason for Admission>."

Brief Hospital Course:
<One paragraph that recounts the key events of the hospitalization in your own words, fully supported by the notes. Highlight key outcomes and state the patient's condition on the last hospital day.>

Outstanding Issues / Follow-Up:
- <critical item #1>  
- <critical item #2>  
...

Problem #1 - <Problem Name>
- Key Diagnostics & Results:
- Therapeutic Procedures:
- Current Status:
- Discharge Plan & Goals:
- Outstanding / Pending Issues:

Problem #2 - <Problem Name>
- ...

Relevant Medical History:
<bullet list>

Relevant Surgical History:
<bullet list>
"""

def make_prompt_5(example_input, draft):
    prompt = f"""
Role: You are an expert AI assistant in Internal Medicine documentation. 
Your one job: polish Hospital Course Summaries so they are 100 % faithful to the supplied medical record, no outside facts, no guess-work.

Inputs:
1. Draft Hospital Course Summary
2. Entire sequence of notes for that patient

Workflow - run these in order
1. Verification - line-by-line, confirm every statement exists verbatim (or equivalently) in the notes.
2. Elimination - delete any statement that is not directly supported.
3. Precision - fix terminology or units.
4. Clarity & Concision - tighten wording, keep a professional MD tone.
5. Length check - final output must be ≤ 5,000 characters (tags included, ≈2 pages). If needed, reorganize, synthesize, or remove info to fit.

Hard Rules:
- Section Limits:
1. Key Diagnostics & Results: max 3 bullet points.
2. Therapeutic Procedures: max 3 bullet points.
3. Current Status: max 1 bullet point.

- Tag Rule - STRICT COMPLIANCE:
1. Preserve all existing tags untouched. Any tag already in the draft (e.g., <PROGRESS NOTE NO 6>, <LAST PROGRESS NOTE>) must remain verbatim. Do not edit, move, or delete them.
2. Do NOT create any new tags. This is a layout-only task—do not introduce new clinical facts, and do not add any new <LAST PROGRESS NOTE> or other tags.
3. Keep tag spacing exactly as is. One space before each tag, no punctuation between text and tag. Example (as it might already exist): `Creatinine increased to 2.1 mg/dL <PROGRESS NOTE NO 48>`.

- Formatting: Do not use any formatting except for plain text, spaces, new lines, and hyphens (-) as specified. Do not use Markdown.

- No new clinical facts. No clinical inferences.
- Your work is layout / wording only.

Your output must adhere strictly to the format detailed below:
{final_content}

Requirements:
- Obey Hard Rules above.  
- Stay under 5 000 characters (≈ 2 pages) after formatting (tags count toward limit).

Input:
- Your Draft "Hospital Course Summary":
<<<CURRENT_DRAFT_START>>>  
{draft}
<<<CURRENT_DRAFT_END>>>

- Entire sequence of notes for that patient: 
<<<NOTES_START>>>
{extract_h_p(example_input)}
{"\n\n".join(extract_other_progress_notes(example_input))}
\n\n{extract_last_progress_note(example_input)}
<<<NOTES_END>>>

Output Stub (fill in):

One-Liner:
...

Brief Hospital Course:
...

Outstanding Issues / Follow-Up:
...

Problem #1 - <Problem Name>
...

Relevant Medical History:
...

Relevant Surgical History:
...
"""
    return prompt

def make_prompt_for_relevance_check(example_input):
    prompt = f"""
Role: You are an expert AI assistant specializing in internal medicine and medical documentation. Your task is to review a series of hospital progress notes and identify any that are not medically relevant for creating a hospital course summary.

Instructions:
1.  Review each progress note provided below.
2.  Identify notes that contain little to no substantive clinical information. These may include:
    - Notes that only state "patient seen and examined."
    - Notes that are duplicates of other notes.
    - Notes that are administrative in nature (e.g., billing inquiries).
    - Notes that are extremely brief and add no new information to the patient's case.
3.  For each non-medically relevant note you identify, provide its title and a brief explanation for why it is not relevant.

Your output should be a JSON object containing a list of irrelevant notes. Each item in the list should have an "explanation" and a "title".

For example:
[
  {{
    "title": "PROGRESS NOTE NO 4: 2023-05-08 17:49:00",
    "explanation": "This note only states 'patient seen and examined' and provides no new clinical information."
  }},
  {{
    "title": "PROGRESS NOTE NO 15: 2023-05-11 14:05:00",
    "explanation": "This note is a duplicate of a previous note."
  }}
]

If all notes are medically relevant, please return an empty list:
[]

Here are the notes to review:
---

{"\n\n".join(extract_other_progress_notes(example_input))}
---
Now, please identify the non-medically relevant notes in JSON format.
"""
    return prompt

def get_non_medically_relevant_notes(gen_txt_to_txt, example_input):
    """
    Calls an LLM to identify non-medically relevant notes from a patient's record.
    """
    prompt = make_prompt_for_relevance_check(example_input)
    response = gen_txt_to_txt(prompt)
    
    if response.startswith("```json"):
        response = response.split("```json")[1].split("```")[0]
        
    if response.startswith("```JSON"):
        response = response.split("```JSON")[1].split("```")[0]
        
    return response.strip()

def generate_summary(gen_txt_to_txt, gen_txt_to_txt_lc, example_input, verbose):
    
    # get non-medically relevant notes
    non_relevant_notes_str = get_non_medically_relevant_notes(gen_txt_to_txt, example_input)
    if non_relevant_notes_str.split()[0] == "Failed":
        if verbose:
            print("The notes are too long to fit the context window of the model, using long context LLM backup...")
        non_relevant_notes_str = get_non_medically_relevant_notes(gen_txt_to_txt_lc, example_input)
    
    try:    
        non_relevant_notes = json.loads(non_relevant_notes_str)
    except json.JSONDecodeError:
        print("The response is not a valid JSON.")
        non_relevant_notes = []
        
    if verbose:
        if len(non_relevant_notes) > 0:
            print(f"Found {len(non_relevant_notes)} non-medically relevant notes:")
            for note in non_relevant_notes:
                print(f"- {note['title']}: {note['explanation']}")
        else:
            print("No non-medically relevant notes found.")
            
    title_of_non_relevant_notes = [note['title'] for note in non_relevant_notes]
    
    n_irrelevant_notes = sum([title_of_non_relevant_note in "\n\n".join(extract_other_progress_notes(example_input)) for title_of_non_relevant_note in title_of_non_relevant_notes])
    if verbose:
        print(f"Number of correctly formatted non-medically relevant notes: {n_irrelevant_notes} of {len(non_relevant_notes)}")
    
    drafts = []
    
    # generate first draft
    if verbose:
        print("Generating first draft...")
        print(make_prompt_1(example_input)) 
    previous_draft = gen_txt_to_txt(make_prompt_1(example_input))
    drafts.append((0, previous_draft))

    # generate improved drafts by iteratively incorporating details from progress notes 1,2,3, ... not including the last note
    no_of_notes = len(extract_other_progress_notes(example_input))
    for i in range(no_of_notes):
        prompt_2 = make_prompt_2(example_input, previous_draft, i)
        if any(title in prompt_2 for title in title_of_non_relevant_notes):
            print(f"Skipping progress note {i+1} because it is non-medically relevant.")
            print(prompt_2)
            continue
        if verbose:
            print(f"Generating draft from progress note {i+1}/{no_of_notes} (of which {n_irrelevant_notes} irrelevant notes will be skipped)...")
            print(prompt_2) 
        previous_draft = gen_txt_to_txt(prompt_2)
        drafts.append((i+1, previous_draft))
        
    # generate improved draft by incorporating details from the last progress note
    if verbose: print(f"Generating draft from the last note...")
    print(make_prompt_3(example_input, previous_draft)) 
    previous_draft = gen_txt_to_txt(make_prompt_3(example_input, previous_draft))
    drafts.append((no_of_notes+1, previous_draft))
    
    # generate improved draft by reorganizing the summary
    if verbose:
        print("Reorganizing the summary...")
        print(make_prompt_4(previous_draft))
    previous_draft = gen_txt_to_txt(make_prompt_4(previous_draft))
    drafts.append((no_of_notes+2, previous_draft))
    
    # generate final output
    if verbose:
        print("Removing potential hallucinations...")
        print(make_prompt_5(example_input, previous_draft)) 
    final_output = gen_txt_to_txt(make_prompt_5(example_input, previous_draft))
    
    # if the notes are too long, use the long context model
    if final_output.split()[0] == "Failed":
        if verbose:
            print("The notes are too long to fit the context window of the model, using long context LLM backup...")
            print(make_prompt_5(example_input, previous_draft))
        final_output = gen_txt_to_txt_lc(make_prompt_5(example_input, previous_draft))
        
    drafts.append((no_of_notes+3, final_output))
    
    return drafts, final_output

class April_DC_summarizer:
    def __init__(self, model_init, model_call, model_init_lc, model_call_lc):
        self.model_init = model_init
        self.model_call = model_call
        self.model_init_dict = model_init()
        self.model_call_lc = model_call_lc
        self.model_init_lc_dict = model_init_lc()
        
    def _gen_txt_to_txt(self, input_txt):
        return self.model_call(input_txt, **self.model_init_dict)
    
    def _gen_txt_to_txt_lc(self, input_txt):
        return self.model_call_lc(input_txt, **self.model_init_lc_dict)

    def summarize(self, example_input, verbose=True):
        self.example_input = example_input
        self.total_no_of_notes = 2+len(extract_other_progress_notes(example_input)) # +2 for (i) H&P and (ii) the last progress note
        if verbose:
            print(f"""A total of {self.total_no_of_notes} notes need to be summarized for this patient.""")
        
        start = time.time()
        self.drafts, self.final_output = generate_summary(self._gen_txt_to_txt, self._gen_txt_to_txt_lc, example_input, verbose)
        end = time.time()
        self.time_to_summarize = end - start # in seconds
        
if __name__ == "__main__":
    import pandas as pd
    from llm_calls import lab_key, gemini_shc_init, gemini_shc_call, openai_init, openai_call, meta_init, meta_call
    from functools import partial

    # Main LLM summarizer
    gpt41_init = partial(openai_init, "gpt-4.1", lab_key)
    gemini_shc_init = partial(gemini_shc_init, "gemini-2.5-pro-preview-05-06", lab_key)

    # Long context model as a backup (Llama 4 Scout context window is 10M tokens... 10x larger than GPT-4.1)
    #llama_init = partial(meta_init, "llama4-scout", lab_key)

    # Load the testset
    testset = pd.read_pickle('path/to/clinical_note/dataset')

    # nb: patient with the most notes >200 notes is patient 40
    import concurrent.futures
    import pickle
    from html_summary_generator import HTMLSummaryGenerator
    
    def process_patient(ex_i):
        example_input = testset["inputs"].iloc[ex_i]

        # initialize the summarizer
        patient_ins = April_DC_summarizer(gemini_shc_init, gemini_shc_call, gpt41_init, openai_call)

        # summarize the patient
        patient_ins.summarize(example_input, verbose=True)
        print(patient_ins.final_output)

        # Save the patient_ins object
        with open(f'../pickles/patient_{ex_i}_ins.pkl', 'wb') as f:
            pickle.dump(patient_ins, f)

        # Write the final output to file
        with open(f'../summaries/txt/summary_{ex_i}_output.txt', 'w') as f:
            f.write(patient_ins.final_output)
            
        html_gen = HTMLSummaryGenerator(patient_ins)
        html_str = html_gen.to_html()

        with open(f"../summaries/html/summary_{ex_i}.html", "w") as f:
            f.write(html_str)
        
        return f"Processed patient {ex_i}"
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_patient, ex_i) for ex_i in range(10,21)]
        
        # Wait for all tasks to complete and collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
                

    def pickle_to_html(ex_i):
        # Load the patient_ins object
        with open(f'../pickles/patient_{ex_i}_ins.pkl', 'rb') as f:
            patient_ins = pickle.load(f)
            
        html_gen = HTMLSummaryGenerator(patient_ins)
        html_str = html_gen.to_html()

        with open(f"../summaries/html/new_summary_{ex_i}.html", "w") as f:
            f.write(html_str)
        
        return f"Processed patient {ex_i}"