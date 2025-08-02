import time 

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

reorganized_content_and_requirements = f"""One-Liner: Provide a concise one-line summary describing the patient case:
Example: "Mr. XX is a YY-year-old M/F with [top 3 past medical history] admitted for [Reason for Admission]."

Brief Description of Hospital Course:
Move your existing concluding paragraph here to clearly summarize the overall hospital course and key outcomes.

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

Additional Requirements:
- Ensure your summary is concise yet comprehensive, not exceeding 2 pages (approximately 5000 characters).
- Maintain a professional tone appropriate for medical documentation.
- Employ precise medical terminology clearly and accurately.
- Avoid acronyms unless standard in medical documentation (e.g., ECG).
- Format the entire output in markdown, and include no text before or after the markdown content.
"""
       
def make_single_prompt(all_input_notes):
    prompt = f"""
Role: You are a specialist AI assistant in internal medicine, charged with writing a Hospital Course Summary STRICTLY from the patient’s provided records. No outside inference or extra knowledge—only what’s in the notes.

FORMAT YOUR OUTPUT IN MARKDOWN ONLY (no extra text before/after).

{reorganized_content_and_requirements}

Input:
Entire sequence of notes for that patient: 
---
{extract_h_p(all_input_notes)}
{"\n\n".join(extract_other_progress_notes(all_input_notes))}
\n\n{extract_last_progress_note(all_input_notes)}
---

Output:
Provide Your Improved "Hospital Course Summary" for the patient below following the guidelines and output format requirements.
"""
    return prompt

def generate_summary(gen_txt_to_txt, gen_txt_to_txt_lc, all_input_notes, verbose):
    
    # generate the summary with a single prompt
    if verbose:
        print("Generating the summary with a single prompt...")
    final_draft = gen_txt_to_txt(make_single_prompt(all_input_notes))
    
    # if the notes are too long, use the long context model
    if final_draft.split()[0] == "Failed":
        if verbose: print("The notes are too long to fit the context window of the model, using long context LLM backup...")
        final_draft = "Required using long context model as backup...\n" + gen_txt_to_txt_lc(make_single_prompt(all_input_notes))
        
    return final_draft

class SP_DC_summarizer:
    """
    Class to summarize the hospital course of a patient based on their medical notes with a SINGLE PROMPT.
    """
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
        self.no_of_notes = 2+len(extract_other_progress_notes(example_input))
        if verbose:
            print(f"""A total of {self.no_of_notes} notes need to be summarized for this patient.""")
        start = time.time()
        self.final_draft = generate_summary(self._gen_txt_to_txt, self._gen_txt_to_txt_lc, example_input, verbose)
        end = time.time()
        self.time_to_summarize = end - start # in seconds
    
if __name__ == "__main__":
    import pandas as pd
    
    # Load the testset
    testset = pd.read_pickle('../../pickle/train_test_dfs/test_id_df.pkl')
    
    # Select an example input
    ex_i = 15
    example_input = testset["inputs"].iloc[ex_i]
    
    # Print the example input (physician's H&P and progress notes)
    #print(example_input)
    
    #### With VertexAI - gemini 2.5 Pro Exp ####
    # For HIPAA compliance, everything remains in our Google Cloud Project
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../mykeys/grolleau_application_default_credentials.json'
    os.environ['GCLOUD_PROJECT'] = 'som-nero-phi-jonc101'
    
    def model_init():
        # Overwrite this function to use another model
        model_name="gemini-2.5-pro-exp-03-25"
        from vertexai.preview.generative_models import GenerativeModel
        return {"loaded_model": GenerativeModel(model_name)}
    
    def model_lc_init():
        # Overwrite this function to use another model
        model_name="gemini-2.0-flash-001"
        from vertexai.preview.generative_models import GenerativeModel
        return {"loaded_model": GenerativeModel(model_name)}

    def model_call(input_txt, **kwargs):
        # Overwrite this function to use another model
        ready_model = kwargs["loaded_model"]
        response = ready_model.generate_content([input_txt])
        return response.candidates[0].content.parts[0].text  
    
    # Instantiate the DC_summarizer class
    DC_summary_example = SP_DC_summarizer(model_init, model_call, model_lc_init, model_call)
    
    # Generate the final draft
    DC_summary_example.summarize(example_input)

    # Print the final draft
    print(DC_summary_example.final_draft)
    
    # Compare to the physician's summary
    print(testset["brief_hospital_course"].iloc[ex_i])