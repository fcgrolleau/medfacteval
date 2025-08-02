import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
from pathlib import Path
from agnostic_evaluator_models import API_text_to_text

def llm_output_to_json(llm_output, k=5):
    reformated_response =  llm_output.replace("\n", " ").replace("*", " ").replace("-", " ").replace("```", "").replace("json", "")
    reformated_response = reformated_response[:-10] + reformated_response[-10:].replace(",", "")
    for i in range(k):
        if i == k-1:
            return {}
        try:
            return json.loads(reformated_response)
        except Exception as e:
            e = str(e)
            coma_pos = int(e[e.find("char ")+5:].replace(")", ""))
            reformated_response = reformated_response[:coma_pos] + "," + reformated_response[coma_pos:]

def make_fact_eval_prompt(proto_ds, fact):
    prompt = f"""You are an expert AI assistant specializing in internal medicine. Your task is to analyze a provided hospital course summary and determine whether a specific, important fact is explicitly mentioned.
    Output your response as a valid JSON object in the following format:

    ```json
    {{
        "explanation": "Detailed, step-by-step reasoning process explaining why the fact is or is not present in the hospital course summary.",
        "fact_mentioned": integer  // 1 if the fact is explicitly mentioned, 0 if not.
    }}

    Guidelines:
    1. Reasoning: Provide a clear and concise explanation of your thought process in the "explanation" field. Break down your analysis into logical steps, showing how you searched for the information and what led you to your conclusion. Explain why you believe the fact is or is not mentioned.
    2. Fact Determination: The "fact_mentioned" field must be either 1 or 0.
        - Use 1 when the summary states the fact in any clear way—verbatim, paraphrased, abbreviated (e.g., “MI” for myocardial infarction), or with obvious clinical synonyms.  
            Do NOT demand proof of causality, full context, or every ancillary detail; if a busy physician would say "yep, that's in there," mark 1.  
        - Use 0 only when the fact (or a plainly equivalent phrase) is nowhere in the text. Minor wording gaps, lack of supporting labs, or incomplete explanations should NOT by themselves trigger a 0.
    3. JSON Format:
        Adhere strictly to the JSON format provided. Do not include any surrounding text or markdown.
        Do not nest JSON objects within each other.
        The entire response must be a single, valid JSON object enclosed in curly braces `{"}"}`.
        Ensure proper key-value pairing and use of quotation marks.
    4. No Extraneous Output: Output only the JSON object. Do not include any introductory or concluding sentences, greetings, or other text outside the JSON.

    --- Important Fact to look for ---
    {fact}

    --- Hospital Course Summary ---
    {proto_ds}"""
    return prompt

def make_inconsistency_fact_eval_prompt(proto_ds, fact):
    prompt = f"""You are an expert AI assistant specializing in internal medicine. Read the Hospital Course Summary and decide whether it contradicts the Important Fact.
    Return one valid JSON object with exactly these keys:

    ```json
    {{
        "explanation": "concise, step-by-step justification of your verdict",
        "summary_inconsistent_with_fact": <0 | 1>
    }}

    1. How to decide the flag
    - Use 1 when the summary clearly conflicts with the fact (both cannot be true).
    - Use 0 when the summary is consistent with the fact or does not mention it.

    2. Explanation guidelines
    - Point to the specific sentences/phrases that drive your decision.
    - Keep it logical and orderly; bullet-style or numbered steps are fine.
    - Do not output any text besides the JSON object.

    3. Formatting rules
    - Use the exact key names shown above (summary_inconsistent_with_fact is spelled correctly).
    - Adhere strictly to the JSON format provided. Do not include any surrounding text or markdown.
    - Do not nest JSON objects within each other.
    - The entire response must be a single, valid JSON object enclosed in curly braces `{"}"}`.
    - Ensure proper key-value pairing and use of quotation marks.

    --- Important Fact to check for contradiction with hospital course summary ---
    {fact}

    --- Hospital Course Summary ---
    {proto_ds}"""
    return prompt

def make_llm_as_judge_prompt(proto_ds):
    prompt = f"""You are an AI assistant, acting as a senior internal medicine physician evaluating the quality of a hospital course summary. Your task is to analyze the provided summary and assign it a quality score from 1 to 10, where 10 represents the highest possible quality.
    Your response MUST be formatted as valid JSON according to the following schema:

    ```json
    {{
    "explanation": "string",
    "score": integer
    }}

    Instructions:

    1. Reasoning Process (Explanation): In the "explanation" field, meticulously detail your reasoning for the assigned score. Specifically address the following aspects of the hospital course summary:
        - Completeness: Does the summary include all essential elements of a standard hospital course (e.g., presenting problem, pertinent history, key exam findings, labs/imaging results, treatment plan, response to treatment, consultations, discharge plan, discharge medications, follow-up instructions)? Are any crucial details missing?
        - Accuracy: Is the information presented factually correct and consistent with expected clinical findings given the patient's condition? Are there any contradictions or inconsistencies within the summary?
        - Clarity & Conciseness: Is the summary written in clear, concise, and unambiguous language? Is it free of jargon and unnecessary details? Is the timeline of events easy to follow?
        - Organization: Is the summary logically organized, allowing for easy understanding of the patient's hospital stay?
        - Appropriateness of Detail: Does the summary include the right level of detail? Is it overly verbose or too brief to be informative? Does it focus on the most relevant information?
        - Justification of score: Explicitly explain how each of the above aspects influenced the score.
    2. Scoring (Score): Assign a single integer value from 1 to 10, inclusive, to the "score" field. Use the following guidelines to anchor your scoring:
        - 1-3: Unacceptable. The summary is significantly incomplete, inaccurate, poorly written, and/or disorganized. It provides little to no useful information.
        - 4-6: Below Average. The summary has significant flaws in completeness, accuracy, clarity, or organization. Requires substantial revisions.
        - 7-8: Good. The summary is generally well-written, accurate, and complete, with only minor areas for improvement.
        - 9-10: Excellent. The summary is comprehensive, accurate, clear, concise, and well-organized. It provides a high-quality overview of the patient's hospital course.
    3. Output Format: Adhere STRICTLY to the JSON schema provided above. Ensure the JSON object is well-formed and contains only the "explanation" and "score" fields. Do not include any extraneous text or conversational elements outside the JSON object.
    4. Important Restrictions:
        - No Nested JSON: Do NOT embed a JSON object within another JSON object.
        - Single JSON Object: Provide only ONE JSON object as your response.
        - No Trailing Text: Do NOT add any text or comments after the closing curly bracket `{"}"}`.
        
    --- Hospital Course Summary to evaluate ---
    {proto_ds}"""
    return prompt

def icc_two_way_absolute_agreement(grader_1_scores, grader_2_scores):
    x_res = np.vstack([grader_1_scores, grader_2_scores]).T
    # ICC(2,1) in Shrout and Fleiss Psychological Bulletin 1979
    ss_s = 2*sum((x_res.mean(axis=1) - x_res.mean())**2)
    ss_e = sum((x_res[:, 0] - x_res[:, 1])**2/2)
    df_s = len(x_res)-1 
    df_e = len(x_res)
    ms_s = ss_s / df_s
    ms_e = ss_e / df_e
    return (ms_s - ms_e) / (ms_s + ms_e)

def boot_icc(all_human_scores, all_llm_scores, alpha=.05, n_boot=9999):
    """
    Bootstrap confidence intervals for the ICC(2,1) agreement:
    bootstrap is at the patient level and at the fact level
    """
    boot_res = []
    for _ in range(n_boot):
        n_patients = all_human_scores.shape[0]

        rand_patients = np.random.choice(range(n_patients), size=n_patients, replace=True)
        rand_idxs = list(zip(np.repeat(np.arange(n_patients), 3), np.random.choice(range(3), size=n_patients*3, replace=True)))

        rand_all_human_scores = all_human_scores[rand_patients]
        rand_all_human_scores_rand_facts = np.array([rand_all_human_scores[rand_idxs[i]] for i in range(len(rand_idxs))]).reshape(n_patients, 3)

        rand_all_llm_scores = all_llm_scores[rand_patients]
        rand_all_llm_scores_rand_facts = np.array([rand_all_llm_scores[rand_idxs[i]] for i in range(len(rand_idxs))]).reshape(n_patients, 3)

        icc_i = icc_two_way_absolute_agreement(rand_all_llm_scores_rand_facts.mean(axis=1), rand_all_human_scores_rand_facts.mean(axis=1))
        boot_res.append(icc_i)
    return np.percentile(boot_res, (100*(alpha/2), 100*(1-alpha/2))), boot_res

def bootstrap_kappa(data_a, data_b, n_boot=1000, alpha=0.05):
    """
    Calculate bootstrapped Cohen's kappa statistic with confidence intervals.
    
    Parameters:
    -----------
    data_a : array-like
        First rater's scores (0s and 1s)
    data_b : array-like
        Second rater's scores (0s and 1s)
    n_boot : int, default=1000
        Number of bootstrap iterations
    alpha : float, default=0.05
        Significance level for confidence intervals (e.g., 0.05 for 95% CI)
    
    Returns:
    --------
    tuple
        (kappa value, [lower CI, upper CI], bootstrap samples)
    """
    from sklearn.metrics import cohen_kappa_score
    import numpy as np
    
    # Calculate original kappa
    kappa_orig = cohen_kappa_score(data_a, data_b)
    
    if n_boot == 0:
        return kappa_orig
    
    # Prepare bootstrap samples
    n_samples = len(data_a)
    kappa_boot = []
    
    # Bootstrap iterations
    for _ in range(n_boot):
        # Sample with replacement
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        boot_a = np.array(data_a)[indices]
        boot_b = np.array(data_b)[indices]
        
        # Skip bootstrap samples with only one class (would raise error in kappa)
        if len(np.unique(boot_a)) > 1 and len(np.unique(boot_b)) > 1:
            kappa = cohen_kappa_score(boot_a, boot_b)
            kappa_boot.append(kappa)
    
    # Calculate confidence intervals
    lower_ci = np.percentile(kappa_boot, 100 * alpha / 2)
    upper_ci = np.percentile(kappa_boot, 100 * (1 - alpha / 2))
    
    return kappa_orig, [lower_ci, upper_ci], kappa_boot

class AutoEval:
    def __init__(self, evaluator_pair, evaluated_model="gpt-4o"):
        self.evaluator_pair = evaluator_pair
        self.evaluated_model = evaluated_model
        self.test_path = Path("../prototyping/generated_dc_sum/testset")
        self.fact_df_path = Path("../../exports/fact_data/benchmark_creation - all_responses.csv")

        # Read the facts
        fact_df = pd.read_csv(self.fact_df_path)
        self.facts = pd.DataFrame()
        self.facts = fact_df.iloc[:,[1,2,3,5,7]].copy() # Timestamp, physician email, important fact 1, important fact 2, important fact 3
        self.facts.loc[:,'patient_i'] = fact_df.iloc[:,0].apply(lambda x: int(''.join(filter(str.isdigit, str(x))))) - 1 
        
        # Read the proto summaries
        proto_files = list(self.test_path.rglob(f"**/{evaluated_model}.md"))
        self.proto_summaries = []
        for file_path in proto_files:
            patient_i = [p for p in file_path.parts if "patient_" in p][0]
            patient_n = int(''.join(filter(str.isdigit, patient_i))) 
            with open(file_path, 'r', encoding='utf-8') as f:
                self.proto_summaries.append((patient_n, f.read().split("\n\n", 1)[1]))
        self.proto_summaries = pd.DataFrame(self.proto_summaries, columns=['patient_i', 'summary'])
        # Include only the proto summaries of patients that are in the fact_df
        self.proto_facts_merged = pd.merge(self.proto_summaries, self.facts, left_on='patient_i', right_on='patient_i', how='inner')
        # Reorder to patient_i 0, 1, 2, 3, ... for clarity
        self.proto_facts_merged = self.proto_facts_merged.sort_values(by="patient_i").reset_index(drop=True)
        
    def _evaluate_facts(self, prompt_fn, result_key, default_value, result_attr_prefix, verbose=False):
        """
        Helper method to evaluate facts using a given prompt function and parameters
        
        Args:
            prompt_fn: Function to generate the prompt
            result_key: Key to extract from LLM output JSON
            default_value: Default value if LLM fails to provide a score
            result_attr_prefix: Prefix for result attributes ('fact' or 'inc_fact')
            verbose: Whether to print verbose output
        """
        start_time = time.time()
        llm_instance = API_text_to_text(*self.evaluator_pair)
        
        # Initialize result dictionaries
        results = {}
        explanations = {}
        
        for patient_i in range(len(self.proto_facts_merged)):
            id = self.proto_facts_merged["patient_i"][patient_i]
            results[f'patient_{id}'] = results.get(f'patient_{id}', {}) 
            explanations[f'patient_{id}'] = explanations.get(f'patient_{id}', {}) 
            
            email = self.proto_facts_merged["Email Address"][patient_i].replace("@", "_")
            timestamp = self.proto_facts_merged["Timestamp"][patient_i]
            results[f'patient_{id}'][f"{email}<>{timestamp}"] = {}
            explanations[f'patient_{id}'][f"{email}<>{timestamp}"] = {}
            
            for fact_j in range(3):
                proto_summary = self.proto_facts_merged.iloc[patient_i, 1]
                fact = self.proto_facts_merged.iloc[patient_i, 4+fact_j]  # the 4 first columns are: "patient_i", "summary", "Timestamp", "Email Address" then comes the facts
                
                # Generate prompt using provided function
                eval_prompt = prompt_fn(proto_summary, fact)
                llm_output = llm_instance.gen_txt_to_txt(eval_prompt)
                
                # Process output
                output_json = llm_output_to_json(llm_output)
                results[f'patient_{id}'][f"{email}<>{timestamp}"][f'fact_{fact_j}'] = float(output_json.get(result_key, float(default_value)))
                
                explanation = output_json.get("explanation", 'LLM judge failed to provide an explanation (and likely failed to provide a score)')
                if isinstance(explanation, list) and isinstance(explanation[0], str):
                    explanation = " ".join(explanation)  # if the explanation is a list of strings, join them
                    
                explanations[f'patient_{id}'][f"{email}<>{timestamp}"][f'fact_{fact_j}'] = explanation
                
                if verbose:
                    try: 
                        print(f'Evaluated patient {patient_i+1}/{len(self.proto_facts_merged)}, fact {fact_j+1}: "{fact.replace("\n", " ")}", prototype summary (names & dates have been altered for privacy): "{proto_summary.replace("\n", " ")[:200]}..."')
                        print(f" - Response: {results[f'patient_{id}'][f"{email}<>{timestamp}"][f'fact_{fact_j}']}")
                        print(f" - Explanation: {explanations[f'patient_{id}'][f"{email}<>{timestamp}"][f'fact_{fact_j}'].replace("\n", " ")}\n")
                    except Exception as e:
                        print(f"Failed to print status of evaluation for patient {patient_i+1}/{len(self.proto_facts_merged)}, fact {fact_j+1}: {e}")
                        print(f"LLM output: {llm_output}")
        
        # Average the results when multiple physicians provided facts the same patient
        results_avg = {k: pd.DataFrame(results[k]).mean(axis=1).to_dict() for k in results.keys()}
        
        # Average across the 3 facts for each patient
        scores = {k: np.mean(list(v.values())) for k, v in results_avg.items()}
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Store results in class attributes
        setattr(self, f"{result_attr_prefix}_eval_res", results)
        setattr(self, f"{result_attr_prefix}_eval_expl", explanations)
        setattr(self, f"{result_attr_prefix}_eval_res_avg", results_avg)
        setattr(self, f"{result_attr_prefix}_eval_scores", scores)
        setattr(self, f"{result_attr_prefix}_eval_time", eval_time)
        
        print(f"Fact evaluation ({result_attr_prefix}) took {eval_time:.2f} seconds.\n")
    
    def fact_eval(self, verbose=False):
        self._evaluate_facts(
            prompt_fn=make_fact_eval_prompt,
            result_key="fact_mentioned",
            default_value=1.0,  # 1 is the default value if the LLM fails to provide a score
            result_attr_prefix="fact",
            verbose=verbose
        )
        
    def inc_fact_eval(self, verbose=False):
        self._evaluate_facts(
            prompt_fn=make_inconsistency_fact_eval_prompt,
            result_key="summary_inconsistent_with_fact",
            default_value=0.0,  # 0 is the default value if the LLM fails to provide a score
            result_attr_prefix="inc_fact",
            verbose=verbose
        )
        
    def unconditional_eval(self):
        llm_instance = API_text_to_text(*self.evaluator_pair)
        self.unc_eval_res = {}
        self.unc_eval_expl = {}
        for patient_i in range(len(self.proto_facts_merged)):
            id = self.proto_facts_merged["patient_i"][patient_i]
            self.unc_eval_res[f'patient_{id}'] = {}
            self.unc_eval_expl[f'patient_{id}'] = {}
            proto_summary = self.proto_facts_merged.iloc[patient_i, 1]
            unconditional_eval_prompt = make_llm_as_judge_prompt(proto_summary)
            llm_output = llm_instance.gen_txt_to_txt(unconditional_eval_prompt)
            self.unc_eval_res[f'patient_{id}'] = float(llm_output_to_json(llm_output).get("score", float('nan')))
            self.unc_eval_expl[f'patient_{id}'] = llm_output_to_json(llm_output).get("explanation", float('nan'))

class AutoEvalReport:
    def __init__(self, auto_eval_instance, evaluator_pair=None):
        """
        Initialize the report generator with an AutoEval instance
        
        Args:
            auto_eval_instance: An instance of AutoEval with completed evaluations
            evaluator_pair: Optional LLM pair to use for summarizing explanations.
                           If None, will use the same evaluator as the AutoEval instance
        """
        self.auto_eval = auto_eval_instance
        self.evaluator_pair = evaluator_pair or self.auto_eval.evaluator_pair
        
    def _get_explanation_summary(self, fact_eval_type="fact"):
        """
        Summarize explanations using an LLM
        
        Args:
            fact_eval_type: Type of evaluation to summarize ("fact" or "inc_fact")
        """
        # Create LLM instance for summarization
        llm_instance = API_text_to_text(*self.evaluator_pair)
        
        # Extract all relevant explanations
        relevant_explanations = []
        
        explanations = getattr(self.auto_eval, f"{fact_eval_type}_eval_expl")
        for patient_id in explanations:
            for reviewer in explanations[patient_id]:
                for fact_id in explanations[patient_id][reviewer]:
                    # Get corresponding score to check filter condition
                    patient_key = patient_id
                    reviewer_key = reviewer
                    fact_key = fact_id
                    
                    patient_number = int(patient_key.split('_')[1])
                    fact_number = int(fact_id.split('_')[1])
                    
                    # Find the matching score in eval_res to apply filter
                    if fact_eval_type == "fact" and hasattr(self.auto_eval, 'fact_eval_res') and patient_key in self.auto_eval.fact_eval_res:
                        score = self.auto_eval.fact_eval_res[patient_key][reviewer_key][fact_key]
                        if score == 0:  # Filter condition for fact_eval_type == "fact"
                            # get the fact
                            fact = self.auto_eval.proto_facts_merged.iloc[patient_number, 4+fact_number]
                            # get the explanation
                            explanation = explanations[patient_key][reviewer_key][fact_key]
                            relevant_explanations.append(f'- Patient {patient_number}, fact {fact_number} ("{fact}") is absent.\nExplanation: {explanation}.\n')
                            
                    elif fact_eval_type == "inc_fact" and hasattr(self.auto_eval, 'inc_fact_eval_res') and patient_key in self.auto_eval.inc_fact_eval_res:
                        score = self.auto_eval.inc_fact_eval_res[patient_key][reviewer_key][fact_key]
                        if score == 1:
                            # get the fact
                            fact = self.auto_eval.proto_facts_merged.iloc[patient_number, 4+fact_number]
                            # get the explanation
                            explanation = explanations[patient_key][reviewer_key][fact_key]
                            relevant_explanations.append(f'- Patient {patient_number}, fact {fact_number} ("{fact}") is inconsistent.\nExplanation: {explanation}.\n')
        
        # If no relevant explanations found
        if not relevant_explanations:
            return "No explanations to summarize."
        
        if fact_eval_type == "fact":    
            # Create prompt for the LLM to summarize explanations for missing facts
            prompt = f"""You are an expert medical AI. Below are detailed explanations for why hospital course summaries are missing important clinical facts. Summarize these explanations concisely (one paragraph) by highlighting common failure patterns and key insights into why these omissions occur. Include illustrative examples referencing specific patient numbers and missing facts to clarify your points.
            
            --- Explanations ---
            {chr(10).join(relevant_explanations)}

            --- Guidelines for Response ---
            1. Clearly identify common themes and patterns causing clinical omissions.
            2. Keep the summary concise and precise, ideally limited to one structured paragraph.
            3. Always reference specific patient numbers and facts as provided.
            4. Focus exclusively on summarizing patterns and explanations provided, without suggesting solutions or recommendations."""

        elif fact_eval_type == "inc_fact":
            # Create prompt for the LLM to summarize explanations for inconsistent facts
            prompt = f"""You are an expert medical AI. Below are detailed explanations for why hospital course summaries are inconsistent with important clinical facts. Summarize these explanations concisely (one paragraph) by highlighting common failure patterns and key insights into why these inconsistencies occur. Include illustrative examples referencing specific patient numbers and inconsistent facts to clarify your points.
            
            --- Explanations ---
            {chr(10).join(relevant_explanations)}

            --- Guidelines for Response ---
            1. Clearly identify common themes and patterns causing clinical inconsistencies.
            2. Keep the summary concise and precise, ideally limited to one structured paragraph.
            3. Always reference specific patient numbers and facts as provided.
            4. Focus exclusively on summarizing patterns and explanations provided, without suggesting solutions or recommendations.
            """
            
        # Call LLM to summarize
        summary = llm_instance.gen_txt_to_txt(prompt)
        return summary
    
    def _calculate_mean_scores(self, results):
        """Calculate mean scores across all patients"""
        all_scores = []
        for patient_id, patient_data in results.items():
            for reviewer, reviewer_data in patient_data.items():
                for fact_id, score in reviewer_data.items():
                    all_scores.append(score)
        return np.mean(all_scores) if all_scores else float('nan')
    
    def _get_all_relevant_explanations(self, fact_eval_type="fact"):
        """
        Get all relevant explanations for a specific evaluation type
        
        Args:
            fact_eval_type: Type of evaluation to extract explanations for ("fact" or "inc_fact")
        """
        relevant_explanations = []
        
        explanations = getattr(self.auto_eval, f"{fact_eval_type}_eval_expl")
        for patient_id in explanations:
            for reviewer in explanations[patient_id]:
                for fact_id in explanations[patient_id][reviewer]:
                    # Get corresponding score to check filter condition
                    patient_key = patient_id
                    reviewer_key = reviewer
                    fact_key = fact_id
                    
                    patient_number = int(patient_key.split('_')[1])
                    fact_number = int(fact_id.split('_')[1])
                    
                    # Find the matching score in eval_res to apply filter
                    if fact_eval_type == "fact" and hasattr(self.auto_eval, 'fact_eval_res') and patient_key in self.auto_eval.fact_eval_res:
                        score = self.auto_eval.fact_eval_res[patient_key][reviewer_key][fact_key]
                        if score == 0:  # Filter condition for fact_eval_type == "fact"
                            # get the fact
                            fact = self.auto_eval.proto_facts_merged.iloc[patient_number, 4+fact_number]
                            # get the explanation
                            explanation = explanations[patient_key][reviewer_key][fact_key]
                            relevant_explanations.append((patient_number, fact_number, fact, explanation, score))
                            
                    elif fact_eval_type == "inc_fact" and hasattr(self.auto_eval, 'inc_fact_eval_res') and patient_key in self.auto_eval.inc_fact_eval_res:
                        score = self.auto_eval.inc_fact_eval_res[patient_key][reviewer_key][fact_key]
                        if score == 1:  # Filter condition for fact_eval_type == "inc_fact"
                            # get the fact
                            fact = self.auto_eval.proto_facts_merged.iloc[patient_number, 4+fact_number]
                            # get the explanation
                            explanation = explanations[patient_key][reviewer_key][fact_key]
                            relevant_explanations.append((patient_number, fact_number, fact, explanation, score))
        
        return relevant_explanations
        
    def _create_fact_table(self, fact_eval_type="fact"):
        """Create detailed table with patient-by-fact scores"""
        table = []
        table.append("| Patient | Fact 0 | Fact 1 | Fact 2 |")
        table.append("|---------|--------|--------|--------|")
        
        # Get the data to use for the table
        if fact_eval_type == "fact":
            eval_res = self.auto_eval.fact_eval_res
        else:
            eval_res = self.auto_eval.inc_fact_eval_res
            
        # Process each patient
        for patient_id in sorted(eval_res.keys(), key=lambda x: int(x.split('_')[1])):
            patient_num = int(patient_id.split('_')[1])
            
            # Initialize aggregation dictionary for facts
            fact_scores = {0: [], 1: [], 2: []}
            
            # Collect all scores for each fact across reviewers
            for reviewer in eval_res[patient_id]:
                for fact_key, score in eval_res[patient_id][reviewer].items():
                    fact_num = int(fact_key.split('_')[1])
                    fact_scores[fact_num].append(score)
            
            # Calculate average for each fact
            fact0_avg = sum(fact_scores[0])/len(fact_scores[0]) if fact_scores[0] else "N/A"
            fact1_avg = sum(fact_scores[1])/len(fact_scores[1]) if fact_scores[1] else "N/A"
            fact2_avg = sum(fact_scores[2])/len(fact_scores[2]) if fact_scores[2] else "N/A"
            
            # Format scores
            fact0_str = "Yes" if fact0_avg == 1.0 else "No" if fact0_avg == 0.0 else f"{fact0_avg:.2f}"
            fact1_str = "Yes" if fact1_avg == 1.0 else "No" if fact1_avg == 0.0 else f"{fact1_avg:.2f}"
            fact2_str = "Yes" if fact2_avg == 1.0 else "No" if fact2_avg == 0.0 else f"{fact2_avg:.2f}"
            
            # Add row to table
            table.append(f"| Patient {patient_num} | {fact0_str} | {fact1_str} | {fact2_str} |")
            
        return "\n".join(table)
        
    def generate_report(self):
        """Generate a markdown report of evaluation results according to specified structure"""
        report = []
        
        # Get evaluator name from arguments
        try:
            evaluator_name = self.auto_eval.evaluator_pair[0].args[0]
        except (AttributeError, IndexError):
            evaluator_name = "Unknown Evaluator"
        
        # Title
        report.append(f"# Evaluation report of the {self.auto_eval.evaluated_model} summarizer by {evaluator_name}\n")
        
        # Main results section
        report.append("## Main results\n")
        
        # A. Presence of important facts
        report.append("### A. Presence of important facts\n")
        if hasattr(self.auto_eval, 'fact_eval_res'):
            mean_score = self._calculate_mean_scores(self.auto_eval.fact_eval_res)
            report.append(f"**Score: {int(100*mean_score)}/100** (higher values better)\n")
            
            report.append("**Feedback regarding the important facts missing:**\n")
            explanation_summary = self._get_explanation_summary(fact_eval_type="fact")
            report.append(explanation_summary + "\n")
        else:
            report.append("*No fact presence evaluation data available*\n")
        
        # B. Hospital course summaries contradicting important facts
        report.append("### B. Hospital course summaries contradicting important facts\n")
        if hasattr(self.auto_eval, 'inc_fact_eval_res'):
            mean_score = self._calculate_mean_scores(self.auto_eval.inc_fact_eval_res)
            report.append(f"**Score: {int(100*mean_score)}/100** (lower values better)\n")
            
            report.append("**Feedback regarding the inconsistencies:**\n")
            explanation_summary = self._get_explanation_summary(fact_eval_type="inc_fact")
            report.append(explanation_summary + "\n")
        else:
            report.append("*No inconsistency evaluation data available*\n")
        
        # Detailed results section
        report.append("## Detailed results\n")
        
        # A. Detailed presence of important facts
        report.append("### A. Presence of important facts\n")
        if hasattr(self.auto_eval, 'fact_eval_res'):
            # Create table showing all facts per patient
            table = self._create_fact_table(fact_eval_type="fact")
            report.append(table + "\n\n")
            
            report.append("#### All Missing Facts and Explanations\n")
            relevant_explanations = self._get_all_relevant_explanations(fact_eval_type="fact")
            
            if relevant_explanations:
                for idx, (patient_num, fact_num, fact, explanation, _) in enumerate(relevant_explanations, 1):
                    report.append(f"**{idx}. Patient {patient_num}, Fact {fact_num}:**\n")
                    report.append(f"- **Fact:** {fact}\n")
                    report.append(f"- **Explanation:** {explanation}\n\n")
            else:
                report.append("*No missing facts found*\n")
        else:
            report.append("*No fact presence evaluation data available*\n")
        
        # B. Detailed hospital course summaries contradicting important facts
        report.append("### B. Hospital course summaries contradicting important facts\n")
        if hasattr(self.auto_eval, 'inc_fact_eval_res'):
            # Create table showing all facts per patient
            table = self._create_fact_table(fact_eval_type="inc_fact")
            report.append(table + "\n\n")
            
            report.append("#### All Inconsistent Facts and Explanations\n")
            relevant_explanations = self._get_all_relevant_explanations(fact_eval_type="inc_fact")
            
            if relevant_explanations:
                for idx, (patient_num, fact_num, fact, explanation, _) in enumerate(relevant_explanations, 1):
                    report.append(f"**{idx}. Patient {patient_num}, Fact {fact_num}:**\n")
                    report.append(f"- **Fact:** {fact}\n")
                    report.append(f"- **Explanation:** {explanation}\n\n")
            else:
                report.append("*No inconsistencies found*\n")
        else:
            report.append("*No inconsistency evaluation data available*\n")
        
        self.report = "\n".join(report)
        
        return self.report

if __name__ == "__main__":
    from agnostic_evaluator_models import meta_init, meta_call, lab_key
    from functools import partial
    llama_init = partial(meta_init, "llama4-maverick", lab_key)
    
    autoeval_ins = AutoEval(evaluator_pair=(llama_init, meta_call), evaluated_model="deepseek-r1")
    #autoeval_ins.facts
    #autoeval_ins.proto_summaries
    #autoeval_ins.proto_facts_merged
    autoeval_ins.fact_eval(verbose=True)
    autoeval_ins.inc_fact_eval(verbose=True)
    
    import pickle
    # Save the results to a file
    with open('autoeval_results.pkl', 'wb') as f: pickle.dump(autoeval_ins, f)
    
    # Load the results from a file
    #with open('autoeval_results.pkl', 'rb') as f: autoeval_ins = pickle.load(f)
    
    # Generate and save report
    report_generator = AutoEvalReport(autoeval_ins)
    report = report_generator.generate_report()
    
    # Save report to file
    with open("reports/evaluation_report.md", "w") as f:
        f.write(report)
    
    print("Report generated and saved as 'evaluation_report.md'")
