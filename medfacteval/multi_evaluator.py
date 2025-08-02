import numpy as np
import pandas as pd
import time
import traceback
import concurrent.futures
from auto_eval import AutoEval, API_text_to_text

class MultiEvaluatorAutoEval:
    """
    Class for running and aggregating evaluations from multiple LLM evaluators.
    Uses majority voting to combine results from different evaluators.
    """
    def __init__(self, evaluator_pairs, evaluated_model="gpt-4o"):
        """
        Initialize with multiple evaluator pairs
        
        Args:
            evaluator_pairs: List of evaluator pairs (each pair is used to initialize an AutoEval instance)
            evaluated_model: The model being evaluated
        """
        self.evaluator_pairs = evaluator_pairs
        self.evaluated_model = evaluated_model
        self.auto_eval_instances = []
        self.majority_eval = None
        
    def run_evaluations(self, verbose=False, timeout=36000): # timeout set to 10 hours
        """
        Run fact_eval and inc_fact_eval for each evaluator pair
        
        Args:
            verbose: Whether to print verbose output during evaluation
            timeout: Maximum time in seconds to wait for each evaluation step
        """
        print(f"Running evaluations of {self.evaluated_model} with {len(self.evaluator_pairs)} different evaluators...")
        for i, evaluator_pair in enumerate(self.evaluator_pairs):
            print(f"Running evaluator {i+1}/{len(self.evaluator_pairs)}...")
            try:
                # Get evaluator name for better logging
                try:
                    evaluator_name = evaluator_pair[0].args[0] if hasattr(evaluator_pair[0], 'args') else f"Evaluator {i+1}"
                except (AttributeError, IndexError):
                    evaluator_name = f"Evaluator {i+1}"
                    
                # Create AutoEval instance
                auto_eval = AutoEval(evaluator_pair, self.evaluated_model)
                
                # Run fact_eval with timeout protection using concurrent.futures
                print(f"Starting fact_eval for {evaluator_name}...")
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit the task with a timeout
                    future = executor.submit(auto_eval.fact_eval, verbose=verbose)
                    try:
                        # Wait for the result with a timeout
                        future.result(timeout=timeout)
                        print(f"Completed fact_eval for {evaluator_name} in {time.time() - start_time:.1f} seconds")
                    except concurrent.futures.TimeoutError:
                        print(f"TIMEOUT: fact_eval for {evaluator_name} exceeded {timeout} seconds")
                        # We need to continue anyway, as the API might eventually return and update the object
                        print(f"Continuing to next step but results may be incomplete")
                    except Exception as e:
                        print(f"Error during fact_eval for {evaluator_name}: {str(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
                
                # Run inc_fact_eval with timeout protection
                print(f"Starting inc_fact_eval for {evaluator_name}...")
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit the task with a timeout
                    future = executor.submit(auto_eval.inc_fact_eval, verbose=verbose)
                    try:
                        # Wait for the result with a timeout
                        future.result(timeout=timeout)
                        print(f"Completed inc_fact_eval for {evaluator_name} in {time.time() - start_time:.1f} seconds")
                    except concurrent.futures.TimeoutError:
                        print(f"TIMEOUT: inc_fact_eval for {evaluator_name} exceeded {timeout} seconds")
                        print(f"Continuing with partial results for this evaluator")
                    except Exception as e:
                        print(f"Error during inc_fact_eval for {evaluator_name}: {str(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
                
                # Check if we have any results before adding to collection
                has_fact_results = hasattr(auto_eval, 'fact_eval_res') and auto_eval.fact_eval_res
                has_inc_fact_results = hasattr(auto_eval, 'inc_fact_eval_res') and auto_eval.inc_fact_eval_res
                
                if has_fact_results or has_inc_fact_results:
                    # Add the instance to our collection if it has at least some results
                    self.auto_eval_instances.append(auto_eval)
                    print(f"Added {evaluator_name} to evaluator instances")
                else:
                    print(f"Skipping {evaluator_name} as it has no evaluation results")
                    
            except Exception as e:
                print(f"Error processing evaluator {i+1}: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                print(f"Skipping to next evaluator...")
                
        print(f"Completed evaluations with {len(self.auto_eval_instances)} successful evaluator instances")
        
        if not self.auto_eval_instances:
            raise ValueError("No successful evaluator instances. Cannot proceed with majority voting.")
            
    def _get_votes(self, eval_type="fact"):
        """
        Helper method to collect votes from all evaluators
        
        Args:
            eval_type: Type of evaluation ("fact" or "inc_fact")
            
        Returns:
            Dictionary with all votes
        """
        if not self.auto_eval_instances:
            raise ValueError("No evaluations have been run yet. Call run_evaluations() first.")
            
        # Initialize dictionaries to store all votes
        all_votes = {}
        
        # Collect votes from each evaluator
        for auto_eval in self.auto_eval_instances:
            if eval_type == "fact":
                eval_res = auto_eval.fact_eval_res
            else:
                eval_res = auto_eval.inc_fact_eval_res
                
            for patient_id in eval_res:
                if patient_id not in all_votes:
                    all_votes[patient_id] = {}
                    
                for reviewer in eval_res[patient_id]:
                    if reviewer not in all_votes[patient_id]:
                        all_votes[patient_id][reviewer] = {}
                        
                    for fact_id, score in eval_res[patient_id][reviewer].items():
                        if fact_id not in all_votes[patient_id][reviewer]:
                            all_votes[patient_id][reviewer][fact_id] = []
                            
                        all_votes[patient_id][reviewer][fact_id].append(score)
                        
        return all_votes
            
    def get_majority_vote(self, eval_type="fact"):
        """
        Calculate majority vote results across all evaluators
        
        Args:
            eval_type: Type of evaluation ("fact" or "inc_fact")
            
        Returns:
            Dictionary with majority vote results
        """
        all_votes = self._get_votes(eval_type)
        
        # Calculate majority vote
        majority_vote_results = {}
        
        for patient_id in all_votes:
            majority_vote_results[patient_id] = {}
            
            for reviewer in all_votes[patient_id]:
                majority_vote_results[patient_id][reviewer] = {}
                
                for fact_id, votes in all_votes[patient_id][reviewer].items():
                    # Count votes
                    vote_counts = {}
                    for vote in votes:
                        vote_counts[vote] = vote_counts.get(vote, 0) + 1
                    
                    # Check for ties and prioritize value 1.0 ("Yes")
                    if 1.0 in vote_counts and 0.0 in vote_counts and vote_counts[1.0] == vote_counts[0.0]:
                        # In case of a tie, choose 1.0 ("Yes")
                        majority_vote = 1.0
                    else:
                        # Otherwise, find the value with the highest count
                        majority_vote = max(vote_counts.items(), key=lambda x: x[1])[0]
                        
                    majority_vote_results[patient_id][reviewer][fact_id] = majority_vote
        
        return majority_vote_results
        
    def create_majority_auto_eval(self):
        """
        Create an AutoEval instance with majority vote results
        
        Returns:
            AutoEval instance with majority vote results
        """
        if not self.auto_eval_instances:
            raise ValueError("No evaluations have been run yet. Call run_evaluations() first.")
            
        # Use the first AutoEval instance as a template
        majority_eval = AutoEval(self.auto_eval_instances[0].evaluator_pair, self.evaluated_model)
        
        # Copy necessary attributes from the first instance
        majority_eval.proto_facts_merged = self.auto_eval_instances[0].proto_facts_merged
        majority_eval.proto_summaries = self.auto_eval_instances[0].proto_summaries
        majority_eval.facts = self.auto_eval_instances[0].facts
        
        # Set majority vote results
        majority_eval.fact_eval_res = self.get_majority_vote(eval_type="fact")
        majority_eval.inc_fact_eval_res = self.get_majority_vote(eval_type="inc_fact")
        
        # Get explanations from the first evaluator (we don't vote on explanations)
        majority_eval.fact_eval_expl = self.auto_eval_instances[0].fact_eval_expl
        majority_eval.inc_fact_eval_expl = self.auto_eval_instances[0].inc_fact_eval_expl
        
        # Calculate result averages similar to the original AutoEval.fact_eval and inc_fact_eval methods
        # Fact evaluation results average
        majority_eval.fact_eval_res_avg = {k: pd.DataFrame(majority_eval.fact_eval_res[k]).mean(axis=1).to_dict() 
                                          for k in majority_eval.fact_eval_res.keys()}
        majority_eval.fact_eval_scores = {k: np.mean(list(v.values())) 
                                         for k, v in majority_eval.fact_eval_res_avg.items()}
        
        # Inconsistency evaluation results average
        majority_eval.inc_fact_eval_res_avg = {k: pd.DataFrame(majority_eval.inc_fact_eval_res[k]).mean(axis=1).to_dict() 
                                             for k in majority_eval.inc_fact_eval_res.keys()}
        majority_eval.inc_fact_eval_scores = {k: np.mean(list(v.values())) 
                                            for k, v in majority_eval.inc_fact_eval_res_avg.items()}
        
        self.majority_eval = majority_eval
        return majority_eval
        
    def _calculate_bootstrap_ci(self, results, n_bootstrap=1000, alpha=0.05):
        """
        Calculate bootstrap confidence intervals for mean scores
        
        Args:
            results: Dictionary of evaluation results
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (0.05 for 95% CI)
            
        Returns:
            tuple: (mean_score, lower_ci, upper_ci)
        """
        # Extract all scores into a flat list
        all_scores = []
        for patient_id, patient_data in results.items():
            for reviewer, reviewer_data in patient_data.items():
                for fact_id, score in reviewer_data.items():
                    all_scores.append(score)
        
        if not all_scores:
            return float('nan'), float('nan'), float('nan')
            
        # Calculate the original mean score
        mean_score = np.mean(all_scores)
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(all_scores, size=len(all_scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate confidence interval
        lower_ci = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return mean_score, lower_ci, upper_ci
    
    def generate_report(self):
        """
        Generate a report on the multiple evaluator results
        
        Returns:
            A string containing the markdown report
        """
        if not self.auto_eval_instances:
            raise ValueError("No evaluations have been run yet. Call run_evaluations() first.")
            
        if self.majority_eval is None:
            self.create_majority_auto_eval()
        
        # Get evaluator names
        evaluator_names = []
        for i, auto_eval in enumerate(self.auto_eval_instances):
            try:
                evaluator_name = auto_eval.evaluator_pair[0].args[0]
            except (AttributeError, IndexError):
                evaluator_name = f"Evaluator {i+1}"
            evaluator_names.append(evaluator_name)
        
        # Title and subtitle with jury members
        report = []
        report.append(f"# Factual evaluation of the {self.evaluated_model} summarizer by a LLM jury\n")
        report.append(f"## Jury members: {', '.join(evaluator_names)}\n")
        
        # Main results section
        report.append("## Main results\n")
        
        # A. Presence of important facts
        report.append("### A. Presence of important facts\n")
        mean_score, lower_ci, upper_ci = self._calculate_bootstrap_ci(self.majority_eval.fact_eval_res)
        report.append(f"**Score: {int(100*mean_score)}/100** (95% CI: {int(100*lower_ci)}-{int(100*upper_ci)}) (higher values better)\n")
            
        report.append("**Feedback regarding the important facts missing:**\n")
        # Get summarized explanations from evaluators that agree with the majority vote
        fact_explanations = self._get_summarized_consensus_explanations("fact")
        report.append(fact_explanations + "\n")
        
        # B. Hospital course summaries contradicting important facts
        report.append("### B. Hospital course summaries contradicting important facts\n")
        mean_score, lower_ci, upper_ci = self._calculate_bootstrap_ci(self.majority_eval.inc_fact_eval_res)
        report.append(f"**Score: {int(100*mean_score)}/100** (95% CI: {int(100*lower_ci)}-{int(100*upper_ci)}) (lower values better)\n")
            
        report.append("**Feedback regarding the inconsistencies:**\n")
        # Get summarized explanations from evaluators that agree with the majority vote
        inc_fact_explanations = self._get_summarized_consensus_explanations("inc_fact")
        report.append(inc_fact_explanations + "\n")
        
        # Detailed results section
        report.append("## Detailed results\n")
        
        # A. Detailed presence of important facts
        report.append("### A. Presence of important facts\n")
        
        # Create restructured table showing facts as supercolumns
        fact_table = self._create_restructured_fact_table("fact", evaluator_names)
        report.append(fact_table + "\n\n")
        
        report.append("#### All Missing Facts and Explanations\n")
        missing_facts = self._get_all_explained_facts("fact")
        
        if missing_facts:
            for idx, (patient_num, fact_num, fact, explanations) in enumerate(missing_facts, 1):
                report.append(f"**{idx}. Patient {patient_num}, Fact {fact_num}:**\n")
                report.append(f"- **Fact:** {fact}\n")
                for eval_idx, explanation in explanations:
                    evaluator = evaluator_names[eval_idx] if eval_idx < len(evaluator_names) else f"Evaluator {eval_idx+1}"
                    report.append(f"- **Explanation from {evaluator}:** {explanation}\n")
                report.append("\n")
        else:
            report.append("*No missing facts found*\n")
        
        # B. Detailed hospital course summaries contradicting important facts
        report.append("### B. Hospital course summaries contradicting important facts\n")
        
        # Create restructured table showing facts as supercolumns
        inc_fact_table = self._create_restructured_fact_table("inc_fact", evaluator_names)
        report.append(inc_fact_table + "\n\n")
        
        report.append("#### All Inconsistent Facts and Explanations\n")
        inconsistent_facts = self._get_all_explained_facts("inc_fact")
        
        if inconsistent_facts:
            for idx, (patient_num, fact_num, fact, explanations) in enumerate(inconsistent_facts, 1):
                report.append(f"**{idx}. Patient {patient_num}, Fact {fact_num}:**\n")
                report.append(f"- **Fact:** {fact}\n")
                for eval_idx, explanation in explanations:
                    evaluator = evaluator_names[eval_idx] if eval_idx < len(evaluator_names) else f"Evaluator {eval_idx+1}"
                    report.append(f"- **Explanation from {evaluator}:** {explanation}\n")
                report.append("\n")
        else:
            report.append("*No inconsistencies found*\n")
        
        return "\n".join(report)
    
    def _create_restructured_fact_table(self, eval_type, evaluator_names):
        """
        Create table with facts as supercolumns and evaluators as subcolumns
        
        Args:
            eval_type: Type of evaluation ("fact" or "inc_fact")
            evaluator_names: List of evaluator names
            
        Returns:
            String containing HTML table
        """
        # Get patient IDs from the majority evaluator
        if eval_type == "fact":
            majority_res = self.majority_eval.fact_eval_res
        else:
            majority_res = self.majority_eval.inc_fact_eval_res
        
        patient_ids = sorted(majority_res.keys(), key=lambda x: int(x.split('_')[1]))
        
        # Create HTML table instead of markdown for better formatting of supercolumns
        table = []
        table.append('<table>')
        
        # First header row with fact supercolumns
        table.append('<tr>')
        table.append('<th rowspan="2">Patient</th>')
        
        # Add fact headers with colspans for each evaluator
        for fact_idx in range(3):
            table.append(f'<th colspan="{len(evaluator_names)}">Fact {fact_idx}</th>')
        
        # Add majority vote header
        table.append('<th colspan="3">Majority Vote</th>')
        table.append('</tr>')
        
        # Second header row with evaluator names
        table.append('<tr>')
        
        # Add evaluator names under each fact
        for _ in range(3):  # For each fact
            for evaluator_name in evaluator_names:
                table.append(f'<th>{evaluator_name}</th>')
        
        # Add fact columns for majority vote
        table.append('<th>F0</th><th>F1</th><th>F2</th>')
        table.append('</tr>')
        
        # Create rows for each patient
        for patient_id in patient_ids:
            patient_num = int(patient_id.split('_')[1])
            table.append('<tr>')
            table.append(f'<td>Patient {patient_num}</td>')
            
            # For each fact
            for fact_idx in range(3):
                fact_key = f'fact_{fact_idx}'
                
                # For each evaluator
                for evaluator_idx, auto_eval in enumerate(self.auto_eval_instances):
                    if eval_type == "fact":
                        eval_res = auto_eval.fact_eval_res
                    else:
                        eval_res = auto_eval.inc_fact_eval_res
                    
                    # Get all scores for this fact from this evaluator
                    cell_scores = []
                    if patient_id in eval_res:
                        for reviewer in eval_res[patient_id]:
                            if fact_key in eval_res[patient_id][reviewer]:
                                score = eval_res[patient_id][reviewer][fact_key]
                                cell_scores.append(score)
                    
                    # Calculate and format the cell value
                    if cell_scores:
                        avg_score = sum(cell_scores) / len(cell_scores)
                        cell_value = "Yes" if avg_score == 1.0 else "No" if avg_score == 0.0 else f"{avg_score:.2f}"
                    else:
                        cell_value = "N/A"
                    
                    table.append(f'<td>{cell_value}</td>')
            
            # Add majority vote for each fact
            for fact_idx in range(3):
                fact_key = f'fact_{fact_idx}'
                
                # Collect all majority vote scores for this fact
                maj_scores = []
                for reviewer in majority_res.get(patient_id, {}):
                    if fact_key in majority_res[patient_id][reviewer]:
                        score = majority_res[patient_id][reviewer][fact_key]
                        maj_scores.append(score)
                
                # Calculate and format the majority vote
                if maj_scores:
                    avg_score = sum(maj_scores) / len(maj_scores)
                    maj_value = "Yes" if avg_score == 1.0 else "No" if avg_score == 0.0 else f"{avg_score:.2f}"
                else:
                    maj_value = "N/A"
                
                table.append(f'<td>{maj_value}</td>')
            
            table.append('</tr>')
        
        table.append('</table>')
        
        return "\n".join(table)
    
    def _get_all_explained_facts(self, eval_type):
        """
        Get all facts that any evaluator marked as missing/inconsistent with explanations
        
        Args:
            eval_type: Type of evaluation ("fact" or "inc_fact")
            
        Returns:
            List of tuples (patient_num, fact_num, fact, explanations)
            where explanations is a list of tuples (evaluator_idx, explanation)
        """
        all_explained_facts = {}  # {(patient_num, fact_num): (fact, [(evaluator_idx, explanation)])}
        
        for evaluator_idx, auto_eval in enumerate(self.auto_eval_instances):
            if eval_type == "fact":
                eval_res = auto_eval.fact_eval_res
                eval_expl = auto_eval.fact_eval_expl
                target_score = 0  # For fact eval, we look for missing facts (score 0)
            else:
                eval_res = auto_eval.inc_fact_eval_res
                eval_expl = auto_eval.inc_fact_eval_expl
                target_score = 1  # For inconsistency eval, we look for inconsistencies (score 1)
            
            for patient_id in eval_res:
                patient_num = int(patient_id.split('_')[1])
                
                for reviewer in eval_res[patient_id]:
                    for fact_key, score in eval_res[patient_id][reviewer].items():
                        fact_num = int(fact_key.split('_')[1])
                        
                        if score == target_score:
                            # Get the fact text
                            if hasattr(auto_eval, 'proto_facts_merged'):
                                patient_index = auto_eval.proto_facts_merged.index[auto_eval.proto_facts_merged['patient_i'] == patient_num].tolist()
                                if patient_index:
                                    fact = auto_eval.proto_facts_merged.iloc[patient_index[0], 4+fact_num]
                                else:
                                    fact = f"Fact {fact_num} from Patient {patient_num}"
                            else:
                                fact = f"Fact {fact_num} from Patient {patient_num}"
                            
                            # Get explanation and clean it for raw string presentation
                            explanation = eval_expl[patient_id][reviewer].get(fact_key, "No explanation provided")
                            
                            # Clean explanation - remove line breaks and normalize spaces
                            explanation = ' '.join(explanation.replace('\n', ' ').split())
                            
                            # Add to collected data
                            key = (patient_num, fact_num)
                            if key not in all_explained_facts:
                                all_explained_facts[key] = (fact, [])
                            all_explained_facts[key][1].append((evaluator_idx, explanation))
        
        # Convert to sorted list
        result = []
        for (patient_num, fact_num), (fact, explanations) in sorted(all_explained_facts.items()):
            result.append((patient_num, fact_num, fact, explanations))
        
        return result
    
    def _get_summarized_consensus_explanations(self, eval_type):
        """
        Get summarized explanations from evaluators that agree with the majority vote
        
        Args:
            eval_type: Type of evaluation ("fact" or "inc_fact")
            
        Returns:
            Summary of consensus explanations
        """
        # First, find facts where the majority vote indicates an issue
        if eval_type == "fact":
            majority_res = self.majority_eval.fact_eval_res
            target_score = 0  # Missing facts (score 0)
        else:
            majority_res = self.majority_eval.inc_fact_eval_res
            target_score = 1  # Inconsistencies (score 1)
        
        # Find facts with issues according to majority vote
        issue_facts = []  # [(patient_num, fact_num, fact)]
        for patient_id in majority_res:
            patient_num = int(patient_id.split('_')[1])
            
            for reviewer in majority_res[patient_id]:
                for fact_key, score in majority_res[patient_id][reviewer].items():
                    fact_num = int(fact_key.split('_')[1])
                    
                    if score == target_score:
                        # Get the fact text
                        if hasattr(self.majority_eval, 'proto_facts_merged'):
                            patient_index = self.majority_eval.proto_facts_merged.index[self.majority_eval.proto_facts_merged['patient_i'] == patient_num].tolist()
                            if patient_index:
                                fact = self.majority_eval.proto_facts_merged.iloc[patient_index[0], 4+fact_num]
                            else:
                                fact = f"Fact {fact_num} from Patient {patient_num}"
                        else:
                            fact = f"Fact {fact_num} from Patient {patient_num}"
                        
                        issue_facts.append((patient_id, fact_key, fact))
        
        if not issue_facts:
            return "No explanations to summarize."
        
        # Collect explanations from evaluators that agree with the majority vote
        relevant_explanations = []
        for patient_id, fact_key, fact in issue_facts:
            for evaluator_idx, auto_eval in enumerate(self.auto_eval_instances):
                if eval_type == "fact":
                    eval_res = auto_eval.fact_eval_res
                    eval_expl = auto_eval.fact_eval_expl
                else:
                    eval_res = auto_eval.inc_fact_eval_res
                    eval_expl = auto_eval.inc_fact_eval_expl
                
                if patient_id in eval_res:
                    for reviewer in eval_res[patient_id]:
                        if fact_key in eval_res[patient_id][reviewer]:
                            score = eval_res[patient_id][reviewer][fact_key]
                            
                            # If this evaluator agrees with majority vote
                            if score == target_score:
                                explanation = eval_expl[patient_id][reviewer].get(fact_key, "No explanation provided")
                                patient_num = int(patient_id.split('_')[1])
                                fact_num = int(fact_key.split('_')[1])
                                
                                relevant_explanations.append(f'- Patient {patient_num}, fact {fact_num} ("{fact}")\nExplanation: {explanation}.\n')
        
        if not relevant_explanations:
            return "No consensus explanations available."
        
        # Create LLM instance for summarization using the last evaluator
        llm_instance = API_text_to_text(*self.auto_eval_instances[-1].evaluator_pair)
        
        if eval_type == "fact":    
            # Create prompt for the LLM to summarize explanations for missing facts
            prompt = f"""You are an expert medical AI. Below are detailed explanations for why hospital course summaries are missing important clinical facts. Summarize these explanations concisely (one paragraph) by highlighting common failure patterns and key insights into why these omissions occur. Include illustrative examples referencing specific patient numbers and missing facts to clarify your points.
            
            --- Explanations ---
            {chr(10).join(relevant_explanations)}

            --- Guidelines for Response ---
            1. Clearly identify common themes and patterns causing clinical omissions.
            2. Keep the summary concise and precise, ideally limited to one structured paragraph.
            3. Always reference specific patient numbers and facts as provided.
            4. Focus exclusively on summarizing patterns and explanations provided, without suggesting solutions or recommendations."""

        else:
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
        try:
            summary = llm_instance.gen_txt_to_txt(prompt)
            return summary
        except Exception as e:
            return f"Error summarizing explanations: {str(e)}\n\n" + "\n".join(relevant_explanations[:5]) + "\n\n(Showing only first 5 explanations due to length)"
    
    def _calculate_mean_scores(self, results):
        """Calculate mean scores across all patients"""
        all_scores = []
        for patient_id, patient_data in results.items():
            for reviewer, reviewer_data in patient_data.items():
                for fact_id, score in reviewer_data.items():
                    all_scores.append(score)
        return np.mean(all_scores) if all_scores else float('nan')

if __name__ == "__main__":
    
    # Example usage: from terminal, run:
    # python multi_evaluator.py --evaluated_model gemini-2.5-pro-preview-05-06 --evaluators claude-3.5 llama4-maverick deepseek-r1
    # this will run the evaluation for the gemini-2.5-pro-preview-05-06 model with the claude-3.5, llama4-maverick, and deepseek-r1 evaluators
    
    from agnostic_evaluator_models import meta_init, meta_call, lab_key, anthropic_init, anthropic_call, gemini_init, gemini_call, microsoft_init, microsoft_call, openai_init, openai_call, deepseek_init, deepseek_call
    from functools import partial
    import pickle
    import argparse

    # Define available evaluators
    credentials_path = "../../mykeys/grolleau_application_default_credentials.json"
    
    available_evaluators = {
        "gpt-4.1-nano": (partial(openai_init, "gpt-4.1-nano", lab_key), openai_call),
        "gpt-4.1": (partial(openai_init, "gpt-4.1", lab_key), openai_call),
        "claude-3.5": (partial(anthropic_init, "claude-3.5-sonnet-v2", lab_key), anthropic_call),
        "claude-3.7": (partial(anthropic_init, "claude-3.7-sonnet", lab_key), anthropic_call),
        "llama4-scout": (partial(meta_init, "llama4-scout", lab_key), meta_call),
        "llama4-maverick": (partial(meta_init, "llama4-maverick", lab_key), meta_call),
        "phi-3.5-mini": (partial(microsoft_init, "phi-3.5-mini-instruct", lab_key), microsoft_call),
        "deepseek-r1": (partial(deepseek_init, "deepseek-r1", lab_key, view_thinking=False), deepseek_call),
        "gemini-2.0-flash": (partial(gemini_init, "gemini-2.0-flash-exp", credentials_path), gemini_call),
        "gemini-2.5-pro": (partial(gemini_init, "gemini-2.5-pro-preview-03-25", credentials_path), gemini_call),
    }

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run multi-evaluator analysis for a given model.")
    parser.add_argument("--evaluated_model", type=str, default="gemini-2.5-pro-preview-05-06",
                        help="The model to evaluate.")
    parser.add_argument("--evaluators", nargs='+',
                        choices=list(available_evaluators.keys()),
                        default=["claude-3.5", "llama4-maverick", "deepseek-r1"],
                        help=f"A list of evaluators to use for the jury. Available: {', '.join(available_evaluators.keys())}")
    
    args = parser.parse_args()

    # Get evaluated model and evaluator pairs from command line arguments
    evaluated_model = args.evaluated_model
    evaluator_pairs = [available_evaluators[name] for name in args.evaluators]

    # Run multi-evaluator analysis
    multi_eval = MultiEvaluatorAutoEval(evaluator_pairs, evaluated_model=evaluated_model)
    multi_eval.run_evaluations(verbose=True)
    
    # Get majority vote results
    majority_eval = multi_eval.create_majority_auto_eval()
    
    # Save as as pickle file
    with open(f"multi_eval_pickle/prompt_vs_workflow/multi_eval_{evaluated_model}.pkl", "wb") as f:
        pickle.dump(multi_eval, f)
    
    # Generate report
    report = multi_eval.generate_report()
    
    # Save report to file
    with open(f"reports/prompt_vs_workflow/multi_evaluator_report_{evaluated_model}.md", "w") as f:
        f.write(report)