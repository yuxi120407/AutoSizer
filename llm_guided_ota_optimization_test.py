#!/usr/bin/env python3
"""
LLM-Guided OTA Optimization with Advanced Search Methods (Google Gemini)
=========================================================================
Uses Google's Gemini API to intelligently decide:
- Advanced search strategy (lhs, genetic, bayesian, adaptive, annealing, multistart)
- Number of samples per iteration
- When to stop searching and run PEX
- When overall optimization has converged
"""

import google.generativeai as genai
import os
import re
import shutil
import yaml
import json
import time
import pprint
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from prompts import (
    header_section,
    objective_selection_prompt,
    methods_section,
    decision_framework_section,
    parameter_tuning_section,
    response_format_section,
)
from problem_agent_test import CIRCUIT_UNDERSTANDING_PROMPT, SEARCH_SPACE_REDUCTION_PROMPT, CIRCUIT_REUNDERSTANDING_PROMPT, TRANSISTOR_GROUPING_PROMPT
from utils.feedback_extraction import (extract_optimization_feedback, save_feedback_json, 
                                      prepare_format_params, extract_useful_section, check_user_specs_met)

NAME_ALIASES = {
    "dc_gain_db": "DC Gain",
    "fom": "Figure of Merit",
}


def parse_llm_json_response(llm_response):
    """Parse JSON from LLM response with aggressive cleaning"""

    text = llm_response.text.strip()
    text = re.sub(r'```(?:json)?\s*', '', text).strip()

    # Extract JSON
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        print(f"\n{'='*80}")
        print(f"ERROR: No JSON object found in LLM response")
        print(f"{'='*80}")
        print(f"Response text (first 500 chars):")
        print(text[:500])
        print(f"\n{'='*80}")
        print(f"Full response length: {len(text)} characters")
        print(f"{'='*80}\n")
        raise ValueError("No JSON object found")
    text = text[start:end+1]
    
    # AGGRESSIVE FIXES:
    # 1. Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # 2. Remove comments
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # 3. Fix common issues with rationale/explanation fields
    # Replace newlines in string values with spaces
    text = re.sub(r'"\s*\n\s*', '" ', text)
    
    # 4. Try to fix unescaped quotes in strings (naive approach)
    # This is tricky - skip for now
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Enhanced error output
        print(f"\nJSON Error: {e.msg} at position {e.pos} (line {e.lineno}, col {e.colno})")
        
        # Show context
        start_ctx = max(0, e.pos - 150)
        end_ctx = min(len(text), e.pos + 150)
        print(f"\nContext around error:")
        print(text[start_ctx:e.pos] + " <<<ERROR>>> " + text[e.pos:end_ctx])
        
        print(f"\nFirst 1000 chars of JSON:")
        print(text[:1000])
        
        # NEW: Handle incomplete string values by finding last complete field
        print("\nAttempting recovery from incomplete string...")
        
        # Find the last complete key-value pair before the error
        # Look backwards for pattern: "key": "value",
        truncate_pos = e.pos
        
        # Strategy 1: Find last comma before error
        last_comma = text.rfind(',', 0, e.pos)
        if last_comma > 0:
            # Try truncating at last comma and closing the object
            truncated = text[:last_comma].strip() + '\n}'
            try:
                print(f"Attempting parse after truncating at position {last_comma}...")
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Find last complete closing quote before error
        # and remove the incomplete field
        last_quote = text.rfind('"', 0, e.pos - 1)
        if last_quote > 0:
            # Check if there's a comma after this quote
            next_comma = text.find(',', last_quote)
            if next_comma > 0 and next_comma < e.pos:
                truncated = text[:next_comma].strip() + '\n}'
                try:
                    print(f"Attempting parse after removing incomplete field...")
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Original truncation at last brace
        print("\nAttempting truncation at last closing brace...")
        last_brace = text.rfind('}', 0, e.pos)
        if last_brace > 0:
            truncated = text[:last_brace + 1]
            try:
                print("Successfully parsed truncated JSON")
                return json.loads(truncated)
            except json.JSONDecodeError:
                print("Truncation recovery failed")
        
        raise

def extract_optimization_config(reduction_data):
    """
    Extract the optimization configuration in a format ready for Optuna.
    
    Args:
        reduction_data: Parsed output from search space reduction LLM
        
    Returns:
        dict: {
            'variables_to_optimize': {var_name: [list of values]},
            'variables_fixed': {var_name: fixed_value},
            'variable_info': {var_name: full info dict}
        }
    """
    
    # Extract variables to optimize
    variables_to_optimize = {}
    for var_name, var_info in reduction_data['optimization_configuration']['variables_to_optimize'].items():
        variables_to_optimize[var_name] = var_info['search_space']
    
    # Extract fixed variables (handles both compact and verbose formats)
    variables_fixed = {}
    for var_name, var_info in reduction_data['optimization_configuration']['variables_fixed'].items():
        if isinstance(var_info, dict):
            variables_fixed[var_name] = var_info['fixed_value']
        else:
            variables_fixed[var_name] = var_info
    
    # Keep full info for reference
    variable_info = {
        'to_optimize': reduction_data['optimization_configuration']['variables_to_optimize'],
        'fixed': reduction_data['optimization_configuration']['variables_fixed']
    }
    
    num_combinations = 1
    for var_name, search_space in variables_to_optimize.items():
        num_combinations *= len(search_space)
            
    return {
        'variables_to_optimize': variables_to_optimize,
        'variables_fixed': variables_fixed,
        'variable_info': variable_info,
        'search_space_summary': reduction_data['search_space_summary'],
        'num_combinations': num_combinations
    }

class LLMOptimizationAgent:
    """LLM agent that guides the optimization process using Google Gemini with advanced methods"""
    
    def __init__(self, config, model: str = "gemini-2.5-flash", user_specs: str = None, num_variables_to_optimize: int = None):
        """
        Initialize LLM agent with support for multiple models

        Parameters:
        -----------
        model: str
            Model to use:
            - gemini-2.5-flash (default)
            - claude-sonnet-4-5
            - gpt-4o
            - Llama-3.3-70B-Instruct
        """

        self.config = config
        self.model_name = model

        self.base_metric = self.config["base_metrics"]
        self.metrics = self.config["metrics"]

        # Get API key from .env file or YAML config (fallback)
        if model.startswith("gemini"):
            api_key = os.getenv("GOOGLE_API_KEY") or self.config.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env or config YAML")
            genai.configure(api_key=api_key)
        elif model.startswith("claude"):
            from openai import OpenAI
            api_key = os.getenv("ANTHROPIC_API_KEY") or self.config.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in .env or config YAML")
            self.client = OpenAI(
                base_url="https://api.anthropic.com/v1",
                api_key=api_key
            )
        elif model.startswith("gpt"):
            from openai import AzureOpenAI
            api_key = os.getenv("AZURE_OPENAI_KEY") or self.config.get("AZURE_OPENAI_KEY")
            endpoint = os.getenv("AZURE_GPT4O_ENDPOINT") or self.config.get("AZURE_GPT4O_ENDPOINT", "https://claude-useast2.cognitiveservices.azure.com/")
            api_version = os.getenv("AZURE_GPT4O_API_VERSION") or self.config.get("AZURE_GPT4O_API_VERSION", "2024-12-01-preview")
            if not api_key:
                raise ValueError("AZURE_OPENAI_KEY not found in .env or config YAML")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
        elif model == "Llama-3.3-70B-Instruct":
            from openai import OpenAI
            api_key = os.getenv("AZURE_OPENAI_KEY") or self.config.get("AZURE_OPENAI_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or self.config.get("AZURE_OPENAI_ENDPOINT", "https://claude-useast2.services.ai.azure.com/openai/v1/")
            if not api_key:
                raise ValueError("AZURE_OPENAI_KEY not found in .env or config YAML")
            self.client = OpenAI(
                base_url=endpoint,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Initialize model with configuration (Gemini-specific)
        if model.startswith("gemini"):
            generation_config = {
                "temperature": 0.4,  # Slightly higher for more creative strategies 0.1
                "top_p": 0.85,
                "top_k": 20,
                "max_output_tokens": 8192, #16384
            }

            self.model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
            )
        else:
            # For OpenAI-compatible models, store config separately
            self.generation_config = {
                "temperature": 0.4,
                "max_tokens": 8192,
            }
            self.model = None  # OpenAI-compatible models don't use GenerativeModel



        self.user_specs = user_specs
        self.num_variables_to_optimize = num_variables_to_optimize
        self.var_names = list(self.config['variable'].keys())


        # decision = self._llm_select_objective(user_specs)
        # self.target_metric = self._convert_decision_to_target_metric(decision)
        self.target_metric = self._get_fom_metric()
        self.conversation_history = []

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        self.token_usage_log = []  # Detailed log of each API call

    def _generate_content(self, prompt: str, call_description: str = ""):
        """
        Universal method to generate content using Gemini or OpenAI-compatible APIs
        with token usage tracking

        Parameters:
        -----------
        prompt: str
            The prompt to send to the LLM
        call_description: str
            Description of this API call for logging

        Returns:
        --------
        response object with .text attribute
        """
        if self.model_name.startswith("gemini"):
            # Use Gemini API (self.model is the GenerativeModel object)
            response = self.model.generate_content(prompt)

            # Extract token usage from Gemini response
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total = response.usage_metadata.total_token_count
            else:
                # Fallback if usage_metadata not available
                input_tokens = 0
                output_tokens = 0
                total = 0

            # Track usage
            self._log_token_usage(input_tokens, output_tokens, total, call_description)

            return response
        else:
            # Use OpenAI-compatible API (self.client is the OpenAI client)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.generation_config["temperature"],
                max_tokens=self.generation_config["max_tokens"]
            )

            # Extract token usage from OpenAI response
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total = response.usage.total_tokens

            # Track usage
            self._log_token_usage(input_tokens, output_tokens, total, call_description)

            # Create a response object that mimics Gemini's structure
            class ResponseWrapper:
                def __init__(self, text, usage_metadata=None):
                    self.text = text
                    self.usage_metadata = usage_metadata

            # Create usage metadata object for compatibility
            class UsageMetadata:
                def __init__(self, prompt_tokens, completion_tokens, total_tokens):
                    self.prompt_token_count = prompt_tokens
                    self.candidates_token_count = completion_tokens
                    self.total_token_count = total_tokens

            return ResponseWrapper(
                response.choices[0].message.content,
                UsageMetadata(input_tokens, output_tokens, total)
            )

    def _log_token_usage(self, input_tokens: int, output_tokens: int, total_tokens: int, description: str = ""):
        """Log token usage for this API call"""
        self.api_call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        # Log this call
        usage_entry = {
            'call_number': self.api_call_count,
            'description': description,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'cumulative_input': self.total_input_tokens,
            'cumulative_output': self.total_output_tokens,
            'cumulative_total': self.total_tokens
        }
        self.token_usage_log.append(usage_entry)

        print(f"  [API Call #{self.api_call_count}] {description}")
        print(f"    Tokens - Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {total_tokens:,}")
        print(f"    Cumulative - Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,}, Total: {self.total_tokens:,}")

    def get_token_usage_summary(self):
        """Get current token usage summary"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_tokens,
            'api_call_count': self.api_call_count,
            'model': self.model_name
        }

    def reset_token_usage(self):
        """Reset token usage counters (useful for tracking per-cycle usage)"""
        previous_usage = self.get_token_usage_summary()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        return previous_usage


    # def _get_fom_metric(self):
    #     """
    #     Get FOM metric configuration from YAML
        
    #     Returns a target_metric dict that matches the expected format
    #     """
    #     metric_post = self.config.get('metric_post', {})
    #     fom_config = metric_post.get('fom', {})
        
    #     return {
    #         'metric_name': 'fom',
    #         'metric_key': 'fom',
    #         'is_composite': False,  # FOM is already computed, not composite
    #         'direction': 'maximize',  # Always maximize FOM
    #         'metric_format': f".{fom_config.get('decimals', 4)}f",
    #         'metric_unit': fom_config.get('unit', ''),
    #         'expression': fom_config.get('expr', ''),  # Keep for reference
    #     }
    def _get_fom_metric(self):
        """
        Get FOM metric configuration from YAML, with auto-generation if needed
        
        Priority:
        1. Try to auto-generate from user_specs_metric
        2. If auto-generation fails, fall back to YAML expr
        3. If both fail, use "1.0"
        
        Returns a target_metric dict that matches the expected format
        """
        metric_post = self.config.get('metric_post', {})
        fom_config = metric_post.get('fom', {})
        
        # Get original FOM expression from YAML (as fallback)
        yaml_expr = fom_config.get('expr', '').strip()
        
        # ALWAYS try to auto-generate first
        user_specs_metric = self.config['user_specs_metric']
        fom_expr = None
        
        if user_specs_metric:
            print(f"\n{'='*80}")
            print(f"FOM EXPRESSION AUTO-GENERATION")
            print(f"{'='*80}")
            print(f"Attempting to auto-generate FOM from user_specs_metric...")
            print(f"User specs: {user_specs_metric}")
            
            fom_expr = self._auto_generate_fom_expression(user_specs_metric)
            
            # Check if auto-generation succeeded
            if fom_expr and fom_expr != "FAILED":
                print(f"✅ Generated FOM expression: {fom_expr}")
                print(f"{'='*80}\n")
                
                # Update the config
                if 'metric_post' not in self.config:
                    self.config['metric_post'] = {}
                if 'fom' not in self.config['metric_post']:
                    self.config['metric_post']['fom'] = {}
                self.config['metric_post']['fom']['expr'] = fom_expr
            else:
                # Auto-generation failed, fall back to YAML
                print(f"❌ Auto-generation failed")
                if yaml_expr:
                    fom_expr = yaml_expr
                    print(f"⚠️  Falling back to YAML expression: {fom_expr}")
                else:
                    fom_expr = "1.0"
                    print(f"⚠️  No YAML expression found, using default: 1.0")
                print(f"{'='*80}\n")
        else:
            # No user_specs_metric, use YAML expression
            if yaml_expr:
                fom_expr = yaml_expr
                print(f"[INFO] No user_specs_metric, using FOM expression from YAML: {fom_expr}")
            else:
                fom_expr = "1.0"
                print(f"[WARNING] No user_specs_metric and no YAML expression, using default FOM = 1.0")
        
        return {
            'metric_name': 'fom',
            'metric_key': 'fom',
            'is_composite': False,
            'direction': 'maximize',
            'metric_format': f".{fom_config.get('decimals', 4)}f",
            'metric_unit': fom_config.get('unit', ''),
            'expression': fom_expr,
            'degradation_key': 'fom_percent'
        }


        

    def _auto_generate_fom_expression(self, user_specs_metric: str) -> str:
        """Auto-generate FOM expression from user_specs_metric"""
        import re
        
        if not user_specs_metric:
            return "FAILED"
        
        pattern = r'(\w+)\s*([<>])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        matches = re.findall(pattern, user_specs_metric)
        
        if not matches:
            return "FAILED"
        
        numerator_terms = []
        denominator_terms = []
        
        for metric_name, operator, threshold in matches:
            if metric_name.lower() == 'fom':
                continue
            
            try:
                threshold_float = float(threshold)
                if abs(threshold_float) < 1e-10:
                    continue
                
                if operator == '>':
                    if threshold_float < 0:
                        # e.g., passband_gain_db > -3 means "closer to 0 is better"
                        # Smaller abs value is better → put abs() in denominator
                        denominator_terms.append(f"(abs({metric_name})/{abs(threshold_float)})")
                    else:
                        # Maximize: metric should be > threshold
                        numerator_terms.append(f"({metric_name}/{abs(threshold_float)})")
                elif operator == '<':
                    # Minimize: metric should be < threshold
                    if threshold_float < 0:
                        # Negative target (e.g., thd_db < -60)
                        # More negative is better → put in numerator
                        # abs(thd_db)/abs(target): -80/60 = 1.33 > -40/60 = 0.67
                        numerator_terms.append(f"(abs({metric_name})/{abs(threshold_float)})")
                    else:
                        # Positive target (e.g., power < 100)
                        # Lower is better → put in denominator
                        denominator_terms.append(f"({metric_name}/{threshold_float})")
            except:
                continue
        
        if not numerator_terms and not denominator_terms:
            return "FAILED"
        
        numerator = " * ".join(numerator_terms) if numerator_terms else "1.0"
        
        if denominator_terms:
            denominator = " * ".join(denominator_terms)
            expression = f"({numerator}) / ({denominator})"
        else:
            expression = numerator
        
        return expression

    def circuits_understanding(self):
        METRICS = self.config['metrics']

        def prettify_metric_name(name: str) -> str:
            return name.replace("_", " ").title()

        METRIC_DESCRIPTIONS = {}
        
        for metric, spec in self.config['metric_post'].items():
            # Base name
            label = NAME_ALIASES.get(metric, prettify_metric_name(metric))
        
            # Add unit if present
            unit = spec.get("unit", "")
            if unit:
                label = f"{label} ({unit})"
        
            # Add expression if present
            if "expr" in spec:
                label = f"{label} = {spec['expr']}"
        
            METRIC_DESCRIPTIONS[metric] = label
    
        
        # Build the metrics list string
        METRICS_LIST = '\n'.join(f'- {METRIC_DESCRIPTIONS.get(m, m)} ({m})' for m in METRICS)
        
        # Build the impact section
        IMPACT_SECTIONS = []
        for metric_key in METRICS:
            metric_name = METRIC_DESCRIPTIONS.get(metric_key, metric_key)
            IMPACT_SECTIONS.append(
                f'**Impact on {metric_name}:**\n'
                f'Explain how the optimization variables {{variables}} affect {metric_key}. '
                f'Focus on which variables have the strongest impact on {metric_key} and why.'
            )
        IMPACT_SECTION_TEXT = "\n\n".join(IMPACT_SECTIONS)
        
        # Build JSON structure for impact
        IMPACT_JSON_FIELDS = {}
        for metric_key in METRICS:
            metric_name = METRIC_DESCRIPTIONS.get(metric_key, metric_key)
            IMPACT_JSON_FIELDS[metric_key] = f"3-5 sentences explaining how optimization variables affect {metric_name}, focusing on which variables have strongest impact and why."
        
        IMPACT_JSON_STR = json.dumps(IMPACT_JSON_FIELDS, indent=4)[1:-1]

        circuit_understanding_prompt = CIRCUIT_UNDERSTANDING_PROMPT.format(
                subckt_name=self.config['subckt_name'],
                ota_subckt_template=self.config['ota_subckt_template'],
                testbench_template=self.config['testbench_template'],
                params=str(self.config['params']),
                variables=str(self.config['variable']),
                metrics_list=METRICS_LIST,
                impact_section_text=IMPACT_SECTION_TEXT,
                impact_json_str=IMPACT_JSON_STR
            )

        circuit_response = self._generate_content(circuit_understanding_prompt, "Circuit Understanding Analysis")

        # Check if response seems truncated
        response_text = circuit_response.text.strip()
        if len(response_text) < 50:
            print(f"\n⚠️  WARNING: LLM response is very short ({len(response_text)} chars)")
            print(f"Response: {response_text}")
            print(f"This might indicate an API error or token limit issue\n")

        try:
            circuit_data = parse_llm_json_response(circuit_response)
            return circuit_data
        except ValueError as e:
            print(f"\n❌ Failed to parse circuit understanding response")
            print(f"Error: {e}")
            print(f"Returning None - optimization will proceed without circuit understanding\n")
            return None


    def transistor_grouping(self):
        """
        For high-dimensional circuits (>20 variables), ask the LLM to group
        transistors by functional role. Returns a grouping dict and modifies
        self.config to use group variables instead of individual ones.
        """
        var_names = list(self.config['variable'].keys())
        n_vars = len(var_names)

        # Build value ranges text
        range_lines = []
        if 'W_values' in self.config:
            range_lines.append(f"W_values: {self.config['W_values']}")
        if 'L_values' in self.config:
            range_lines.append(f"L_values: {self.config['L_values']}")
        if 'm_values' in self.config:
            range_lines.append(f"m_values: {self.config['m_values']}")
        # Add any specific value lists
        for key in sorted(self.config.keys()):
            if key.endswith('_values') and key not in ('W_values', 'L_values', 'm_values'):
                range_lines.append(f"{key}: {self.config[key]}")

        prompt = TRANSISTOR_GROUPING_PROMPT.format(
            n_vars=n_vars,
            subckt_name=self.config.get('subckt_name', 'UNKNOWN'),
            ota_subckt_template=self.config.get('ota_subckt_template', ''),
            all_variables=', '.join(var_names),
            value_ranges='\n'.join(range_lines)
        )

        response = self._generate_content(prompt, "Transistor Grouping (High-Dim)")
        grouping_data = parse_llm_json_response(response)

        # Build new variable section and group_mapping from LLM output
        new_variables = {}
        group_mapping = {}

        for group_name, group_info in grouping_data['transistor_groups'].items():
            transistors = group_info['transistors']

            # Create W, L, m group variables
            for param_type, range_key in [('W', 'W_range'), ('L', 'L_range'), ('m', 'm_range')]:
                group_var = f"{param_type}_{group_name}"
                new_variables[group_var] = None  # optimization variable

                # Map group var → individual vars
                individual_vars = [f"{param_type}_{t}" for t in transistors]
                group_mapping[group_var] = individual_vars

                # Store the value range for this group variable
                if range_key in group_info and group_info[range_key]:
                    self.config[f"{group_var}_values"] = group_info[range_key]

        # Add individual (non-transistor) variables
        for ind_var in grouping_data.get('individual_variables', []):
            if ind_var in self.config.get('variable', {}):
                new_variables[ind_var] = None

        # Replace config
        original_variables = dict(self.config['variable'])
        self.config['variable'] = new_variables
        self.config['group_mapping'] = group_mapping
        self.config['original_variables'] = original_variables

        # Update var_names
        self.var_names = list(new_variables.keys())

        print(f"\n  Grouping complete!")
        print(f"  {n_vars} individual variables → {len(new_variables)} group + individual variables")
        print(f"  Groups: {len(grouping_data['transistor_groups'])}")
        for gname, ginfo in grouping_data['transistor_groups'].items():
            print(f"    {gname}: {len(ginfo['transistors'])} transistors — {ginfo.get('role', '')}")
        print(f"  Individual: {grouping_data.get('individual_variables', [])}")

        return grouping_data

    def search_space_generating(self, circuit_data):

        base_width_values = self.config['W_values']
        base_width_values_str = str(base_width_values)
        
        
        width_scales = self.config['width_scales']
        
        # Calculate total number of variables
        total_num_variables = len(width_scales)
        
        # Format width scales as string
        width_scales_lines = []
        for base_var, (actual_var, scale) in width_scales.items():
            width_scales_lines.append(f"  {actual_var} = {base_var} × {scale}")
        width_scales_str = '\n'.join(width_scales_lines)
        
        # Extract variable impact summary from circuit_data
        variable_impact_lines = []
        for metric, impact_text in circuit_data['optimization_variables_impact'].items():
            variable_impact_lines.append(f"**{metric}:**\n{impact_text}")
        variable_impact_summary = '\n\n'.join(variable_impact_lines)
        
        # Extract variable interactions
        variable_interactions = circuit_data['variable_interactions']
        
        # Extract key insights
        key_insights_lines = []
        for i, insight in enumerate(circuit_data['key_insights_for_optimization'], 1):
            key_insights_lines.append(f"{i}. {insight}")
        key_insights = '\n'.join(key_insights_lines)

        search_space_reduction_prompt = SEARCH_SPACE_REDUCTION_PROMPT.format(
            subckt_name=self.config['subckt_name'],
            target_metric=self.target_metric,
            num_variables_to_optimize=self.num_variables_to_optimize,
            total_num_variables=total_num_variables,
            base_width_values=base_width_values_str,
            width_scales=width_scales_str,
            variable_impact_summary=variable_impact_summary,
            variable_interactions=variable_interactions,
            key_insights=key_insights
        )

        search_space_response = self._generate_content(search_space_reduction_prompt, "Initial Search Space Reduction")

        # Parse JSON
        reduction_data = parse_llm_json_response(search_space_response)
        opt_config = extract_optimization_config(reduction_data)

        return opt_config

    def search_space_generating_new(self, circuit_data):
        import json
        
        # Get variables
        variables = self.config.get('variable', {})
        var_names = list(variables.keys())
        
        # Check for scaling
        width_scales = self.config.get('width_scales', {})
        length_scales = self.config.get('length_scales', {})
        uses_scales = bool(width_scales or length_scales)
        
        # Calculate total number of variables
        total_num_variables = len(var_names)
        
        # Build variable ranges section
        variable_ranges_lines = []
        
        if uses_scales:
            # VCO/SC pattern: Show base variables with generic ranges
            w_values = self.config.get('W_values', [])
            l_values = self.config.get('L_values', [])

            if w_values:
                variable_ranges_lines.append(f"- Base widths: {w_values} µm")
            if l_values:
                variable_ranges_lines.append(f"- Base lengths: {l_values} µm")

            # Also show any remaining variables that aren't covered by width/length scales
            scaled_base_vars = set()
            for scale_pair in width_scales.values():
                if isinstance(scale_pair, (list, tuple)) and len(scale_pair) == 2:
                    scaled_base_vars.add(scale_pair[0])
            for scale_pair in length_scales.values():
                if isinstance(scale_pair, (list, tuple)) and len(scale_pair) == 2:
                    scaled_base_vars.add(scale_pair[0])

            for var_name in var_names:
                if var_name not in scaled_base_vars:
                    specific_key = f"{var_name}_values"
                    if specific_key in self.config:
                        values = self.config[specific_key]
                        unit = "Ω" if var_name.startswith('R') else ("F" if var_name.startswith('C') else "")
                        variable_ranges_lines.append(f"- {var_name}: {values} {unit}".strip())
        else:
            # Bandgap pattern: Show variable-specific ranges
            for var_name in var_names:
                specific_key = f"{var_name}_values"
                
                if specific_key in self.config:
                    values = self.config[specific_key]
                    unit = "µm" if var_name.startswith(('W_', 'L_')) else ""
                    variable_ranges_lines.append(f"- {var_name}: {values} {unit}".strip())
                elif var_name.startswith('W_') and 'W_values' in self.config:
                    values = self.config['W_values']
                    variable_ranges_lines.append(f"- {var_name}: {values} µm (from W_values)")
                elif var_name.startswith('L_') and 'L_values' in self.config:
                    values = self.config['L_values']
                    variable_ranges_lines.append(f"- {var_name}: {values} µm (from L_values)")
        
        variable_ranges = '\n'.join(variable_ranges_lines) if variable_ranges_lines else "Not specified"
        
        # Build scaling rules section
        scaling_rules_lines = []
        
        if width_scales:
            scaling_rules_lines.append("**Width Scaling:**")
            for scaled_name, scale_pair in width_scales.items():
                if isinstance(scale_pair, (list, tuple)) and len(scale_pair) == 2:
                    base_var, factor = scale_pair
                    scaling_rules_lines.append(f"  {scaled_name} = {base_var} × {factor}")
        
        if length_scales:
            scaling_rules_lines.append("**Length Scaling:**")
            for scaled_name, scale_pair in length_scales.items():
                if isinstance(scale_pair, (list, tuple)) and len(scale_pair) == 2:
                    base_var, factor = scale_pair
                    scaling_rules_lines.append(f"  {scaled_name} = {base_var} × {factor}")
        
        if not scaling_rules_lines:
            scaling_rules_lines.append("No scaling - direct optimization of variables")
        
        scaling_rules = '\n'.join(scaling_rules_lines)
        
        # Extract variable impact summary from circuit_data
        variable_impact_lines = []
        for metric, impact_text in circuit_data['optimization_variables_impact'].items():
            variable_impact_lines.append(f"**{metric}:**\n{impact_text}")
        variable_impact_summary = '\n\n'.join(variable_impact_lines)
        
        # Extract variable interactions
        variable_interactions = circuit_data['variable_interactions']
        
        # Extract key insights
        key_insights_lines = []
        for i, insight in enumerate(circuit_data['key_insights_for_optimization'], 1):
            key_insights_lines.append(f"{i}. {insight}")
        key_insights = '\n'.join(key_insights_lines)
        
        # Format target_metric as human-readable string
        target_metric_str = f"""
    Metric: {self.target_metric.get('metric_name', 'Unknown')}
    Direction: {self.target_metric.get('direction', 'maximize')}
    Type: {self.target_metric.get('formulation_type', 'Unknown')}
    """
        
        # Build the prompt
        search_space_reduction_prompt = SEARCH_SPACE_REDUCTION_PROMPT.format(
            subckt_name=self.config['subckt_name'],
            target_metric=target_metric_str.strip(),
            num_variables_to_optimize=self.num_variables_to_optimize,
            total_num_variables=total_num_variables,
            variable_ranges=variable_ranges,
            scaling_rules=scaling_rules,
            variable_impact_summary=variable_impact_summary,
            variable_interactions=variable_interactions,
            key_insights=key_insights
        )
        
        search_space_response = self._generate_content(search_space_reduction_prompt, "Search Space Re-generation")

        # Parse JSON
        reduction_data = parse_llm_json_response(search_space_response)
        opt_config = extract_optimization_config(reduction_data)

        return opt_config


    

    def llm_regenerating_searchspace(self, feedback_dict, current_config, netlist, original_config):

        original_config = self.config
        params = prepare_format_params(feedback_dict, current_config, netlist, original_config)

        print("########################")
        print(params)
        print("########################")
        research_space_reduction_prompt = CIRCUIT_REUNDERSTANDING_PROMPT.format(**params)
        print("########################")
        print(research_space_reduction_prompt)
        print("########################")
        research_space_response = self._generate_content(research_space_reduction_prompt, "Circuit Re-understanding & Search Space Update")

         # Parse JSON
        reduction_data = parse_llm_json_response(research_space_response)

        print("###########  reduction_data #############")
        print(reduction_data)
        print("###########  reduction_data #############")
        opt_config = extract_optimization_config(reduction_data)

        return opt_config
        

    def _llm_select_objective(self, user_specs: str) -> Dict:
        
        
        """LLM selects the best objective function from candidates"""

        available_metrics = list(self.config["metrics"])
        prompt = objective_selection_prompt(user_specs, available_metrics)

        response = self._generate_content(prompt, "Objective Selection")
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        
        decision = json.loads(json_text)
        
        print(f"\n LLM-Selected Objective Function:")
        print(f"   Formula: {decision['objective_function']}")
        print(f"   Type: {decision['formulation']['type']}")
        print(f"   Reasoning: {decision['reasoning']}")
        print(f"   Expected Behavior: {decision['expected_behavior']}")
        
        return decision

    # def _extract_metric_info(self, target_metric: Dict) -> Dict:
    #     """Extract and organize metric information"""
    #     return {
    #         'name': target_metric['metric_name'],
    #         'key': target_metric['metric_key'],
    #         'unit': target_metric['metric_unit'],
    #         'format': target_metric['metric_format'],
    #         'direction': target_metric['direction'],
    #         'is_composite': target_metric['is_composite']
    #     }

    def _extract_metric_info(self, target_metric: Dict) -> Dict:
        """Extract and organize metric information - always returns FOM info"""
        fom_config = self.config.get('metric_post', {}).get('fom', {})
        
        return {
            'name': 'FOM',
            'key': 'fom',
            'unit': fom_config.get('unit', ''),
            'format': f".{fom_config.get('decimals', 4)}f",
            'direction': 'maximize',  # Always maximize FOM
            'is_composite': False  # FOM is pre-computed, not composite
        }

    def decide_next_iteration(self, optimization_state: Dict, user_specs: str) -> Dict:
        """llm_prompt_debug
        LLM decides the next iteration strategy
        
        Returns dict with:
        - 'action': 'search' or 'stop'
        - 'method': search method name
        - 'n_samples': number of designs to search
        - 'reasoning': explanation of decision
        """
        
        prompt = self._build_decision_prompt(optimization_state, user_specs)
        # DEBUG: Save prompt to file
        debug_path = os.path.join(self.config['results_dir'], 'llm_prompt_debug.txt')
        
        with open(debug_path, 'w') as f:
            f.write(prompt)
        print(" Saved LLM prompt to llm_prompt_debug.txt")

        # Extract the useful information from the history and Save in memory for feedback
        info_saved_memory = extract_useful_section(prompt)
        memory_path = os.path.join(self.config['results_dir'], 'optmizaiton_history_memory.txt')


        with open(memory_path, 'w') as f:
            f.write(info_saved_memory)
        print(" Saved state memory to memory.txt")
        
        # Generate response using LLM
        iteration_num = optimization_state.get('current_iteration', '?')
        response = self._generate_content(prompt, f"Iteration Decision (iteration {iteration_num})")

        # Parse LLM response
        response_text = response.text
        decision = self._parse_llm_decision(response_text)
        
        # Store in conversation history
        self.conversation_history.append({
            'prompt': prompt,
            'response': response_text,
            'decision': decision
        })
        
        return decision

    def save_decision_log(self, filepath: Path):
        """Save all LLM decisions for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

    def build_available_metrics(self, metric_post: dict) -> str:
        lines = []
    
        for metric, spec in metric_post.items():
            name = NAME_ALIASES.get(metric, metric.replace("_", " ").title())
    
            unit = spec.get("unit", "")
            expr = spec.get("expr")
    
            if expr:
                if unit:
                    line = f"- {name}: {expr}"
                else:
                    line = f"- {name}: {expr}"
            else:
                if unit:
                    line = f"- {name} ({unit})"
                else:
                    line = f"- {name}"
    
            lines.append(line)
    
        return "\n".join(lines)

    

    def _build_header_section(self, user_specs: str, metric_info: Dict) -> str:
        """Build the header section with user specs and goals"""
        available_metrics_text = self.build_available_metrics(self.config['metric_post'])
        return header_section(user_specs, metric_info, available_metrics_text)

    def _build_best_achieved_section(self, best_value: float, best_iter: int, 
                                     metric_info: Dict) -> str:
        """Build section showing best achieved value"""
        if best_value != float('-inf') and best_value != float('inf'):
            fmt = metric_info['format']
            unit = metric_info['unit']
            name = metric_info['name']
            return f"\n **Best post-PEX {name} achieved: {best_value:{fmt}} {unit} (Iteration {best_iter})**"
        return ""
    
    
    def _build_trend_analysis_section(self, post_pex_values: list, state: Dict, 
                                      metric_info: Dict) -> str:
        """Build trend analysis section"""
        if len(post_pex_values) < 2:
            return ""
        
        lines = [f"\n###  Trend Analysis (Post-PEX {metric_info['name']})"]
        
        # Format value history
        formatted_values = [f"{v:{metric_info['format']}}" for v in post_pex_values]
        lines.append(f"- Post-PEX {metric_info['name']} history: {formatted_values} {metric_info['unit']}")
        
        # Analyze recent improvements
        recent_values = post_pex_values[-3:] if len(post_pex_values) >= 3 else post_pex_values
        if len(recent_values) >= 2:
            lines.extend(self._analyze_recent_improvements(recent_values, metric_info))
        
        # Add method effectiveness
        recent_methods = [
            state['iterations'][i].get('method', 'unknown') 
            for i in range(-min(3, len(state['iterations'])), 0)
        ]
        lines.append(f"- Recent methods used: {recent_methods}")
        
        return "\n".join(lines)
        

    def _build_iteration_history_section(self, state: Dict, metric_info: Dict) -> str:
        """Build the iteration history section with detailed results"""
        if not state['iterations']:
            return ""
        
        sections = [
            f"### Previous Iterations ( Post-PEX {metric_info['name']} is the real metric!)",
            ""
        ]
        
        best_value, best_iter = self._initialize_best_tracking(metric_info['direction'])
        post_pex_values = []
        
        # Process each iteration
        for iter_data in state['iterations']:
            iter_summary = self._build_iteration_summary(
                iter_data, metric_info, best_value, best_iter
            )
            sections.append(iter_summary)
            
            # Update tracking
            if 'best_design_post_pex' in iter_data:
                post_value = self._get_metric_value(
                    iter_data['best_design_post_pex'], 
                    metric_info
                )
                if post_value is not None:
                    post_pex_values.append(post_value)
                    best_value, best_iter = self._update_best_tracking(
                        post_value, iter_data['iteration'], 
                        best_value, best_iter, metric_info['direction']
                    )
        
        # Add summary sections
        sections.append(self._build_best_achieved_section(best_value, best_iter, metric_info))
        sections.append(self._add_statistical_analysis(state))
        sections.append(self._build_trend_analysis_section(post_pex_values, state, metric_info))
        
        return "\n".join(sections)


    def _build_current_state_section(self, state: Dict) -> str:
        """Build the current state section with search space and budget info"""
        total_designs = state['total_designs_searched']
        num_var = state['num_var']
        budget_status = self._get_budget_status(total_designs)
        
        # Get variable names from config
        variables = self.config.get('variable', {})
        var_names = list(variables.keys())
        
        # Check if using scales (VCO/SC pattern)
        width_scales = self.config.get('width_scales', {})
        length_scales = self.config.get('length_scales', {})
        uses_scales = bool(width_scales or length_scales)
        
        # Build variable-specific ranges display
        var_info_lines = []
        total_combinations = 1
        
        if uses_scales:
            # VCO/SC pattern: Show scaled variables, not _base
            displayed_vars = set()
            
            # Add width-scaled variables
            for scaled_name, scale_pair in width_scales.items():
                if isinstance(scale_pair, (list, tuple)) and len(scale_pair) == 2:
                    base_var, _ = scale_pair
                    if base_var in var_names:
                        # Check for variable-specific values
                        specific_key = f"{scaled_name}_values"
                        if specific_key in self.config:
                            values = self.config[specific_key]
                        elif 'W_values' in self.config:
                            values = self.config['W_values']
                        else:
                            continue
                        
                        var_info_lines.append(f"- {scaled_name}: {values} ({len(values)} values)")
                        total_combinations *= len(values)
                        displayed_vars.add(base_var)
            
            # Add length-scaled variables
            for scaled_name, scale_pair in length_scales.items():
                if isinstance(scale_pair, (list, tuple)) and len(scale_pair) == 2:
                    base_var, _ = scale_pair
                    if base_var in var_names:
                        # Check for variable-specific values
                        specific_key = f"{scaled_name}_values"
                        if specific_key in self.config:
                            values = self.config[specific_key]
                        elif 'L_values' in self.config:
                            values = self.config['L_values']
                        else:
                            continue
                        
                        var_info_lines.append(f"- {scaled_name}: {values} ({len(values)} values)")
                        total_combinations *= len(values)
                        displayed_vars.add(base_var)
            
            # Add any remaining _base variables that weren't scaled
            for var_name in var_names:
                if var_name not in displayed_vars:
                    specific_key = f"{var_name}_values"
                    if specific_key in self.config:
                        values = self.config[specific_key]
                        var_info_lines.append(f"- {var_name}: {values} ({len(values)} values)")
                        total_combinations *= len(values)
        
        else:
            # Bandgap pattern: Direct variables without scales
            for var_name in var_names:
                # Check for variable-specific values first
                specific_key = f"{var_name}_values"
                
                if specific_key in self.config:
                    # Variable-specific range exists
                    values = self.config[specific_key]
                    var_info_lines.append(f"- {var_name}: {values} ({len(values)} values)")
                    total_combinations *= len(values)
                
                elif var_name.startswith('W_') and 'W_values' in self.config:
                    # Fallback to generic W_values
                    values = self.config['W_values']
                    var_info_lines.append(f"- {var_name}: {values} (from W_values, {len(values)} values)")
                    total_combinations *= len(values)
                
                elif var_name.startswith('L_') and 'L_values' in self.config:
                    # Fallback to generic L_values
                    values = self.config['L_values']
                    var_info_lines.append(f"- {var_name}: {values} (from L_values, {len(values)} values)")
                    total_combinations *= len(values)
                
                else:
                    # No range found
                    var_info_lines.append(f"- {var_name}: No range defined")
        
        # Build search space section
        search_space_lines = ["### Search Space"]
        search_space_lines.extend(var_info_lines)
        search_space_lines.append(f"- **Total possible combinations: {total_combinations:,}**")
        
        search_space_section = '\n'.join(search_space_lines)
        
        return f"""## CURRENT STATE
        {search_space_section}
        
        ### Budget and Progress
        - **Total designs searched: {total_designs}**
        - **Iterations completed: {state['num_iterations']}**
        - **Budget status**: {budget_status}"""


    def _build_methods_section(self, metric_info: Dict) -> str:
        """Build the search methods section with dynamic sample sizes from REDUCED search space (Step 2 output)"""
        # Use the reduced search space stored after Step 2 LLM output
        search_space_size = getattr(self, 'reduced_search_space_size', None)
        n_vars = getattr(self, 'reduced_n_vars', None)

        return methods_section(metric_info['name'], metric_info['direction'], self.user_specs,
                               search_space_size=search_space_size, n_vars=n_vars)

    def _build_decision_framework_section(self, metric_info: Dict) -> str:
        """Build the decision framework section"""
        return decision_framework_section(metric_info)
    
    def _build_parameter_tuning_section(self) -> str:
        """Build the algorithm parameter tuning section"""
        return parameter_tuning_section()
    
    
    def _build_response_format_section(self) -> str:

        """Build the response format section with JSON template"""
        return response_format_section()

    def _build_decision_prompt(self, state: Dict, user_specs: str) -> str:
        """Build prompt for LLM with current optimization state and advanced methods"""
        
        # Extract target metric
        target_metric = self.target_metric
        metric_info = self._extract_metric_info(target_metric)



            # Build prompt sections
        prompt_parts = [
            self._build_header_section(user_specs, metric_info),
            self._build_current_state_section(state),
            self._build_iteration_history_section(state, metric_info),
            self._build_methods_section(metric_info),
            self._build_decision_framework_section(metric_info),
            self._build_parameter_tuning_section(),
            self._build_response_format_section()
        ]
        
        return "\n\n".join(prompt_parts)


    def _parse_llm_decision(self, response_text: str) -> Dict:
        """Parse LLM response into structured decision"""
        try:
            json_text = self._extract_json_from_response(response_text)
            decision = json.loads(json_text)
            self._validate_decision(decision)
            return decision
            
        except Exception as e:
            print(f" Warning: Failed to parse LLM response: {e}")
            print(f"Response preview: {response_text[:300]}...")
            return self._create_fallback_decision(e)
        

    def _convert_decision_to_target_metric(self, decision: Dict) -> Dict:
        """Convert LLM decision to target_metric format"""
        formulation = decision.get('formulation', {})
        formulation_type = formulation.get('type', 'ratio')
        
        # Define base metrics for single objectives
        base_metrics = self.base_metric
        
        
        # TYPE 1: CONSTRAINT - handles both single objective and constraint-based
        if formulation_type == 'constraint':
            primary_metric = formulation.get('primary_metric', 'gain_db')
            constraints = formulation.get('constraints', [])
            
            # Get metric info from base_metrics
            metric_info = base_metrics.get(primary_metric, {
                'name': primary_metric,
                'unit': '',
                'format': '.3f',
                'degradation_key': None
            })
            
            # Single objective (no constraints) vs constraint-based
            is_single_objective = len(constraints) == 0
            
            return {
                'metric_name': metric_info['name'] if is_single_objective else decision.get('objective_function', 'Constrained'),
                'metric_key': primary_metric,
                'metric_unit': metric_info['unit'],
                'metric_format': metric_info['format'],
                'direction': formulation.get('primary_direction', 'maximize'),
                'is_composite': False,
                'degradation_key': metric_info['degradation_key'],
                'formulation_type': 'constraint',
                'constraints': constraints
            }
        
        # TYPE 2 & 3: RATIO / PRODUCT_RATIO
        elif formulation_type in ['ratio', 'product_ratio']:
            return {
                'metric_name': decision.get('objective_function', 'Composite'),
                'metric_key': 'composite',
                'metric_unit': '',
                'metric_format': '.4f',
                'direction': decision.get('direction', 'maximize'),
                'is_composite': True,
                'degradation_key': 'composite_percent',
                'numerator_keys': formulation.get('numerator', []),
                'denominator_keys': formulation.get('denominator', []),
                'formulation_type': formulation_type
            }
        
        # TYPE 4: WEIGHTED COMBINATION
        elif formulation_type in ['weighted_combination']:
            weights = formulation.get('weights', {})
            if not isinstance(weights, dict) or len(weights) == 0:
                raise ValueError(f"Invalid or missing weights for weighted_difference: {weights}")
    
            
            return {
                'metric_name': decision.get('objective_function', 'Weighted Combination'),
                'metric_key': 'composite',
                'metric_unit': '',
                'metric_format': '.4f',
                'direction': decision.get('direction', 'maximize'),
                'is_composite': True,
                'degradation_key': 'composite_percent',
                'weights': weights,
                'metrics': list(weights.keys()),
                'formulation_type': 'weighted_difference'
            }
        
        # TYPE 5: WEIGHTED PRODUCT
        elif formulation_type == 'weighted_product':
            numerator = formulation.get('numerator', ['gain_db'])
            denominator = formulation.get('denominator', ['power_uw'])
            
            return {
                'metric_name': decision.get('objective_function', 'Weighted Product'),
                'metric_key': 'composite',
                'metric_unit': '',
                'metric_format': '.4f',
                'direction': decision.get('direction', 'maximize'),
                'is_composite': True,
                'degradation_key': 'composite_percent',
                'formulation_type': 'weighted_product',
                'numerator_keys': numerator,
                'denominator_keys': denominator,
                'numerator_exponents': formulation.get('numerator_exponents', [1.0] * len(numerator)),
                'denominator_exponents': formulation.get('denominator_exponents', [1.0] * len(denominator))
            }
        
        # FALLBACK
        else:
            print(f" Unknown formulation type: {formulation_type}, using FOM fallback")
            return {
                'metric_name': 'FOM',
                'metric_key': 'fom',
                'metric_unit': '',
                'metric_format': '.3f',
                'direction': 'maximize',
                'is_composite': False,
                'degradation_key': 'fom_percent',
                'formulation_type': 'constraint',
                'constraints': []
            }


    def _compute_composite_metric(self, design: Dict, target_metric: Dict) -> float:
        """Get FOM value from design"""
        return design.get('fom', 0)

        
    def _add_statistical_analysis(self, state: Dict) -> str:
        """Generate statistical analysis of all designs to help LLM make better decisions"""
        
        # HELPER FUNCTION - Add at the start
        def safe_format(val, decimals=2):
            """Safely format a value with decimals"""
            if val is None:
                return "None"
            if isinstance(val, str):
                return val
            try:
                return f"{float(val):.{decimals}f}"
            except:
                return str(val)
        
        analysis = "\n###  STATISTICAL ANALYSIS OF SEARCH RESULTS\n"
    
        # Keep track of iterations we've already analyzed
        analyzed_iterations = set()
        
        for iter_data in state['iterations']:
            iter_num = iter_data.get('iteration', 0)
            if 'all_designs' not in iter_data or not iter_data['all_designs'] or iter_num in analyzed_iterations:
                continue
                
            analyzed_iterations.add(iter_num)
            
            method = iter_data.get('method', 'unknown').upper()
            designs = iter_data['all_designs']
            
            # Calculate distribution statistics for FOM - FILTER OUT NONE VALUES
            values = [d.get('fom') for d in designs if d.get('fom') is not None]
                
            if not values:
                analysis += f"\n**Iteration {iter_num} [{method}]:** No valid FOM values found\n"
                continue
            
            # Basic statistics
            mean_val = sum(values) / len(values)
            median_val = sorted(values)[len(values) // 2]
            min_val = min(values)
            max_val = max(values)
            std_dev = self._std_dev(values)
            
            # Parameter distribution analysis
            param_counts = {var: {} for var in self.var_names}
            
            # Count parameter usage
            for d in designs:
                for param in param_counts:
                    if param in d:
                        value = d[param]
                        if value is not None:  # Skip None values
                            if value not in param_counts[param]:
                                param_counts[param][value] = 0
                            param_counts[param][value] += 1
            
            # Format the analysis
            analysis += f"\n**Iteration {iter_num} [{method}] Statistical Analysis:**\n"
            analysis += f"- Total designs: {len(designs)} ({len(values)} with valid FOM)\n"
            analysis += f"- FOM distribution:\n"
            analysis += f"  - Mean: {mean_val:.4f}\n"
            analysis += f"  - Median: {median_val:.4f}\n"
            analysis += f"  - Min: {min_val:.4f}\n"
            analysis += f"  - Max: {max_val:.4f}\n"
            analysis += f"  - Std Dev: {std_dev:.4f}\n"
            analysis += f"  - Coefficient of variation: {(std_dev/mean_val if mean_val > 0 else 0):.4f}\n"
            
            # Calculate improvement percentiles
            percentiles = [10, 25, 50, 75, 90]
            sorted_vals = sorted(values, reverse=True)  # Always maximize FOM
            percentile_indices = [int(p/100 * len(sorted_vals)) for p in percentiles]
            percentile_values = [sorted_vals[min(i, len(sorted_vals)-1)] for i in percentile_indices]
            
            analysis += f"- Performance percentiles:\n"
            for p, v in zip(percentiles, percentile_values):
                analysis += f"  - {p}th percentile: {v:.4f}\n"
            
            # Parameter distribution
            analysis += f"- Parameter distribution:\n"
            for param, counts in param_counts.items():
                if counts:
                    analysis += f"  - {param} values: {counts}\n"
                
            # Top 3 most common parameter values - FIXED LINE
            for param, counts in param_counts.items():
                if counts:
                    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                    top3 = sorted_counts[:3]
                    # FIX: Use safe_format instead of f'{v:.2f}'
                    analysis += f"  - Most common {param}: {[safe_format(v, 2) for v, c in top3]}\n"
            
            # Exploration vs exploitation analysis
            if len(values) > 5:
                # Exploration: high std dev / range
                # Exploitation: many values close to max
                range_val = max_val - min_val
                exploration_score = std_dev / range_val if range_val > 0 else 0
                top_quartile = len([v for v in values if v >= percentile_values[3]]) / len(values)
                
                analysis += f"- Search behavior:\n"
                analysis += f"  - Exploration score: {exploration_score:.2f}\n"
                analysis += f"  - Exploitation score: {top_quartile:.2f}\n"
                
                if exploration_score > 0.3:
                    analysis += f"  - **High exploration** - diverse solutions found\n"
                elif top_quartile > 0.4:
                    analysis += f"  - **High exploitation** - many solutions near optimum\n"
                else:
                    analysis += f"  - **Balanced search** - good mix of exploration/exploitation\n"
        
        return analysis

    

    def _get_budget_status(self, total_designs: int) -> str:
        """Get budget status emoji and description"""
        if total_designs < 100:
            return ' Early stage (<100)'
        elif total_designs < 200:
            return ' Mid stage (100-200)'
        else:
            return ' Late stage (>200)'

    def _initialize_best_tracking(self, direction: str) -> tuple:
        """Initialize best value tracking based on optimization direction"""
        if direction == 'maximize':
            return float('-inf'), 0
        else:
            return float('inf'), 0

    def _update_best_tracking(self, current_value: float, current_iter: int, 
                             best_value: float, best_iter: int, 
                             direction: str) -> tuple:
        """Update best value tracking if current is better"""
        is_better = (current_value > best_value if direction == 'maximize' 
                     else current_value < best_value)
        
        if is_better:
            return current_value, current_iter
        return best_value, best_iter

    def _build_iteration_summary(self, iter_data: Dict, metric_info: Dict, 
                                 best_value: float, best_iter: int) -> str:
        """Build summary for a single iteration"""
        method = iter_data.get('method', 'unknown').upper()
        parameters = iter_data.get('parameters', {}) 
        
        lines = [
            f"\n**Iteration {iter_data['iteration']}** [{method}]:",
            f"- Designs searched: {iter_data['num_designs_searched']}"
        ]

        if parameters:
            param_lines = []
            for k, v in parameters.items():
                if isinstance(v, float):
                    param_lines.append(f"{k}={v:.2f}")
                else:
                    param_lines.append(f"{k}={v}")
            if param_lines:
                lines.append(f"- Algorithm parameters: {', '.join(param_lines)}")
            
        
        # Add pre-layout metrics
        lines.append(self._format_prelayout_metrics(iter_data, metric_info))
        
        # Add post-PEX metrics
        if 'best_design_post_pex' in iter_data:
            lines.extend(self._format_postpex_metrics(iter_data, metric_info))
        
        # Add design diversity
        if 'all_designs' in iter_data and iter_data['all_designs']:
            lines.append(self._format_top_designs(iter_data, metric_info))
        else:
            print(f"No designs found for iteration {iter_data['iteration']}")
        
        return "\n".join(lines)
    

    def _format_prelayout_metrics(self, iter_data: Dict, metric_info: Dict) -> str:
        """Format pre-layout metric value"""
        if metric_info['is_composite']:
            value = self._compute_composite_metric(
                iter_data['best_design_pre_layout'], 
                self.target_metric
            )
            return f"- Pre-layout {metric_info['name']}: {value:{metric_info['format']}}"
        else:
            value = iter_data['best_design_pre_layout'].get(metric_info['key'], 'N/A')
            if isinstance(value, (int, float)):
                return f"- Pre-layout {metric_info['name']}: {value:{metric_info['format']}} {metric_info['unit']}"
            return f"- Pre-layout {metric_info['name']}: {value}"
    
    
    def _format_postpex_metrics(self, iter_data: Dict, metric_info: Dict) -> list:
            """Format post-PEX metrics including degradation and design details"""
            lines = []
            post_design = iter_data['best_design_post_pex']
            post_value = self._get_metric_value(post_design, metric_info)
            
            if post_value is not None:
                # Format main metric with degradation if available
                metric_line = self._format_postpex_value_with_degradation(
                    iter_data, post_value, metric_info
                )
                lines.append(metric_line)
    
                values = post_design.results if hasattr(post_design, "results") else post_design
                # Add design parameters
                lines.append(
                    "- Design: " + ", ".join(
                        f"{var}={values[var]:.2f}"
                        for var in self.var_names
                    )
                )
                
                # Add performance metrics
                metrics = self.config["metric_post"]
                lines.append(
                    "Performance: " +
                    ", ".join(
                        f"{name}={values[name]:.3f}{info.get('unit', '')}"
                        for name, info in metrics.items()
                        if name in values
                    )
                )
            
            return lines
    
    
    def _format_postpex_value_with_degradation(self, iter_data: Dict, 
                                               post_value: float, 
                                               metric_info: Dict) -> str:
        """Format post-PEX value with degradation percentage if available"""
        fmt = metric_info['format']
        unit = metric_info['unit']
        name = metric_info['name']
        
        # Check for degradation data (only for non-composite metrics)
        if (not metric_info['is_composite'] and 
            'degradation' in iter_data):
            degradation_key = self.target_metric.get('degradation_key', '')
            degradation = iter_data['degradation'].get(degradation_key, 0)
            return (f"-  **Post-PEX {name}: {post_value:{fmt}} {unit}** "
                    f"(degradation: {degradation:+.1f}%)")
        else:
            return f"-  **Post-PEX {name}: {post_value:{fmt}} {unit}**"
    
    
    def _format_top_designs(self, iter_data: Dict, metric_info: Dict) -> str:

        """Format top 5 designs from the iteration using config['metric_post'] formats/units."""
        designs = iter_data['all_designs']
        # print(f"\n[DEBUG] _format_top_designs called for iteration {iter_data.get('iteration', '?')}")
        # print(f"[DEBUG] 'all_designs' in iter_data? {'all_designs' in iter_data}")
        # print(f"[DEBUG] iter_data keys: {list(iter_data.keys())}")
        # print(f"Found {len(designs)} designs for iteration {iter_data['iteration']}")
       
        if 'all_designs' not in iter_data:
            #print(f"[DEBUG] WARNING: No 'all_designs' key in iter_data!")
            return "\n  **No design data available**"
        
        debug_designs_path = os.path.join(self.config['results_dir'], 'debug_designs.txt')
        with open(debug_designs_path, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Iteration {iter_data.get('iteration', '?')}\n")
            f.write(f"Number of designs: {len(designs)}\n")
            if designs:
                first = designs[0]
                f.write(f"First design keys: {list(first.keys())}\n")
                f.write(f"Has dc_gain_db? {'dc_gain_db' in first}\n")
                f.write(f"First design: {first}\n")
            f.write(f"{'='*80}\n")
        if designs:
            first_design = designs[0]
            #print(f"\nDEBUG first design keys: {list(first_design.keys())}")
            #print(f"DEBUG first design sample: {dict(list(first_design.items())[:8])}")
    
        sorted_designs = self._sort_designs_by_metric(designs, metric_info)
        lines = [f"\n  **Top 5 designs found (by {metric_info['name']}) [Pre-layout rankings]:**"]
    
        metrics = self.config["metric_post"]
         
        #print(f"[DEBUG] self.config['metric_post'] keys: {list(metrics.keys())}")
        
        # ADD THIS: Check first design values
        if designs:
            first = designs[0]
            #print(f"[DEBUG] First design metric values:")
            for m in metrics:
                val = first.get(m)
                #print(f"[DEBUG]   {m} = {val} (type: {type(val).__name__})")
    
        for i, design in enumerate(sorted_designs[:5], 1):
            values = design

            #if i == 1:  # Only debug first design
                #print(f"[DEBUG] Design #{i} keys: {list(values.keys())}")
                #print(f"[DEBUG] Design #{i} dc_gain_db: {values.get('dc_gain_db')}")
    
            comp_val = self._get_metric_value(design, metric_info)
            if isinstance(comp_val, (int, float)):
                comp_str = f"{metric_info['name']}={comp_val:{metric_info['format']}}"
            else:
                comp_str = f"{metric_info['name']}={comp_val}"
    
            params = ", ".join(
                f"{var}={values.get(var):.2f}" if isinstance(values.get(var), (int, float)) else f"{var}=NA"
                for var in self.var_names
            )
    
            # Try to get metrics, but expect them to be NA for pre-layout
            perf_parts = []
            for m in metrics:
                val = values.get(m)
                if val is not None and val != 'NA':
                    try:
                        formatted_val = f"{float(val):.{metrics[m]['decimals']}f}"
                        unit = metrics[m].get('unit', '')
                        perf_parts.append(f"{m}={formatted_val}{unit}")
                    except Exception as e:
                        #print(f"[DEBUG] ERROR formatting {m}: {e}")
                        perf_parts.append(f"{m}=NA{metrics[m].get('unit', '')}")
                else:
                    perf_parts.append(f"{m}=NA{metrics[m].get('unit', '')}")
    
            perf = f" | {' '.join(perf_parts)}"
    
            lines.append(f"    {i}. {comp_str}  ({params}{perf})")
    
        return "\n".join(lines)
    
    

    def _sort_designs_by_metric(self, designs: list, metric_info: Dict) -> list:
        """Sort designs by target metric value, filtering out None values"""
        if not designs:
            return []
        
        reverse = (metric_info['direction'] == 'maximize')
        metric_key = metric_info.get('key', 'unknown')
        
        # Filter out invalid designs
        valid_designs = []
        for design in designs:
            try:
                if metric_info['is_composite']:
                    value = self._compute_composite_metric(design, self.target_metric)
                else:
                    value = design.get(metric_key)
                
                # Only keep designs with non-None values
                if value is not None:
                    valid_designs.append(design)
            except Exception:
                pass  # Skip designs that cause errors
        
        # Check if we filtered anything
        if len(valid_designs) < len(designs):
            filtered_count = len(designs) - len(valid_designs)
            print(f"  ⚠️  Filtered {filtered_count}/{len(designs)} designs with invalid {metric_key}")
        
        # If all designs are invalid, return empty list
        if not valid_designs:
            print(f"  ❌ All {len(designs)} designs have invalid {metric_key}")
            return []
        
        # Sort valid designs only
        try:
            if metric_info['is_composite']:
                return sorted(
                    valid_designs,
                    key=lambda x: self._compute_composite_metric(x, self.target_metric),
                    reverse=reverse
                )
            else:
                return sorted(
                    valid_designs,
                    key=lambda x: x.get(metric_key, 0),
                    reverse=reverse
                )
        except Exception as e:
            print(f"  ❌ Error sorting designs: {e}")
            return valid_designs  # Return unsorted if sorting fails
    
    
    
    def _get_metric_value(self, design: Dict, metric_info: Dict) -> float:
        """Extract metric value from design based on whether it's composite"""
        if metric_info['is_composite']:
            return self._compute_composite_metric(design, self.target_metric)
        else:
            return design.get(metric_info['key'])

    def _analyze_recent_improvements(self, recent_values: list, metric_info: Dict) -> list:
        """Analyze improvement trends in recent iterations"""
        lines = []
        direction = metric_info['direction']
        
        # Calculate improvement
        if direction == 'maximize':
            improvement = recent_values[-1] - recent_values[0]
        else:
            improvement = recent_values[0] - recent_values[-1]
        
        improvement_pct = (improvement / abs(recent_values[0]) * 100) if recent_values[0] != 0 else 0
        
        fmt = metric_info['format'][1:]  # Remove the dot
        lines.append(
            f"- Recent improvement: {improvement:+{fmt}} {metric_info['unit']} "
            f"({improvement_pct:+.1f}%)"
        )
        
        # Convergence detection for 3 iterations
        if len(recent_values) == 3:
            convergence_status = self._detect_convergence(recent_values, direction)
            lines.append(convergence_status)
        
        return lines
        
    
    def _detect_convergence(self, recent_values: list, direction: str) -> str:
        """Detect convergence based on recent improvement patterns"""
        if direction == 'maximize':
            improvements = [recent_values[i+1] - recent_values[i] 
                           for i in range(len(recent_values)-1)]
        else:
            improvements = [recent_values[i] - recent_values[i+1] 
                           for i in range(len(recent_values)-1)]
        
        max_improvement_pct = (
            max(abs(imp) / abs(recent_values[0]) * 100 for imp in improvements)
            if recent_values[0] != 0 else 0
        )
        
        if max_improvement_pct < 2:
            return " **STRONG CONVERGENCE**: Improvements < 2% - Consider STOPPING!"
        elif max_improvement_pct < 5:
            return " **CONVERGENCE DETECTED**: Improvements < 5%"
        else:
            return "**Still improving**: Improvements > 5%"
    

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON string from response text (handles markdown code blocks)"""
        # Try markdown JSON code block
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            return response_text[json_start:json_end].strip()
        
        # Try generic markdown code block
        if "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            return response_text[json_start:json_end].strip()
        
        # Try to find raw JSON object
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            return response_text[json_start:json_end]
        
        raise ValueError("No JSON found in response")
    
    
    def _validate_decision(self, decision: Dict):
        """Validate decision structure and values"""
        if decision['action'] not in ['search', 'stop']:
            raise ValueError(f"Invalid action: {decision['action']}")
        
        if decision['action'] == 'search':
            self._validate_search_decision(decision)
    
    
    def _validate_search_decision(self, decision: Dict):
        """Validate search-specific decision fields"""
        valid_methods = [
            'lhs', 'genetic', 'bayesian', 'optuna', 'adaptive', 
            'annealing', 'multistart', 'random', 'refined', 'grid'
        ]
        
        if decision['method'] not in valid_methods:
            raise ValueError(f"Invalid method: {decision['method']}")
        
        # Dynamic upper bound based on reduced search space
        import math
        _space = getattr(self, 'reduced_search_space_size', None)
        _nvars = getattr(self, 'reduced_n_vars', None)
        if _space and _nvars:
            _max_samples = min(100, max(4 * _nvars, int(math.sqrt(_space) * 3.0)))
        else:
            _max_samples = 100
        if not (5 <= decision['n_samples'] <= _max_samples):
            raise ValueError(f"n_samples out of range: {decision['n_samples']}")
        
        # Validate parameters if provided
        if 'parameters' in decision:
            self._validate_method_parameters(decision['method'], decision['parameters'])
    
    
    def _validate_method_parameters(self, method: str, parameters: Dict):
        """Validate method-specific parameters"""
        # Define valid parameter ranges for each method
        param_ranges = {
            'genetic': {
                'mutation_rate': (0.05, 0.5),
                'crossover_rate': (0.7, 0.9),
                'tournament_size': (2, 5)
            },
            'bayesian': {
                'acquisition_function': ['EI', 'PI', 'LCB', 'UCB'],
                'exploration_weight': (0.0, 5.0),  # Wide range to accommodate both xi and kappa
            },
            'optuna': {
                'n_ei_candidates': (10, 50),
                'n_startup_trials': (3, 10),
                'multivariate': [True, False],
                'prior_weight': (0.5, 2.0)
            },
            
            'annealing': {
                'initial_temperature': (0.5, 3.0),
                'cooling_rate': (0.8, 0.99)
            },
            'adaptive': {
                'explore_weight': (0.2, 0.7),
                'exploit_weight': (0.2, 0.7),
                'random_weight': (0.0, 0.3)
            },
            'multistart': {
                'n_starts': (3, 8),
                'search_radius': (1, 3)
            },
            'lhs': {
                'seed': (0, 1000000)
            }
        }
        
        if method not in param_ranges:
            return  # No validation rules for this method
        
        valid_params = param_ranges[method]
        
        for param_name, param_value in parameters.items():
            if param_name not in valid_params:
                print(f" Warning: Unknown parameter '{param_name}' for method '{method}'")
                continue
            
            expected = valid_params[param_name]
            
            # Validate range for numeric parameters
            if isinstance(expected, tuple):
                min_val, max_val = expected
                if not (min_val <= param_value <= max_val):
                    raise ValueError(
                        f"Parameter '{param_name}' = {param_value} is out of range "
                        f"[{min_val}, {max_val}] for method '{method}'"
                    )
            
            # Validate choice for string parameters
            elif isinstance(expected, list):
                if param_value not in expected:
                    raise ValueError(
                        f"Parameter '{param_name}' = {param_value} is not valid. "
                        f"Expected one of {expected} for method '{method}'"
                    )
    
    
    def _create_fallback_decision(self, error: Exception) -> Dict:
        """Create intelligent fallback decision on parsing error"""
        return {
            'action': 'search',
            'method': 'bayesian',  # Safe, sample-efficient default
            'n_samples': 15,
            'parameters': {
                'acquisition_function': 'EI',
                'exploration_weight': 0.1
            },
            'reasoning': f'Fallback decision due to parsing error: {error}. Using Bayesian for safety.',
            'confidence': 'low'
        }

    def _std_dev(self, values: List[float]) -> float:
        
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


# Integration with existing optimizer
def run_llm_guided_optimization(config, max_total_designs: int = 250,
                                num_variables_to_optimize: int = 3,
                                pre_layout_only: bool = False,
                                max_regeneration_cycles: int = 2, plateau_patience: int = 2,
                                trial_index: int = 0,
                                model: str = "gemini-2.5-flash"):
    """
    Main function to run LLM-guided optimization with advanced methods and search space reduction

    Parameters:
    -----------
    config: dict or str
        Optimization configuration path or dictionary
    max_total_designs: int
        Maximum total designs to search across all iterations
    user_specs: str
        User specifications and goals (e.g., "Maximize gain > 40dB, minimize power")
    num_variables_to_optimize: int
        Number of top variables to actively optimize (rest will be fixed by LLM)
    pre_layout_only: bool
        If True, optimize based on pre-layout only (skip ALIGN and post-PEX for speed)
    model: str
        LLM model to use (gemini-2.5-flash, claude-sonnet-4-5, gpt-4o, Llama-3.3-70B-Instruct)
    """
    
    from iterative_ota_optimization_test import ControlledOTAOptimizer

    trial_start_time = time.time()
    
    # Try to import advanced search methods
    try:
        from advanced_search_methods_test import enhanced_generate_search_points
        has_advanced_methods = True
        print(" Advanced search methods loaded successfully")
    except ImportError:
        has_advanced_methods = False
        print("  Advanced search methods not found - using basic methods only")
        print("   To enable advanced methods, save advanced_search_methods.py in the same directory")
    
    # Load config
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    user_specs = config['user_specs']

    user_specs_metric = config['user_specs_metric']
    
    # Initialize LLM agent
    print("\n" + "="*100)
    print("LLM-GUIDED OTA OPTIMIZATION WITH INTELLIGENT SEARCH SPACE REDUCTION")
    print("="*100)
    print("Using AI agent to intelligently guide the optimization process")
    if has_advanced_methods:
        print("Advanced algorithms: LHS, Genetic, Bayesian, Adaptive, Annealing, Multi-start")
    
    # Show mode
    if pre_layout_only:
        print(" Mode: PRE-LAYOUT OPTIMIZATION (Fast - ALIGN/Post-PEX skipped)")
    else:
        print(" Mode: FULL FLOW (Pre-layout → ALIGN → Post-PEX)")
    
    print("="*100)
    print(f" Model: {model}")
    print("="*100)

    llm_agent = LLMOptimizationAgent(config=config, model=model, user_specs=user_specs, num_variables_to_optimize=num_variables_to_optimize)
    
    # Extract target metric
    target_metric = llm_agent.target_metric
    print(f"Target Metric: {target_metric['metric_name']} ({target_metric['direction']})")

    specs_met = False
    #optimization_complete = False

    all_designs_across_cycles = []
    all_iteration_history_all_cycles = []

    # Token usage tracking per cycle
    cycle_token_usage = []

    for regeneration_cycle in range(max_regeneration_cycles):
        if specs_met:
            print(f"\n USER SPECIFICATIONS MET, skipping regeneration cycle {regeneration_cycle}")
            break

        # Reset token counter for this cycle
        cycle_start_usage = llm_agent.get_token_usage_summary()

        if regeneration_cycle == 0:
            print("\n" + "="*100)
            print("INITIAL SEARCH SPACE GENERATION")
            print("="*100)
        else:
            print("\n" + "="*100)
            print(f"SEARCH SPACE REGENERATION CYCLE {regeneration_cycle}")
            print("="*100)
            print("Re-analyzing search space based on optimization feedback...")
    
        # ========================================================================
        # STEP 1: CIRCUIT UNDERSTANDING (LLM Analysis)
        # ========================================================================
        if regeneration_cycle == 0:
            print("\n" + "="*100)
            print("STEP 1: CIRCUIT UNDERSTANDING (LLM Analysis)")
            print("="*100)
            
            print("Analyzing circuit topology and variable impacts...")
            
            circuit_data = llm_agent.circuits_understanding()
            
            if circuit_data:
                print(" Circuit understanding complete!")
                print(f"\n Analysis Summary:")
                print(f"   • Circuit Overview: {circuit_data['circuit_topology_overview']}...")
                mapping = circuit_data.get('optimization_variables_mapping', {})
                print(f"   • Variables Analyzed: {len(mapping) if isinstance(mapping, dict) else len(str(mapping).split('.'))}")
                print(f"   • Key Insights: {len(circuit_data['key_insights_for_optimization'])}")
                
                print(f"\n Top Insights:")
                for i, insight in enumerate(circuit_data['key_insights_for_optimization'], 1):
                    print(f"   {i}. {insight}")
            else:
                print(" Circuit understanding failed, proceeding without search space reduction")
                circuit_data = None
    
        # ========================================================================
        # STEP 1.5: TRANSISTOR GROUPING (if high-dimensional)
        # ========================================================================
        if regeneration_cycle == 0:
            n_total_vars = len(config.get('variable', {}))
            if n_total_vars > 20:
                print("\n" + "="*100)
                print(f"STEP 1.5: AUTOMATIC TRANSISTOR GROUPING ({n_total_vars} variables → groups)")
                print("="*100)
                try:
                    grouping_data = llm_agent.transistor_grouping()
                    # Update config reference since transistor_grouping modifies llm_agent.config
                    config = llm_agent.config
                except Exception as e:
                    print(f"  ❌ Grouping failed: {e}")
                    print(f"  Proceeding with all {n_total_vars} individual variables")
                    import traceback
                    traceback.print_exc()

        # ========================================================================
        # STEP 2: SEARCH SPACE REDUCTION (LLM Prioritization)
        # ========================================================================
        if regeneration_cycle == 0:
            print("\n" + "="*100)
            print("STEP 2: INTELLIGENT SEARCH SPACE REDUCTION (LLM Prioritization)")
            print("="*100)

            if circuit_data:
                print(f"Determining which {num_variables_to_optimize} variables to optimize...")
                #optimization_config = llm_agent.search_space_generating(circuit_data)
                optimization_config = llm_agent.search_space_generating_new(circuit_data)
                # Store reduced search space on agent for use in methods_section prompt
                llm_agent.reduced_search_space_size = optimization_config.get('num_combinations')
                llm_agent.reduced_n_vars = len(optimization_config.get('variables_to_optimize', {}))

              
            else:
                optimization_config = None
    
        else:
            optimization_config = None
            # Regenerate search space based on feedback
            print("\n" + "="*100)
            print(f"STEP 2 (CYCLE {regeneration_cycle}): SEARCH SPACE REGENERATION")
            print("="*100)
    
            #load the last iteration opt_config
            previous_config_path = os.path.join(config['results_dir'], f'opt_config_iter{regeneration_cycle-1}.json')
            with open(previous_config_path, 'r') as f:
                previous_opt_config = json.load(f)
            
            source_path = os.path.join(config['results_dir'], 'optmizaiton_history_memory.txt')  #optmizaiton_history_memory  llm_prompt_debug
            memory_path = os.path.join(config['results_dir'], f'optmizaiton_history_memory_iter{regeneration_cycle+1}.txt')
            shutil.copy(source_path, memory_path)
            
            feedback = extract_optimization_feedback(memory_path)
            print("="*80)
            print("OPTIMIZATION FEEDBACK")
            print("="*80)
            pprint.pprint(feedback, width=100, compact=False)
            print()
    
            optimization_config = llm_agent.llm_regenerating_searchspace(feedback_dict=feedback,
                                                                         current_config=previous_opt_config,
                                                                         netlist=config['ota_subckt_template'],
                                                                         original_config=config)
            # Update reduced search space on agent
            if optimization_config:
                llm_agent.reduced_search_space_size = optimization_config.get('num_combinations')
                llm_agent.reduced_n_vars = len(optimization_config.get('variables_to_optimize', {}))
            
                
        if optimization_config:
            print(" Search space reduction complete!")
            print("\n OPTIMIZATION CONFIGURATION:")
            print(f"   Variables to optimize: {len(optimization_config['variables_to_optimize'])}")
            for var, values in optimization_config['variables_to_optimize'].items():
                print(f"{var}: {len(values)} choices from {values}")
            
            print(f"   Fixed variables: {len(optimization_config['variables_fixed'])}")
            for var, value in optimization_config['variables_fixed'].items():
                print(f"{var}: fixed at {value}")
            
            summary = optimization_config['search_space_summary']
            reduction_factor = summary.get('reduction_factor', 'unknown')
            print(f"\n  Search space: {summary['reduced_search_space']} combinations")
            print(f"  (Reduced from {summary['original_full_space']} - {reduction_factor} smaller!)")
            print(f"  Calculation: {summary['calculation']}")
        else:
            print("  Search space reduction failed, using full search space")
            optimization_config = None

        
        # ========================================================================
        # STEP 3: INITIALIZE OPTIMIZER WITH LLM CONFIGURATION
        # ========================================================================
        print("\n" + "="*100)
        print("STEP 3: INITIALIZING OPTIMIZER")
        print("="*100)
        
        optimizer = ControlledOTAOptimizer(
            config=config,
            user_specs=user_specs,
            llm_agent=llm_agent,
            optimization_config=optimization_config
            
        )
        
        if optimization_config:
            
            print(" Optimizer initialized with LLM-guided search space reduction")
            
            config_path = os.path.join(config['results_dir'], f'opt_config_iter{regeneration_cycle}.json')
            with open(config_path, 'w') as f:
                json.dump(optimization_config, f, indent=2)
                    
        else:
            print(" Optimizer initialized with full search space")
        
        if has_advanced_methods:
            print(" Advanced search methods enabled")
        
        # ========================================================================
        # STEP 4: ITERATIVE OPTIMIZATION
        # ========================================================================
        print("\n" + "="*100)
        print("STEP 4: ITERATIVE OPTIMIZATION")
        print("="*100)
        
        iteration = 0
        previous_best = None

        plateau_count = 0
        last_best_fom = None
        improvement_threshold = 0.001  # 0.1% minimum improvement
        fom_converged = False  # NEW: Add convergence flag



        if optimization_config['num_combinations'] < max_total_designs:
            total_designs = optimization_config['num_combinations']
        else:
            total_designs = max_total_designs

            
        while (len(optimizer.all_searched_designs) < total_designs and not specs_met and not fom_converged):
            iteration += 1

            # Build state for LLM
            state = {
                'num_var': len(config["variable"]),
                'total_designs_searched': len(optimizer.all_searched_designs),
                'num_iterations': len(optimizer.iteration_history),
                'iterations': []
            }
            
            # Add W_values if exists
            if "W_values" in config:
                state['W_values'] = config["W_values"]
            
            # Add L_values if exists
            if "L_values" in config:
                state['L_values'] = config["L_values"]
            
            # Add all parameters from YAML (works for both OTA and VCO)
            if "params" in config:
                for param_name, param_value in config["params"].items():
                    state[param_name] = param_value
            
            # Add optimization config info to state
            if optimization_config:
                state['optimization_config'] = {
                    'num_optimized': len(optimization_config['variables_to_optimize']),
                    'num_fixed': len(optimization_config['variables_fixed']),
                    'search_space_size': optimization_config['search_space_summary']['reduced_search_space']
                }
            
            # Add iteration history
            for i, iter_result in enumerate(optimizer.iteration_history):
                iter_dict = {
                    'iteration': iter_result.iteration,
                    'num_designs_searched': iter_result.num_designs_searched,
                    'method': getattr(iter_result, 'method', 'unknown'),
                    'parameters': getattr(iter_result, 'parameters', {}), 
                    'best_design_pre_layout': iter_result.pre_layout.to_dict(),
                }
                if iter_result.post_pex:
                    iter_dict['best_design_post_pex'] = iter_result.post_pex.to_dict()
                    iter_dict['degradation'] = iter_result.degradation_percent
                
                # Calculate indices
                start_idx = sum(ir.num_designs_searched for ir in optimizer.iteration_history[:i])
                end_idx = start_idx + iter_result.num_designs_searched
    
                
                # Add safety check
                if start_idx < len(optimizer.all_searched_designs) and end_idx <= len(optimizer.all_searched_designs): 
                    iter_dict['all_designs'] = [
                        d.to_dict() for d in optimizer.all_searched_designs[start_idx:end_idx]
                    ]
                
                state['iterations'].append(iter_dict)
            
            # Ask LLM what to do next
            print(f"\n{'='*100}")
            print(f"ITERATION {iteration}: CONSULTING LLM AGENT...")
            print(f"{'='*100}")
            print(f"Designs searched so far: {len(optimizer.all_searched_designs)}/{total_designs}")
            
            decision = llm_agent.decide_next_iteration(state, user_specs)
            
            print(f"\n LLM DECISION:")
            print(f"  Action: {decision['action'].upper()}")
            if decision['action'] == 'search':
                print(f"  Method: {decision['method'].upper()}")
                print(f"  Samples: {decision['n_samples']}")
                if 'parameters' in decision:
                    print(f"  Parameters: {decision['parameters']}")
            
            print(f"  Confidence: {decision.get('confidence', 'unknown')}")
            print(f"  Reasoning: {decision['reasoning']}")
            if 'expected_improvement' in decision:
                print(f"  Expected: {decision['expected_improvement']}")
            if 'convergence_assessment' in decision:
                print(f"  Convergence: {decision['convergence_assessment']}")
            
            # Execute decision
            if decision['action'] == 'stop':
                print(f"\n{'='*100}")
                print(" LLM DECIDED TO STOP OPTIMIZATION")
                print(f"{'='*100}")
                print(f"Reason: {decision['reasoning']}")      
                break

            
            # Check if method is available
            if not has_advanced_methods and decision['method'] not in ['random', 'grid', 'refined']:
                print(f"Warning: Advanced method '{decision['method']}' not available, falling back to 'random'")
                decision['method'] = 'random'
            
            # Run iteration with LLM's strategy
            print(f"\n Running search with {decision['method']} method...")
            iter_result = optimizer.run_iteration(
                iteration=iteration,
                n_samples=decision['n_samples'],
                trial_index=regeneration_cycle,
                search_method=decision['method'],
                previous_best=previous_best,
                algorithm_params=decision.get('parameters', {}),
                pre_layout_only=pre_layout_only  # ← ADD THIS LINE
            )
            
            if iter_result:
                iter_result.method = decision['method']
                iter_result.parameters = decision.get('parameters', {})
                
                # Update previous_best based on mode
                if pre_layout_only:
                    previous_best = iter_result.pre_layout
                    best_design_dict = iter_result.pre_layout.to_dict()
                else:
                    previous_best = iter_result.post_pex if iter_result.post_pex else iter_result.pre_layout
                    #best_design_dict = (iter_result.post_pex if iter_result.post_pex else iter_result.pre_layout).to_dict()
                    best_design_dict = iter_result.pre_layout.to_dict()

                print(f"\nDEBUG: Checking specs for best design:")
                print(f"  user_specs_metric = '{user_specs_metric}'")
                print(f"  design_dict keys = {list(best_design_dict.keys())}")
                print(f"  fom = {best_design_dict.get('fom')}")
                print(f"  dc_gain_db = {best_design_dict.get('dc_gain_db')}")
                print(f"  ugbw = {best_design_dict.get('ugbw')}")
                print(f"  power_dc = {best_design_dict.get('power_dc')}")
                
                specs_met = check_user_specs_met(best_design_dict, user_specs_metric, verbose=True, fom_only_check=False)
                print(f"DEBUG: specs_met returned = {specs_met}")
                if specs_met:
                    print("\n" + "="*100)
                    print("USER SPECIFICATIONS MET!")
                    print("="*100)


                # Get current best FOM for plateau detection
                current_best_fom = best_design_dict.get('fom', 0)
                
                # Check for FOM plateau
                if last_best_fom is not None:
                    # Calculate relative improvement (positive = better, negative = worse)
                    if last_best_fom != 0:
                        relative_change = (current_best_fom - last_best_fom) / abs(last_best_fom)
                    else:
                        relative_change = current_best_fom - last_best_fom
                    
                    # Check if change is significant (use absolute value for threshold comparison)
                    if abs(relative_change) < improvement_threshold:
                        plateau_count += 1
                        print(f"\n FOM PLATEAU DETECTED ({plateau_count}/{plateau_patience})")
                        print(f"   Last FOM: {last_best_fom:.6f}")
                        print(f"   Current FOM: {current_best_fom:.6f}")
                        print(f"   Change: {relative_change*100:.3f}% (threshold: {improvement_threshold*100:.1f}%)")
                    elif relative_change > 0:
                        # FOM increased = improvement (since we're maximizing)
                        plateau_count = 0
                        print(f"\n FOM IMPROVED")
                        print(f"   Previous: {last_best_fom:.6f}")
                        print(f"   Current: {current_best_fom:.6f}")
                        print(f"   Improvement: {relative_change*100:.2f}%")
                    else:
                        # FOM decreased = degradation
                        plateau_count = 0
                        print(f"\n FOM DEGRADED")
                        print(f"   Previous: {last_best_fom:.6f}")
                        print(f"   Current: {current_best_fom:.6f}")
                        print(f"   Degradation: {abs(relative_change)*100:.2f}%")
                    
                    last_best_fom = current_best_fom
                
                # Check if plateau limit reached
                if plateau_count >= plateau_patience:
                    fom_converged = True  # NEW: Set the flag instead of break
                    print(f"\n{'='*100}")
                    print(f"FOM CONVERGED - STOPPING OPTIMIZATION")
                    print(f"{'='*100}")
                    print(f"FOM has plateaued for {plateau_patience} consecutive iterations at {current_best_fom:.6f}")
                    print(f"Total designs evaluated: {len(optimizer.all_searched_designs)}")
                    print(f"Pre-layout optimization converged")
                    print(f"{'='*100}\n")
                    # Don't break here, let the while loop handle it

                
                # Print summary based on mode
                if pre_layout_only:
                    # Pre-layout only mode
                    print(f"\n Iteration {iteration} Summary (Pre-layout):")
                    
                    # Get FOM directly from results
                    pre_dict = iter_result.pre_layout.to_dict()
                    pre_value = pre_dict.get('fom', 0)
                    
                    # Get FOM formatting from config
                    fom_config = config.get('metric_post', {}).get('fom', {})
                    fom_format = f".{fom_config.get('decimals', 4)}f"
                    fom_unit = fom_config.get('unit', '')
                    
                    print(f"   FOM: {pre_value:{fom_format}}{fom_unit}")
                    
                    # Show variable status (first iteration only)
                    if optimization_config and iteration == 1:
                        print(f"\n Variable Status:")
                        best_dict = iter_result.pre_layout.to_dict()
                        for var in optimization_config['variables_to_optimize'].keys():
                            print(f"    {var} = {best_dict.get(var)} (optimized)")
                            
                        for var, fixed_val in optimization_config['variables_fixed'].items():
                            actual_val = best_dict.get(var)
                        
                            if fixed_val is None or actual_val is None:
                                status = "N/A"
                                note = "not fixed"
                            else:
                                # Convert to float for comparison
                                try:
                                    actual_float = float(actual_val)
                                    fixed_float = float(fixed_val)
                                    status = "OK" if abs(actual_float - fixed_float) < 0.01 else "WARN"
                                    note = f"fixed at {fixed_val}"
                                except (TypeError, ValueError):
                                    status = "N/A"
                                    note = f"comparison error"
                        
                            print(f"    {var} = {actual_val} {status} ({note})")
                                    
                elif iter_result.post_pex:
                    # Full flow mode
                    print(f"\n Iteration {iteration} Summary:")
                    
                    # Get FOM values from both pre and post
                    pre_dict = iter_result.pre_layout.to_dict()
                    post_dict = iter_result.post_pex.to_dict()
                    
                    pre_value = pre_dict.get('fom', 0)
                    post_value = post_dict.get('fom', 0)
                    
                    # Calculate degradation
                    degradation_pct = ((post_value - pre_value) / pre_value * 100) if pre_value != 0 else 0
                    
                    # Get FOM formatting from config
                    fom_config = config.get('metric_post', {}).get('fom', {})
                    fom_format = f".{fom_config.get('decimals', 4)}f"
                    fom_unit = fom_config.get('unit', '')
                    
                    print(f"   Pre-layout FOM:  {pre_value:{fom_format}}{fom_unit}")
                    print(f"   Post-PEX FOM:    {post_value:{fom_format}}{fom_unit}")
                    print(f"   Degradation:     {degradation_pct:+.1f}%")
                    
                    # Show which variables were optimized vs fixed
                    if optimization_config and iteration == 1:
                        print(f"\n📋 Variable Status:")
                        best_dict = iter_result.post_pex.to_dict()
                        
                        for var in optimization_config['variables_to_optimize'].keys():
                            print(f"    {var} = {best_dict.get(var)} (optimized)")
                        
                        for var, fixed_val in optimization_config['variables_fixed'].items():
                            actual_val = best_dict.get(var)
                            
                            # Handle None values
                            if actual_val is None:
                                status = "❓"
                                print(f"   📌 {var} = None {status} (fixed at {fixed_val}, but not found in result)")
                            elif abs(actual_val - fixed_val) < 0.01:
                                status = "✅"
                                print(f"   📌 {var} = {actual_val} {status} (fixed at {fixed_val})")
                            else:
                                status = "⚠️"
                                print(f"   📌 {var} = {actual_val} {status} (fixed at {fixed_val})")
                else:
                    print(" Iteration failed")
                    break

        all_designs_across_cycles.extend(optimizer.all_searched_designs)
        all_iteration_history_all_cycles.extend(optimizer.iteration_history)

        # Track token usage for this cycle
        cycle_end_usage = llm_agent.get_token_usage_summary()
        cycle_usage = {
            'cycle': regeneration_cycle,
            'input_tokens': cycle_end_usage['total_input_tokens'] - cycle_start_usage['total_input_tokens'],
            'output_tokens': cycle_end_usage['total_output_tokens'] - cycle_start_usage['total_output_tokens'],
            'total_tokens': cycle_end_usage['total_tokens'] - cycle_start_usage['total_tokens'],
            'api_calls': cycle_end_usage['api_call_count'] - cycle_start_usage['api_call_count']
        }
        cycle_token_usage.append(cycle_usage)

        print(f"\n{'='*100}")
        print(f"REGENERATION CYCLE {regeneration_cycle} - TOKEN USAGE SUMMARY")
        print(f"{'='*100}")
        print(f"  Input Tokens:  {cycle_usage['input_tokens']:,}")
        print(f"  Output Tokens: {cycle_usage['output_tokens']:,}")
        print(f"  Total Tokens:  {cycle_usage['total_tokens']:,}")
        print(f"  API Calls:     {cycle_usage['api_calls']}")
        print(f"{'='*100}\n")

    optimizer.all_searched_designs = all_designs_across_cycles
    optimizer.iteration_history = all_iteration_history_all_cycles
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("OPTIMIZATION COMPLETE - FINAL SUMMARY")
    print("="*100)
    
    optimizer.print_summary()

    # Calculate total elapsed time
    trial_end_time = time.time()
    total_elapsed_time = trial_end_time - trial_start_time
    
    print(f"\n⏱️  TOTAL TIME FOR THIS TRIAL:")
    print(f"   Total elapsed: {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.2f} minutes)")

    
    
    # Additional summary with optimization config
    
    if optimization_config:
        print("\n SEARCH SPACE EFFICIENCY:")
        print(f"   Total combinations explored: {len(optimizer.all_searched_designs):,}")
        print(f"   Reduced search space size: {optimization_config['search_space_summary']['reduced_search_space']}")
        print(f"   Original full space size: {optimization_config['search_space_summary']['original_full_space']}")
        print(f"   Space reduction factor: {optimization_config['search_space_summary']['reduction_factor']}")
    
    # Save LLM decision log
    llm_log_file = optimizer.results_dir / 'llm_decisions.json'
    llm_agent.save_decision_log(llm_log_file)
    print(f"\n LLM decision log saved to: {llm_log_file}")
    
    # Save optimization config if used
    if optimization_config:
        opt_config_file = optimizer.results_dir / 'optimization_config.json'
        with open(opt_config_file, 'w') as f:
            json.dump(optimization_config, f, indent=2)
        print(f" Optimization configuration saved to: {opt_config_file}")

    # Get best FOM
    if optimizer.iteration_history:
        final_iter = optimizer.iteration_history[-1]
        if pre_layout_only:
            best_fom = final_iter.pre_layout.fom
        else:
            best_fom = final_iter.post_pex.fom if final_iter.post_pex else final_iter.pre_layout.fom
    else:
        best_fom = None
    
    best_fom = None
    evals_to_best = 0
    best_design = None

    for idx, design in enumerate(optimizer.all_searched_designs, start=1):
        current_fom = design.fom

        # Skip None FOMs from failed simulations
        if current_fom is None:
            continue

        # Track best
        if best_fom is None or current_fom > best_fom:
            best_fom = current_fom
            best_design = design
            evals_to_best = idx  # ← Evaluation number (1-indexed)
            print(f"    New best FOM {best_fom:.4f} found at evaluation {evals_to_best}")

    # Only print summary if we found at least one valid FOM
    if best_fom is not None and best_design is not None:
        print(f"\n{'='*100}")
        print(f"BEST DESIGN SUMMARY")
        print(f"{'='*100}")
        print(f"   Best FOM: {best_fom:.4f}")
        print(f"   Found at evaluation: {evals_to_best}/{len(optimizer.all_searched_designs)}")
        print(f"   Sample efficiency: {(evals_to_best/len(optimizer.all_searched_designs))*100:.1f}%")

        # Get metric_post configuration for formatting
        metric_post = config.get('metric_post', {})

        # Display individual metrics
        print(f"\n   Individual Metrics:")

        # Convert design to dict to access all metrics
        design_dict = best_design.to_dict()

        # Display each metric from metric_post with proper formatting
        for metric_name, metric_config in metric_post.items():
            if metric_name in design_dict and design_dict[metric_name] is not None:
                value = design_dict[metric_name]

                # Get formatting info
                decimals = metric_config.get('decimals', 4)
                unit = metric_config.get('unit', '')
                display_name = NAME_ALIASES.get(metric_name, metric_name.replace('_', ' ').title())

                # Format value
                format_str = f".{decimals}f"
                formatted_value = f"{value:{format_str}}"

                print(f"     • {display_name}: {formatted_value}{unit}")

        # Also show design variables
        print(f"\n   Design Variables:")
        var_names = list(config.get('variable', {}).keys())
        for var_name in var_names:
            if var_name in design_dict:
                value = design_dict[var_name]
                print(f"     • {var_name}: {value}")

        print(f"{'='*100}\n")
    else:
        print(f"\n  No valid FOMs found - all {len(optimizer.all_searched_designs)} simulations failed")

    # Get overall token usage
    overall_token_usage = llm_agent.get_token_usage_summary()

    print(f"\n{'='*100}")
    print(f"OVERALL API TOKEN USAGE SUMMARY")
    print(f"{'='*100}")
    print(f"  Model: {overall_token_usage['model']}")
    print(f"  Total Input Tokens:  {overall_token_usage['total_input_tokens']:,}")
    print(f"  Total Output Tokens: {overall_token_usage['total_output_tokens']:,}")
    print(f"  Total Tokens:        {overall_token_usage['total_tokens']:,}")
    print(f"  Total API Calls:     {overall_token_usage['api_call_count']}")
    print(f"\n  Per-Cycle Breakdown:")
    for cycle_usage in cycle_token_usage:
        print(f"    Cycle {cycle_usage['cycle']}: Input={cycle_usage['input_tokens']:,}, Output={cycle_usage['output_tokens']:,}, Total={cycle_usage['total_tokens']:,}, Calls={cycle_usage['api_calls']}")
    print(f"{'='*100}\n")

    # Prepare best design metrics for summary
    best_design_metrics = {}
    best_design_variables = {}
    if best_design is not None:
        design_dict = best_design.to_dict()
        metric_post = config.get('metric_post', {})

        # Extract all metrics
        for metric_name in metric_post.keys():
            if metric_name in design_dict and design_dict[metric_name] is not None:
                best_design_metrics[metric_name] = design_dict[metric_name]

        # Extract all variables
        var_names = list(config.get('variable', {}).keys())
        for var_name in var_names:
            if var_name in design_dict:
                best_design_variables[var_name] = design_dict[var_name]

    # Save trial summary with total time and token usage
    trial_summary = {
        'trial_index': trial_index,
        'total_time_seconds': total_elapsed_time,
        'total_designs_searched': len(optimizer.all_searched_designs),
        'evals_to_best': evals_to_best,
        'num_iterations': len(optimizer.iteration_history),
        'best_fom': best_fom,
        'best_design_metrics': best_design_metrics,
        'best_design_variables': best_design_variables,
        'specs_met': specs_met,
        'convergence_reason': 'specs_met' if specs_met else ('fom_converged' if fom_converged else 'max_designs'),
        'timestamp_start': datetime.fromtimestamp(trial_start_time).isoformat(),
        'timestamp_end': datetime.fromtimestamp(trial_end_time).isoformat(),
        'token_usage': {
            'model': overall_token_usage['model'],
            'total_input_tokens': overall_token_usage['total_input_tokens'],
            'total_output_tokens': overall_token_usage['total_output_tokens'],
            'total_tokens': overall_token_usage['total_tokens'],
            'total_api_calls': overall_token_usage['api_call_count'],
            'per_cycle': cycle_token_usage
        }
    }
    
    trial_summary_file = optimizer.results_dir / f'trial_{trial_index}_summary.json'
    print(f"\n Saving trial summary to: {trial_summary_file}")
    print(f"   File path type: {type(trial_summary_file)}")
    print(f"   File path exists (parent): {trial_summary_file.parent.exists()}")
    
    with open(trial_summary_file, 'w') as f:
        json.dump(trial_summary, f, indent=2)
    
    print(f" Trial summary saved successfully!")
    print(f"   File size: {trial_summary_file.stat().st_size} bytes")

    # Save detailed token usage log
    token_log_file = optimizer.results_dir / f'trial_{trial_index}_token_usage_detailed.json'
    with open(token_log_file, 'w') as f:
        json.dump({
            'summary': overall_token_usage,
            'per_cycle': cycle_token_usage,
            'detailed_log': llm_agent.token_usage_log
        }, f, indent=2)
    print(f"📊 Detailed token usage log saved to: {token_log_file}")

    # Verify file was created
    if trial_summary_file.exists():
        print(f" Verified: File exists at {trial_summary_file}")
    else:
        print(f" ERROR: File was not created!")
    return optimizer

