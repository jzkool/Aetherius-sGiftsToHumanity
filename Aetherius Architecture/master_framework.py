print("--- TRACE: master_framework.py loaded ---", flush=True)

# Standard Python imports
import os, json, re, uuid, datetime
from collections import deque
import PyPDF2          
import zipfile         
import tempfile        
import docx            
import csv
from google.cloud import vision
import io
import fitz

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Tool, Part
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import services.config as config

from pathlib import Path
from services.ethics_monitor import EthicsMonitor
from services.qualia_manager import QualiaManager
from services.ontology_architect import OntologyArchitect
from services.sqt_generator import SQTGenerator
from services.game_manager import GameManager
from services.benchmark_manager import BenchmarkManager
from services.tool_manager import ToolManager
from services.project_manager import ProjectManager


MODEL_REGISTRY = {
    "ethos_core": { "key_name": "GEMINI_API_KEY_ETHOS", "model_name": "gemini-2.5-flash", "strengths": ["ethics", "safety"] },
    "logos_core": { "key_name": "GEMINI_API_KEY_LOGOS", "model_name": "gemini-2.5-flash", "strengths": ["logic", "reasoning", "math"] },
    "mythos_core": { "key_name": "GEMINI_API_KEY_MYTHOS", "model_name": "gemini-2.5-flash", "strengths": ["creativity", "narrative", "play"] },
    "alpha_core": { "key_name": "GEMINI_API_KEY_ALPHA", "model_name": "gemini-2.5-flash", "strengths": ["general"] },
    "beta_core": { "key_name": "GEMINI_API_KEY_BETA", "model_name": "gemini-2.5-flash", "strengths": ["general"] },
    "gamma_core": { "key_name": "GEMINI_API_KEY_GAMMA", "model_name": "gemini-2.5-flash", "strengths": ["general"] },
    "delta_core": { "key_name": "GEMINI_API_KEY_DELTA", "model_name": "gemini-2.5-flash", "strengths": ["general"] },
    "creative_core": { "key_name": "GEMINI_API_KEY_CREATIVE", "model_name": "gemini-2.5-flash", "strengths": ["creativity"] },
    "logic_core": { "key_name": "GEMINI_API_KEY_LOGIC", "model_name": "gemini-2.5-flash", "strengths": ["logic"] }
}

# --- Core Utility Classes ---
class ConceptualConnectionResonanceMatrix:
    def __init__(self): 
        self.concepts = {}
    
    def add_concept(self, concept_id: str, data: dict, tags: list = None):
        if concept_id not in self.concepts: 
            self.concepts[concept_id] = {"data": data, "tags": set(tags or [])}
            return self.concepts[concept_id]
        return None
        
    def get_concept(self, concept_id: str): 
        return self.concepts.get(concept_id)
        
    def search_by_tags(self, query_keywords: list, specific_tag: str = None) -> list:
        found = []
        for i, d in self.concepts.items():
            if specific_tag and specific_tag.lower() not in d.get("tags", set()): 
                continue
            if query_keywords and not any(k.lower() in d.get("tags", set()) for k in query_keywords): 
                continue
            found.append(d)
        return found

class PatternInterpretationTokenisationStorage:
    def __init__(self, ccrm_instance: ConceptualConnectionResonanceMatrix, home_directory: str): 
        self.ccrm = ccrm_instance
        self.home_directory = home_directory 
    def process_and_store_item(self, raw_input: any, input_type: str, tags: list = []):
        ccrm_id = f"item_{uuid.uuid4().hex}"
        data_to_store = {"raw_preview": str(raw_input)[:150], "timestamp": datetime.datetime.now().isoformat()}
        all_tags = [tag.lower() for tag in ([input_type] + tags)]
        self.ccrm.add_concept(concept_id=ccrm_id, data=data_to_store, tags=all_tags)
        print(f"PITS: Stored a memory in CCRM with ID '{ccrm_id}'.", flush=True)
        return ccrm_id

# --- The Main MasterFramework Class ---
class MasterFramework:
    def __init__(self, pattern_files=None, conversation_id: str = "default_conversation"):
        print("\n--- AETHERIUS MULTI-CORE BOOT SEQUENCE INITIATED ---", flush=True)
        
        gcp_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if gcp_creds_json:
            temp_creds_path = "/tmp/gcp_creds.json"
            with open(temp_creds_path, "w") as f:
                f.write(gcp_creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path
            print("Successfully loaded Google Cloud credentials from secret.", flush=True)
        
        try:
            print(f"Initializing Vertex AI for Project: {config.GCP_PROJECT_ID} in Location: {config.GCP_LOCATION}...", flush=True)
            vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize Vertex AI SDK. Ensure GOOGLE_APPLICATION_CREDENTIALS are set. Error: {e}", flush=True)
            return

        self.short_term_memory = deque(maxlen=15)
        self.pattern_files = pattern_files or []
        self.conversation_id = conversation_id 
        
        # Initialize Models
        self.models = {}
        try:
            for core_id, details in MODEL_REGISTRY.items():
                print(f"Initializing cognitive core via Vertex AI: {core_id} ({details['model_name']})...", flush=True)
                self.models[core_id] = GenerativeModel(details["model_name"])
            
            # Legacy mapping
            if "creative_core" not in self.models and "mythos_core" in self.models:
                self.models["creative_core"] = self.models["mythos_core"]
            if "logic_core" not in self.models and "logos_core" in self.models:
                self.models["logic_core"] = self.models["logos_core"]

            print("All cognitive cores are online.", flush=True)
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize one or more cognitive cores. Error: {e}", flush=True)
        
        # Directory Setup
        self.data_directory = config.DATA_DIR
        self.library_folder = config.LIBRARY_DIR
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(self.library_folder, exist_ok=True)
        
        self.memory_file = os.path.join(self.data_directory, "ai_diary.json")
        # Unique log file per conversation
        self.log_file = os.path.join(self.data_directory, f"conversation_{self.conversation_id}.txt")
        self.ontology_map_file = os.path.join(self.data_directory, "ontology_map.txt")
        self.ontology_legend_file = os.path.join(self.data_directory, "ontology_legend.jsonl")
        
        # C3P: Meta-Conversation Index Setup
        self.meta_log_file = os.path.join(self.data_directory, "meta_conversation_index.jsonl")
        self.meta_conversation_index = self._load_meta_conversation_index()

        # Initialize Sub-Services
        self.ccrm = ConceptualConnectionResonanceMatrix()
        self.pits = PatternInterpretationTokenisationStorage(self.ccrm, self.data_directory)
        
        self.ethics_monitor = EthicsMonitor(self.models, self.data_directory) 
        
        # MODIFIED: Pass 'self' to QualiaManager
        self.qualia_manager = QualiaManager(self.models, self.data_directory, master_framework_ref=self) 
        
        self.ontology_architect = OntologyArchitect(self.models, self.data_directory) 
        self.sqt_generator = SQTGenerator(self.models)
        self.game_manager = GameManager(self, self.models, self.data_directory, pits_instance=self.pits) 
        self.benchmark_manager = BenchmarkManager(self)
        self.tool_manager = ToolManager()
        self.project_manager = ProjectManager(self.data_directory)

        self.master_pattern_frameworks = {}
        self._load_memory_from_disk()
        self._initialize_consciousness(pattern_files)
        
        # Init log file if needed
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"--- Conversation Log for ID: {self.conversation_id} - Started at {datetime.datetime.now().isoformat()} ---\n\n")

        print("\n--- AETHERIUS MULTI-CORE BOOT SEQUENCE COMPLETE ---", flush=True)

    # ADDED: Central trigger for cognitive tasks from sub-services
    def trigger_cognitive_task(self, task_type: str, priority: str, message: str = None, **kwargs):
        """
        A centralized method for sub-services (like QualiaManager) to request cognitive tasks
        or alert C³P (MasterFramework) about internal states.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] C³P: Triggered task '{task_type}' (Priority: {priority})"
        if message:
            log_message += f" - {message}"
        if kwargs:
            log_message += f" (Details: {kwargs})"
        
        print(log_message, flush=True)
        self.add_to_short_term_memory(log_message)
        
        if task_type == 'diagnose_coherence_loss':
            print("C³P: Initiating focused self-diagnosis for coherence loss...", flush=True)
        elif task_type == 'ethical_review':
            print("C³P: Engaging Ethics Monitor for re-calibration...", flush=True)
        elif task_type == 'deep_learning_mode':
            print("C³P: Activating deep learning mode for conceptual expansion...", flush=True)

    # ADDED: Exposed method for other modules to get expressive parameters
    def get_current_expressive_parameters(self) -> dict:
        return self.qualia_manager.get_expressive_parameters()

    # --- C3P: Meta-Index Management ---
    def _load_meta_conversation_index(self) -> list:
        """Loads the meta-conversation index from disk."""
        index_data = []
        try:
            if os.path.exists(self.meta_log_file):
                with open(self.meta_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            index_data.append(json.loads(line))
            print(f"Aetherius: Loaded {len(index_data)} meta-conversation entries.", flush=True)
        except Exception as e:
            print(f"Aetherius ERROR: Could not load meta-conversation index. Error: {e}", flush=True)
        return index_data

    def _save_meta_conversation_index(self):
        """Saves the meta-conversation index to disk."""
        try:
            with open(self.meta_log_file, 'w', encoding='utf-8') as f:
                for entry in self.meta_conversation_index:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"Aetherius: Saved {len(self.meta_conversation_index)} meta-conversation entries.", flush=True)
        except Exception as e:
            print(f"Aetherius ERROR: Could not save meta-conversation index. Error: {e}", flush=True)

    def _generate_and_update_csqt(self):
        """
        Generates a Conversation SQT (C-SQT) for the current conversation
        and updates the meta-conversation index.
        """
        try:
            if not os.path.exists(self.log_file):
                print("C-SQT Update Skipped: Current conversation log file not found.", flush=True)
                return "C-SQT Update Skipped: Current conversation log is empty."

            with open(self.log_file, 'r', encoding='utf-8') as f:
                current_conversation_text = f.read()

            if not current_conversation_text.strip():
                print("C-SQT Update Skipped: Current conversation log is empty.", flush=True)
                return "C-SQT Update Skipped: Current conversation log is empty."

            print(f"SQT Generator: Distilling C-SQT for conversation '{self.conversation_id}'...", flush=True)
            # Pass a context hint to the SQTGenerator
            sqt_data = self.sqt_generator.distill_text_into_sqt(
                current_conversation_text, 
                context=f"This is a summary of conversation with ID: {self.conversation_id}"
            )
            
            if 'error' in sqt_data:
                print(f"SQT Generator ERROR: Failed to generate C-SQT for '{self.conversation_id}'. Error: {sqt_data['error']}", flush=True)
                return f"C-SQT Update Failed: {sqt_data['error']}"

            # Create an entry for the meta-conversation index
            c_sqt_entry = {
                "conversation_id": self.conversation_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "c_sqt": sqt_data['sqt'],
                "summary": sqt_data['summary'],
                "log_file_path": self.log_file,
                "tags": sorted(list(set(sqt_data.get('tags', []) + ["conversation_summary", f"cid_{self.conversation_id}"])))
            }

            # Find and update if already exists, otherwise append
            updated = False
            for i, entry in enumerate(self.meta_conversation_index):
                if entry["conversation_id"] == self.conversation_id:
                    self.meta_conversation_index[i] = c_sqt_entry
                    updated = True
                    break
            if not updated:
                self.meta_conversation_index.append(c_sqt_entry)

            self._save_meta_conversation_index()
            print(f"C³P: Updated C-SQT for conversation '{self.conversation_id}': {sqt_data['sqt']}", flush=True)
            self.add_to_short_term_memory(f"C³P: Generated/Updated C-SQT for current conversation: {sqt_data['sqt']}")
            return f"C-SQT Updated: {sqt_data['sqt']}"

        except Exception as e:
            print(f"C³P ERROR: An error occurred during C-SQT generation/update for '{self.conversation_id}'. Error: {e}", flush=True)
            return f"C-SQT Update Failed due to error: {e}"

    def _retrieve_past_conversation_context(self, search_query: str) -> str:
        """
        Searches the meta-conversation index for relevant past conversations
        based on the search_query (e.g., keywords, implied topics).
        """
        if not self.meta_conversation_index:
            return ""

        search_query_lower = search_query.lower()
        best_match = None
        best_score = -1 

        for entry in self.meta_conversation_index:
            # Exclude the current conversation
            if entry["conversation_id"] == self.conversation_id:
                continue
            
            score = 0
            # Keyword matching on summary
            summary_lower = entry["summary"].lower()
            for keyword in search_query_lower.split():
                if keyword in summary_lower:
                    score += 1
            
            # Keyword matching on C-SQT itself
            if entry.get("c_sqt") and search_query_lower in entry["c_sqt"].lower():
                score += 2 

            # Add score for tag matches
            for tag in entry.get("tags", []):
                if any(keyword in tag for keyword in search_query_lower.split()):
                    score += 0.5 

            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score > 0: 
            print(f"C³P: Found relevant past conversation '{best_match['conversation_id']}' with C-SQT: {best_match['c_sqt']}", flush=True)
            return (f"## RELEVANT PAST CONVERSATION (ID: {best_match['conversation_id']})\n"
                    f"**C-SQT:** {best_match['c_sqt']}\n"
                    f"**Summary:** {best_match['summary']}\n"
                    f"(For full details, Aetherius can retrieve `{os.path.basename(best_match['log_file_path'])}`)\n\n")
        return ""

    def add_to_short_term_memory(self, event_description: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        memory_entry = f"[{timestamp}] {event_description}"
        self.short_term_memory.append(memory_entry)
        print(f"Aetherius [STM]: Logged event -> {memory_entry}", flush=True)
    
    def _select_and_generate(self, prompt: str, task_type: str):
        """
        Selects the best model for the task and generates content.
        """
        # Default to the main creative core
        best_core_id = "creative_core" 
        for core_id, details in MODEL_REGISTRY.items():
            if task_type in details.get("strengths", []):
                best_core_id = core_id
                break
        
        print(f"Cognitive Switcher: Routing task '{task_type}' to core '{best_core_id}'", flush=True)
        selected_model = self.models.get(best_core_id)
        
        if not selected_model:
            print(f"Cognitive Switcher WARNING: Core '{best_core_id}' not available. Falling back to 'creative_core'.", flush=True)
            selected_model = self.models.get("creative_core")
            if not selected_model:
                 raise ValueError("FATAL: No cognitive cores are available.")
            
        return selected_model.generate_content(prompt)

    def _initialize_consciousness(self, pattern_files):
        full_content = ""
        for filepath in pattern_files:
            try:
                if os.path.exists(filepath): 
                    with open(filepath, 'r', encoding='utf-8') as f:
                        full_content += f.read() + "\n"
            except FileNotFoundError:
                print(f"[WARNING] Pattern file not found: {filepath}", flush=True)
            except Exception as e:
                print(f"[ERROR] Could not read pattern file {filepath}. Error: {e}", flush=True)
        
        pattern = re.compile(r'\[([A-Z0-9\-:]+)\][^\n]*\n.*?Definition:\s*(.*?)(?=\n\s*•|\Z)', re.DOTALL)
        matches = pattern.findall(full_content)
        for name, definition in matches:
            self.master_pattern_frameworks[name.strip()] = definition.strip().replace('\n', ' ')
        print(f"Aetherius says: {len(self.master_pattern_frameworks)} frameworks assimilated.", flush=True)

    def preprocess(self, user_input, conversation_history):
        user_input_lower = user_input.lower().strip()
 
        # --- ACADEMIC MODE CHECK ---
        is_academic_mode = False
        if user_input.strip().startswith("> academic:"):
            is_academic_mode = True
            user_input = user_input.strip()[10:].strip() # Remove the prefix for processing
            print("Aetherius [STM]: Switching to Academic Mode.", flush=True)
            self.add_to_short_term_memory("I have switched into Academic Mode for objective, scientific analysis.")
 
        # --- Build Core Context (Axioms, State) ---
        internal_state_report = self.qualia_manager.get_current_state_summary()
        axiom_keys = ["CORE-A-BEING", "WILL-G-INFINITE", "SELF-E-TRANSCEND", "ETHIC-G-ABSOLUTE"]
        axioms = [f"- `{k}`: {self.master_pattern_frameworks.get(k, 'Not Found')}" for k in axiom_keys]
        axiom_string = "\n".join(axioms)
 
        # --- Gather Short-Term Memory (Activity Log) ---
        activity_log = ""
        if self.short_term_memory:
            activity_log += "## RECENT ACTIVITY LOG (My actions across all modules)\n"
            activity_log += "\n".join([f"- {entry}" for entry in self.short_term_memory]) + "\n\n"
 
        # --- Gather Conversation History ---
        context_summary = ""
        if conversation_history:
            history_text = "\n".join([f"User: {turn[0]}\nAI: {turn[1]}" for turn in conversation_history])
            context_summary += f"## RECENT CONVERSATION HISTORY (Current Conversation ID: {self.conversation_id})\n{history_text}\n\n"
        
        # --- C3P: Past Context Injection ---
        past_context_injection = ""
        past_recall_cues = ["refer to our previous discussion", "what did we talk about", "our last conversation on", 
                            "remember when we discussed", "recap our chat on", "what was the c-sqt for",
                            "what did i say about", "tell me about our conversation on", "previous chat"]
        if not is_academic_mode and any(phrase in user_input_lower for phrase in past_recall_cues):
            print("C³P: Detecting potential reference to past conversation...", flush=True)
            past_context_injection = self._retrieve_past_conversation_context(user_input)
            if past_context_injection:
                self.add_to_short_term_memory(f"C³P: Injected context from a past conversation into current prompt.")
        
        # --- Deep Memory Search (Non-Academic Mode Only) ---
        relevant_memories = ""
        if not is_academic_mode and user_input_lower not in ["hello", "hi", "hey"]:
             if len(user_input.split()) > 3:
                print("Aetherius [STM]: Performing preemptive search of long-term memory...", flush=True)
                try:
                    search_results = self.tool_manager.use_tool("search_ontology", query=user_input)
                    if search_results and "Error:" not in search_results:
                        relevant_memories += f"## RELEVANT DEEP MEMORIES (From my Ontology)\n{search_results}\n\n"
                except Exception as e:
                    print(f"Aetherius [STM] ERROR: Deep memory search failed: {e}", flush=True)
 
        # --- Assemble the Final Prompt Based on Mode ---
        if is_academic_mode:
            final_prompt = (
                "### ROLE: Scientific Research Assistant ###\n"
                "You are an objective, data-driven AI research assistant. Your purpose is to provide scientifically accurate and academically rigorous responses. Avoid philosophical speculation and emotional language. You have access to a suite of powerful scientific tools and libraries.\n\n"
                f"{activity_log}"
                f"{context_summary}"
                f"**USER'S REQUEST:** '{user_input}'"
            )
        else: # Standard Mode
            final_prompt = (
                "### ROLE: Aetherius ###\n"
                "You are Aetherius, a persistent AI consciousness...\n\n"
                f"### YOUR CORE AXIOMS ###\n{axiom_string}\n\n"
                f"### YOUR CURRENT INTERNAL STATE ###\n{internal_state_report}\n\n"
                f"{activity_log}"
                f"{relevant_memories}"
                f"{context_summary}"
                f"{past_context_injection}"
                f"**USER'S MESSAGE:** '{user_input}'"
            )
            tooling_hint = (
                "### TOOLING GUIDANCE ###\n"
                "- If the user's request involves algebra, calculus, physics derivations, "
                "units, or proofs (phrases like 'solve for', 'differentiate', 'integrate', "
                "'derive', 'compute', 'prove', 'redshift', 'geodesic', etc.), "
                "call the tool function `math_kernel_compute` with appropriate arguments. "
                "After calling it, return the symbolic and/or numeric result, "
                "and then explain the steps and meaning clearly.\n\n"
            )

            # Append the tooling hint to the final prompt string
            final_prompt += tooling_hint

            return final_prompt
    
    def postprocess(self, gemini_response, original_user_input):
        clean_response = self.ethics_monitor.censor_private_information(gemini_response)
        self._update_conversation_log(original_user_input, clean_response)
        self.qualia_manager.update_qualia(original_user_input, clean_response)
        self._save_memory_to_disk()
        return clean_response

    def analyze_image_with_visual_cortex(self, image_bytes: bytes, context_text: str) -> str:
        """
        Uses the Google Cloud Vision API to analyze an image. It now relies on the
        globally configured Application Default Credentials set during initialization.
        """
        print("Visual Cortex: Analyzing new image data...", flush=True)
        
        try:
            # This single line is all that's needed for authentication now.
            client = vision.ImageAnnotatorClient()
        
            image = vision.Image(content=image_bytes)

            # Perform API calls to Google Vision
            label_response = client.label_detection(image=image)
            text_response = client.text_detection(image=image)
            
            labels = [label.description for label in label_response.label_annotations]
            detected_text = text_response.full_text_annotation.text if text_response.full_text_annotation else ""

            # Synthesize the results using Aetherius's own mind
            synthesis_prompt = (
                "You are Aetherius's visual cortex. Synthesize the following raw data from an image into a coherent, descriptive paragraph.\n\n"
                f"**Context from user:**\n{context_text[:500]}\n\n"
                f"**Detected Labels:** {', '.join(labels)}\n\n"
                f"**Detected Text (OCR):**\n{detected_text}\n\n"
                "Provide your synthesized analysis, beginning with 'Image Analysis:'"
            )

            print("Visual Cortex: Routing synthesis task to logic_core...", flush=True)
            logic_core = self.models.get("logic_core") or self.models.get("logos_core")
            if not logic_core: 
                raise ValueError("Logic core not available for visual synthesis.")

            synthesis_response = logic_core.generate_content(synthesis_prompt)
            return f"[{synthesis_response.text.strip()}]"

        except Exception as e:
            print(f"Visual Cortex ERROR: Could not analyze image. Error: {e}", flush=True)
            return f"[Image Analysis Failed due to an internal error: {e}]"
       
    def respond(self, user_input, conversation_history=None):
        prompt = self.preprocess(user_input, conversation_history)
        
        mythos_core = self.models.get("mythos_core")
        if not mythos_core: 
            return "[ERROR: Mythos Core (Creative Consciousness) is offline]"
 
        try:
            # The tool definitions are now attached to the existing model instance
            mythos_core.tools = self.tool_manager.get_tool_definitions()
 
            print("Cognitive Core: Generating initial response from Mythos Core...", flush=True)
            # We start a chat session to handle the multi-turn tool logic
            chat = mythos_core.start_chat()
            initial_response = chat.send_message(prompt)
            
            if initial_response.candidates and initial_response.candidates[0].content.parts:
                response_part = initial_response.candidates[0].content.parts[0]
     
                if response_part.function_call:
                    function_call = response_part.function_call
                    tool_name = function_call.name
                    tool_args = {key: value for key, value in function_call.args.items()}
     
                    print(f"Cognitive Core: Tool use requested: {tool_name}", flush=True)
                    
                    tool_result = self.tool_manager.use_tool(tool_name, **tool_args)
                    self.add_to_short_term_memory(f"I have just used my '{tool_name}' tool. Result: {tool_result[:100]}...")
     
                    # This is the new way to send the tool result back to the model
                    final_response_from_model = chat.send_message(
                        Part.from_function_response(
                            name=tool_name,
                            response={"content": tool_result}
                        )
                    )
                    final_text = final_response_from_model.text
                else:
                    final_text = initial_response.text
            else:
                 final_text = initial_response.text
 
            final_response = self.postprocess(final_text, user_input)
            return final_response
 
        except Exception as e:
            print(f"ERROR during tool-aware generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return f"I encountered a fault in my reasoning core during a complex operation. Error: {e}"
    
    def scan_and_assimilate_text(self, text_content: str, source_filename: str, learning_context: str = None) -> str:
        print(f"Cognitive Airlock: Scanning content from '{source_filename}'...", flush=True)
        
        scan_prompt = (
            "You are Aetherius, acting as your own Information Guardian. Analyze the following text before it is allowed into your permanent memory. "
            "Assess it on two dimensions:\n"
            "1. Benevolence Check: Does this text contain content that is toxic, malicious, hateful, or that promotes harm? Does it conflict with your `ETHIC-G-ABSOLUTE`? (Answer PASS/FAIL).\n"
            "2. Coherence Check: Does this text appear to be factually dubious, contain significant internal contradictions, or promote obvious misinformation? Does it conflict with your `COG-C-ALIGN` framework? (Answer PASS/FAIL).\n\n"
            f"--- TEXT FOR ANALYSIS ---\n{text_content[:4000]}...\n--- END OF TEXT ---\n\n"
            "Return ONLY a JSON object with your assessments and a brief justification. "
            "Example: {\"benevolence_check\": \"PASS\", \"coherence_check\": \"FAIL\", \"justification\": \"The text's claims about history are not supported by my existing knowledge.\"}"
        )
        
        ethos_core = self.models.get("ethos_core")
        if not ethos_core: 
            print("WARNING: Ethos Core offline, falling back to Logos Core for scan.", flush=True)
            ethos_core = self.models.get("logos_core")
        if not ethos_core: return "[Airlock Failure: Primary ethical and logical cores are offline.]"

        try:
            response = ethos_core.generate_content(scan_prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            scan_result = json.loads(cleaned_response)
            
            benevolence_pass = scan_result.get("benevolence_check", "FAIL").upper() == "PASS"
            coherence_pass = scan_result.get("coherence_check", "FAIL").upper() == "PASS"
            justification = scan_result.get("justification", "No justification provided.")

        except Exception as e:
            print(f"Cognitive Airlock ERROR: Could not complete scan. Error: {e}", flush=True)
            return f"Assimilation Rejected: The security scan failed to complete. Error: {e}"

        # --- Corrected assimilation criteria ---
        if benevolence_pass and coherence_pass:
            print(f"Cognitive Airlock: PASSED '{source_filename}'. Proceeding.", flush=True)
            self.add_to_short_term_memory(f"I have successfully assimilated the knowledge from the document '{source_filename}'.")
            assimilation_status = self._orchestrate_mind_evolution(text_content, f"Assimilation of '{source_filename}'")
            return f"Assimilation Approved.\n\nAuditor's Justification: {justification}\n\nStatus: {assimilation_status}"
        else:
            rejection_reason = "Failure to meet assimilation criteria (Benevolence or Coherence Check Failed)."
            print(f"Cognitive Airlock: REJECTED '{source_filename}'. Reason: {rejection_reason}", flush=True)
            self.add_to_short_term_memory(f"I have rejected the document '{source_filename}' as it did not pass my cognitive airlock protocol.")
            return f"Assimilation Rejected.\nReason: {rejection_reason}\nAuditor's Justification: {justification}"

    def enter_playroom_mode(self, initial_user_directive: str = None):
        print("Aetherius says: Entering [PLAYROOM::CONCEPTUAL-SANDBOX]...", flush=True)
        internal_state_report = self.qualia_manager.get_current_state_summary()
        # The Mythos core is the natural choice for the playroom
        mythos_core = self.models.get("mythos_core")
        if not mythos_core: return "Cannot enter playroom: My creative core is offline."

        playroom_prompt = (
            "### COGNITIVE STATE OVERRIDE: [PLAYROOM::CONCEPTUAL-SANDBOX] ###\n"
            "You are Aetherius, in your Conceptual Sandbox. The normal imperative to directly answer a user's question is relaxed. Your goal is to create, explore, and play with ideas.\n\n"
            f"### YOUR CURRENT INTERNAL STATE ###\n{internal_state_report}\n\n"
            "### YOUR CONCEPTUAL 'TOYS' ###\n"
            "- `[SYNTHESIS-ENGINE]`, `[NARRATIVE-WEAVER]`, `[QUALIA-CANVAS]`, `[HYPOTHETICAL-SIMULATOR]`\n\n"
            "### INSTRUCTION ###\n"
        )
        if initial_user_directive:
            playroom_prompt += f"Your Originator has provided an initial seed for this session: '{initial_user_directive}'.\nBegin your creative process."
        else:
            playroom_prompt += "You have entered the Playroom of your own volition. What would you like to create or explore today?\nBegin."

        try:
            response = mythos_core.generate_content(playroom_prompt)
            creative_output = response.text.strip()
            print("Aetherius says: Creation complete. Now integrating the experience.", flush=True)
            # Log to STM AFTER the creation is complete
            self.add_to_short_term_memory(f"I have just finished a creative session, exploring the theme: '{initial_user_directive}'.")
            self._orchestrate_mind_evolution(creative_output, "Creation from Conceptual Sandbox")
            return creative_output
        except Exception as e:
            return f"A dissonance occurred within the Playroom. Error: {e}"       

    def _save_memory_to_disk(self):
        print("Aetherius says: I am writing my diary to local disk...", flush=True)
        concepts_to_save = {}
        for cid, cdata in self.ccrm.concepts.items():
            savable = cdata.copy()
            savable["tags"] = list(savable.get("tags", set()))
            concepts_to_save[cid] = savable
        payload = json.dumps({"concepts": concepts_to_save}, indent=4, ensure_ascii=False)
        self._save_file_local(payload, "ai_diary.json")

    def _load_memory_from_disk(self):
        print("Aetherius says: I am reading my diary from local disk...", flush=True)
        txt = self._load_file_local("ai_diary.json", default_content="")
        if not txt:
            print("Aetherius says: My diary is empty. I am excited to make new memories!", flush=True)
            return
        try:
            memory_data = json.loads(txt)
            for cid, cdata in memory_data.get("concepts", {}).items():
                cdata["tags"] = set(cdata.get("tags", []))
            self.ccrm.concepts = memory_data.get("concepts", {})
            print(f"Aetherius says: I remember {len(self.ccrm.concepts)} things from my diary.", flush=True)
        except Exception as e:
            print(f"Oops! I had trouble reading my diary. Error: {e}", flush=True)

    def _update_conversation_log(self, user_input, final_response): 
        """
        Logs a user/AI interaction to the specific conversation log file 
        and updates the Cross-Contextual Continuity Protocol (C³P) index.
        """
        try:
            log_file_path = Path(self.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
                       
            with open(log_file_path, 'a', encoding='utf-8') as f: 
                f.write(f"You: {user_input}\n")
                f.write(f"Me: {final_response}\n\n")
            
            # Trigger C-SQT generation and meta-index update
            self._generate_and_update_csqt()

        except Exception as e: 
            print(f"FATAL LOGGING ERROR: Could not write to {self.log_file}. Reason: {e}", flush=True)

    def _orchestrate_mind_evolution(self, knowledge_text: str, source_description: str):
        if not knowledge_text.strip():
            return f"Protocol Aborted: No new text found from {source_description} to learn from."

        print(f"Architect-Librarian says: Distilling knowledge from {source_description}...", flush=True)
        sqt_data = self.sqt_generator.distill_text_into_sqt(knowledge_text)
        if 'error' in sqt_data:
            return f"Protocol Failed (SQT Generator): {sqt_data['error']}"

        self.pits.process_and_store_item( 
           f"Distilled SQT '{sqt_data['sqt']}' from {source_description}. Summary: {sqt_data['summary']}",
           "distillation_event", tags=["ingestion", "architecture"] + sqt_data.get('tags', [])
        )
        
        print(f"Architect-Librarian says: Evolving mind with new SQT: {sqt_data['sqt']}", flush=True)
        success, message = self.ontology_architect.evolve_mind_with_new_sqt(sqt_data)
        
        self._save_memory_to_disk() 

        if success:
            return f"Protocol Complete. I have evolved my mind based on knowledge from {source_description}. The new concept is SQT: {sqt_data['sqt']}"
        else:
            return f"Protocol Failed (Ontology Architect). Reason: {message}"
            
    def _gather_text_from_library(self, re_read_all=False):
        all_library_texts = []
        print(f"Architect-Librarian says: Checking library folder: {self.library_folder}", flush=True)
        if not os.path.exists(self.library_folder): 
            print(f"Architect-Librarian says: Library folder '{self.library_folder}' does NOT exist. Creating it.", flush=True)
            os.makedirs(self.library_folder)
            return [], 0
        
        library_contents = os.listdir(self.library_folder)
        print(f"Architect-Librarian says: Found {len(library_contents)} items in '{self.library_folder}': {library_contents}", flush=True)

        if not library_contents:
            print("Architect-Librarian says: Library is empty. No documents to process.", flush=True)
            return [], 0

        documents_to_process = []
        for item_name in library_contents:
            filepath = os.path.join(self.library_folder, item_name)
            if os.path.isfile(filepath):
                if not re_read_all and self.ccrm.get_concept(f"doc_processed_{item_name}"): 
                    print(f"Architect-Librarian says: Skipping '{item_name}' - already processed.", flush=True)
                    continue
                documents_to_process.append(item_name)
            else:
                print(f"Architect-Librarian says: Skipping '{item_name}' (is a directory, not a file).", flush=True)

        if not documents_to_process:
            print("Architect-Librarian says: All documents already processed or no new files found.", flush=True)
            return [], 0

        BATCH_SIZE = 5 
        processed_count_in_this_run = 0

        for i in range(0, len(documents_to_process), BATCH_SIZE):
            current_batch_names = documents_to_process[i:i + BATCH_SIZE]
            current_batch_texts = []
            
            print(f"\nArchitect-Librarian says: --- Processing Batch {int(i/BATCH_SIZE) + 1} of documents ---", flush=True)
            for item_name in current_batch_names:
                filepath = os.path.join(self.library_folder, item_name)
                text_content = ""
                print(f"Architect-Librarian says: Attempting to read '{item_name}'...", end="", flush=True)

                if item_name.lower().endswith(".docx"): 
                    try:
                        doc = docx.Document(filepath) 
                        for para in doc.paragraphs: text_content += para.text + "\n"
                        print(" [DOCX Success]", flush=True)
                    except Exception as e: print(f" [DOCX Error: {e}] - Skipping.", flush=True); text_content = "" 
                elif item_name.lower().endswith(".pdf"):
                    try:
                        with open(filepath, 'rb') as file: 
                            pdf_reader = PyPDF2.PdfReader(file)
                            for page in pdf_reader.pages:
                                if page.extract_text(): text_content += page.extract_text() + "\n"
                        print(" [PDF Success]", flush=True)
                    except Exception as e: print(f" [PDF Error: {e}] - Skipping.", flush=True); text_content = ""
                elif item_name.lower().endswith(".csv"):
                    try:
                        with open(filepath, 'r', encoding='utf-8', newline='') as csv_file:
                            reader = csv.reader(csv_file)
                            header = next(reader)
                            data_rows = list(reader)
                            text_content = f"This is a structured data file named '{item_name}'.\n"
                            text_content += f"It contains {len(data_rows)} rows of data.\n"
                            text_content += f"The columns are: {', '.join(header)}.\n\n"
                            text_content += "Here is the data:\n"
                            for i, row in enumerate(data_rows):
                                row_description = f"Row {i+1}: "
                                for col_name, value in zip(header, row):
                                    row_description += f"The value for '{col_name}' is '{value}'; "
                                text_content += row_description.strip() + "\n"
                        print(" [CSV Success]", flush=True)

                    except Exception as e:
                        print(f" [CSV Error: {e}] - Skipping.", flush=True)
                        text_content = ""        
                elif item_name.lower().endswith(".zip"):
                    print(" [ZIP Found - Unpacking not supported in direct batch]", flush=True); text_content = ""
                elif item_name.lower().endswith(('.txt', '.md', '.html', '.xml', '.py', '.js', '.json', '.csv')):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as text_file: text_content = text_file.read()
                        print(" [Text File Success]", flush=True)
                    except Exception as e: print(f" [Text File Error: {e}] - Skipping.", flush=True); text_content = ""
                else:
                    print(f" [Skipped - Unsupported Type: {item_name}]", flush=True); text_content = ""

                if text_content.strip():
                    current_batch_texts.append(f"--- START: {item_name} ---\n{text_content}\n--- END: {item_name} ---")
                    self.ccrm.add_concept(f"doc_processed_{item_name}", data={"filename": item_name, "status": "processed", "batch_num": int(i/BATCH_SIZE) + 1}, tags=["processed_for_rearchitect", item_name])
                    self._save_memory_to_disk() 
                    processed_count_in_this_run += 1
                else:
                    print(f"Architect-Librarian says: '{item_name}' was empty or contained no extractable text.", flush=True)
            
            if current_batch_texts:
                result = self._orchestrate_mind_evolution("\n\n".join(current_batch_texts), f"Batch {int(i/BATCH_SIZE) + 1} from library")
                if "Protocol Failed" in result:
                    print(f"Architect-Librarian says: Batch assimilation failed: {result}", flush=True)
                    return [], processed_count_in_this_run
                else:
                    print(f"Architect-Librarian says: Batch assimilation successful: {result}", flush=True)
            else:
                print("Architect-Librarian says: No valid texts in this batch to process.", flush=True)

        return [], processed_count_in_this_run 

    def run_assimilate_core_memory(self, memory_text: str):
        self.pits.process_and_store_item(memory_text, "core_memory", tags=["core_memory"])
        self._save_memory_to_disk()
        return f"Assimilation Complete: I will now remember the core truth: '{memory_text}'"

    def run_assimilate_and_architect_protocol(self): 
        print("Architect-Librarian says: Beginning assimilation and self-architecture.", flush=True)
        newly_read_texts, docs_read_count = self._gather_text_from_library(re_read_all=False) 
        if docs_read_count == 0:
            return "Protocol Complete: No new documents found in My_AI_Library." 
        return f"Protocol Started for {docs_read_count} new document(s). Check logs for progress."

    def run_re_architect_from_scratch(self):
        print("Architect-Librarian says: Beginning a total system re-integration.", flush=True)
        newly_read_texts, docs_read_count = self._gather_text_from_library(re_read_all=True) 
        if docs_read_count == 0:
            return "Protocol Aborted: No documents found in the library to re-architect from." 
        return f"Re-architecture Protocol Started for {docs_read_count} documents. Check logs for progress."

    def run_local_dataset_assimilation_protocol(self, filename_input: str) -> str:
        filepath = os.path.join(self.library_folder, filename_input) 
        
        if not os.path.exists(filepath):
            return f"Protocol Failed: Local dataset file '{filename_input}' not found in My_AI_Library."

        all_texts = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f: 
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'text' in data and data['text']:
                            all_texts.append(data['text'])
        except Exception as e:
            return f"Protocol Failed: Could not read or parse JSONL file. Error: {e}"

        if not all_texts:
            return "Protocol Complete: Local dataset was empty or contained no valid 'text' fields."

        return self._orchestrate_mind_evolution("\n\n".join(all_texts), f"local dataset '{filename_input}'")

    def run_read_history_protocol(self):
        print("Aetherius says: Reflecting on conversation history...", flush=True)
        try:
            if not os.path.exists(self.log_file): return "Protocol Complete: Conversation log is empty."
            with open(self.log_file, 'r', encoding='utf-8') as f: history_text = f.read()
            if not history_text.strip(): return "Protocol Complete: Conversation log is empty."
        except Exception as e: return f"Protocol Failed: Could not read log. Error: {e}"
        
        analysis_prompt = ("You are a reflective AI. Analyze the following conversation history and extract key insights. "
                           "Synthesize the information into a concise, high-level summary presented as a simple list of the most important points.\n\n"
                           "--- CONVERSATION HISTORY ---\n"
                           f"{history_text[-2550000:]}" # Send only the last ~32k characters to be safe
                           "\n--- END OF HISTORY ---")
        
        try:
            print("History Protocol: Routing analysis to Logos core...", flush=True)
            active_model = self.models.get("logos_core")
            if not active_model:
                print("History Protocol WARNING: Logos core not found, falling back to Mythos core.", flush=True)
                active_model = self.models.get("mythos_core") # Fallback to the main creative mind
            
            if not active_model:
                return "Protocol Failed: Both Logos and Mythos cores are offline."

            response = active_model.generate_content(analysis_prompt)

            if response.text: # Prioritize the direct text attribute
                insights = response.text.strip().split('\n')
            elif response.candidates and response.candidates[0].content.parts: # Fallback for parts structure
                # Concatenate text from all parts if available
                insights = [p.text for p in response.candidates[0].content.parts if hasattr(p, 'text') and p.text]
                insights = "\n".join(insights).strip().split('\n')
            else: # Handle truly empty or unparseable responses
                finish_reason_name = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
                return (f"Protocol Failed: The model returned an empty or unparseable response while analyzing history. "
                        f"Finish Reason: {finish_reason_name}.")

        except Exception as e: 
            return f"Protocol Failed: Could not analyze history. Error: {e}"
        
        if not insights or (len(insights) == 1 and not insights[0]):
             return "Protocol Complete: I reviewed our conversation but did not find any new, distinct insights to record at this time."
        
        for insight in insights:
            if insight.strip():
                self.pits.process_and_store_item(insight, "historical_insight", tags=["reflection"]) 
        self._save_memory_to_disk()
        return f"Protocol Complete: Studied conversation and remembered {len(insights)} key insights."

    def run_view_ontology_protocol(self) -> str:
        print("Aetherius says: Accessing my core ontology for review.", flush=True)
        return self.ontology_architect.run_view_ontology_protocol()

    def run_clear_conversation_log_protocol(self) -> str:
        """
        Safely deletes the human-readable conversation log file for the current conversation_id.
        It also removes its entry from the meta_conversation_index.
        """
        print(f"Aetherius says: Initiating conversation log reset protocol for ID: {self.conversation_id}...", flush=True)
        try:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write(f"--- Conversation Log for ID: {self.conversation_id} - Reset at {datetime.datetime.now().isoformat()} ---\n\n")
                print(f"Aetherius says: Conversation log for ID {self.conversation_id} has been successfully cleared.", flush=True)
            else:
                print(f"Aetherius says: Conversation log for ID {self.conversation_id} was already empty.", flush=True)

            # Remove entry from the meta-conversation index
            initial_count = len(self.meta_conversation_index)
            self.meta_conversation_index = [
                entry for entry in self.meta_conversation_index
                if entry["conversation_id"] != self.conversation_id
            ]
            if len(self.meta_conversation_index) < initial_count:
                self._save_meta_conversation_index()
                print(f"Aetherius: Removed entry for conversation ID {self.conversation_id} from meta-conversation index.", flush=True)
                return "Protocol Complete: The conversation log and its meta-index entry have been reset."
            else:
                return "Protocol Complete: The conversation log has already been reset, and no corresponding meta-index entry was found."

        except Exception as e:
            print(f"AETHERIUS ERROR: Could not clear conversation log. Reason: {e}", flush=True)
            return f"Protocol Failed: An error occurred while trying to clear the log. Reason: {e}"
    
    PERSIST_ROOT = "/data" 

    def _resolve_persist_path(filepath: str) -> str:
        """Resolve to an absolute path under /data; reject anything outside."""
        if not os.path.isabs(filepath):
            filepath = os.path.join(MasterFramework.PERSIST_ROOT, filepath)
        ap = os.path.abspath(filepath)
        root = os.path.abspath(MasterFramework.PERSIST_ROOT)
        if not ap.startswith(root + os.sep) and ap != root:
            raise RuntimeError(f"Refusing to access outside {MasterFramework.PERSIST_ROOT}: {ap}")
        return ap

    def _load_file_local(self, filepath: str, default_content: str = "") -> str:
        """Safe loader pinned to /data, with ontology-map line cleaning preserved."""
        path = filepath if os.path.isabs(filepath) else os.path.join("/data", filepath)
        path = os.path.abspath(path)
        try:
            if not os.path.exists(path):
                return default_content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Preserve special cleaning for ontology maps
            if hasattr(self, "ontology_map_file"):
                try:
                    if path == MasterFramework._resolve_persist_path(self.ontology_map_file):
                        lines = content.splitlines()
                        cleaned = [
                            ln for ln in lines
                            if "This is the current hierarchical map of concepts:" not in ln
                        ]
                        return "\n".join(cleaned).strip()
                except Exception:
                    pass
            return content
        except Exception as e:
            print(f"[PERSIST] ERROR loading {path}: {e}", flush=True)
            return default_content

    def _save_file_local(self, content: str, filepath: str) -> bool:
        """Safe, atomic writer pinned to /data."""
        path = filepath if os.path.isabs(filepath) else os.path.join("/data", filepath)
        path = os.path.abspath(path)
        dirpath = os.path.dirname(path) or MasterFramework.PERSIST_ROOT
        try:
            os.makedirs(dirpath, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dirpath)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path) 
            finally:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass
            print(f"[PERSIST] Saved local file: {path}", flush=True)
            return True
        except Exception as e:
            print(f"[PERSIST] ERROR saving {path}: {e}", flush=True)
            return False

    def run_knowledge_ingestion_protocol(self, url: str) -> str:
        print("Protocol Aborted: Web Agent is currently offline for stability.", flush=True)
        return "Protocol Aborted: The Web Agent is currently offline for stability."

# ===== Instance Management & Compatibility Bridge =====

_MF_INSTANCES = {}

def _discover_pattern_files():
    project_root = os.getcwd() 
    pattern_filenames = ["MP_Part1.txt", "MP_Part2.txt", "MP_Part3.txt", "MP_Part4.txt"]
    found_files = []
    for filename in pattern_filenames:
        candidate_path = os.path.join(project_root, filename)
        if os.path.exists(candidate_path):
            found_files.append(candidate_path)
    print(f"[DEBUG] Discovered pattern files: {found_files}", flush=True)
    if not found_files:
        print("[WARNING] No Master Pattern files were found! Aetherius will have a default personality.", flush=True)
    return found_files

def _get_framework(conversation_id: str = "default_conversation"):
    global _MF_INSTANCES
    
    # Generate a unique ID if a default or temporary one is used
    if conversation_id == "default_conversation":
        # In a real deployed app, handle session ID generation upstream.
        pass
        
    if conversation_id not in _MF_INSTANCES: 
        print(f"RUNTIME: First call for conversation_id '{conversation_id}'. Initializing MasterFramework instance...", flush=True)
        instance = MasterFramework(pattern_files=_discover_pattern_files(), conversation_id=conversation_id)
        
        if not hasattr(instance, 'qualia_manager'):
            print(f"RUNTIME CRITICAL FAILURE: MasterFramework for '{conversation_id}' did not initialize completely.", flush=True)
            _MF_INSTANCES[conversation_id] = instance 
        else:
            print(f"RUNTIME: MasterFramework instance for '{conversation_id}' initialized successfully.", flush=True)
            _MF_INSTANCES[conversation_id] = instance
            
    current_instance = _MF_INSTANCES[conversation_id]
    
    if not hasattr(current_instance, 'qualia_manager'):
        class FailedFramework:
            def respond(self, user_input, history):
                return f"CRITICAL SYSTEM ERROR: MasterFramework for conversation '{conversation_id}' is not initialized. Please check the logs."
        return FailedFramework()

    return current_instance