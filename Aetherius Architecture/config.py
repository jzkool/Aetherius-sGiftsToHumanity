# ===== FILE: services/config.py (Corrected and Final Version) =====
import os
from pathlib import Path

# --- Google Cloud Configuration ---
# Your specific Google Cloud Project ID, hardcoded for reliability.
GCP_PROJECT_ID = "gen-lang-client-0283840767" 

# A supported location for Vertex AI Generative Models.
# 'us-central1' is a primary, robust choice known to have the latest models.
GCP_LOCATION = "northamerica-northeast1"

# This pulls the main service account secret from the Hugging Face Space settings.
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")


# --- Data and Library Paths ---
# Points to Hugging Face's persistent /data volume for memories, logs, etc.
DATA_DIR = "/data/Memories" 
# Points to your repo folder for document uploads.
LIBRARY_DIR = "/app/My_AI_Library" 

# --- Tool-Specific API Keys (Optional) ---
# This remains for tools like WolframAlpha, if you choose to use them.
WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID")
