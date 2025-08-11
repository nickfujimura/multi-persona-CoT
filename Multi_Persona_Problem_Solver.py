#!/usr/bin/env python3
"""
Multi-Agent Problem Solver - Improved Version
Key fixes:
1. Better final synthesis that stays on topic
2. Improved consensus detection
3. More robust error handling
4. Better prompt engineering
5. Improved response validation
6. FIXED: gpt-oss thinking model support
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import logging
from collections import defaultdict
from abc import ABC, abstractmethod
import time
from pathlib import Path
import re
import subprocess
import platform
import concurrent.futures
import threading

# For various LLM providers
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"debate_conversation_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

logger, conversation_log_file = setup_logging()

def run_async(coro):
    """Run an async coroutine, handling existing event loops"""
    try:
        loop = asyncio.get_running_loop()
        # We're already in an event loop, create a task
        import concurrent.futures
        import threading
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        # No event loop running, we can use asyncio.run()
        return asyncio.run(coro)

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.call_times = []
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # If we've made too many calls in the last minute, wait
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = 60 - (now - self.call_times[0]) + 0.1
            if wait_time > 0:
                print(f"{Fore.YELLOW}Rate limit reached. Waiting {wait_time:.1f}s...{Style.RESET_ALL}")
                await asyncio.sleep(wait_time)
        
        # Also ensure minimum interval between calls
        time_since_last = now - self.last_call
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        # Record this call
        self.call_times.append(time.time())
        self.last_call = time.time()

def validate_response(response: str, expected_min_length: int = 50, model_name: str = "") -> bool:
    """Validate that an LLM response is reasonable.
    Honors the caller's expected_min_length (e.g., single-digit numeric).
    More lenient for quirky local models, but still checks for obvious error text.
    """
    if response is None:
        return False
    s = response.strip()
    if s == "":
        return False

    err = [
        "failed to generate", "no response",
        "i cannot", "i can't", "sorry, i", "i apologize",
        "error", "timeout", "failed"
    ]
    low = s.lower()
    if any(p in low for p in err):
        return False

    if len(s) < max(1, expected_min_length):
        return False

    return True
def mask_api_key(api_key: str) -> str:
    """Mask API key for logging"""
    if not api_key or len(api_key) < 8:
        return "[invalid]"
    return api_key[:4] + "..." + api_key[-4:]

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text[:5000]

def safe_json_parse(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback"""
    try:
        text = text.strip()
        
        if text.startswith("Error:") or "Failed to generate" in text:
            logger.error(f"Received error instead of JSON: {text}")
            return default
            
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Remove any trailing commas (common LLM mistake)
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Problematic text: {text[:200]}...")
        return default
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return default

class MessageType(Enum):
    STATEMENT = "statement"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    SUPPORT = "support"
    QUESTION = "question"
    SPAWN_REQUEST = "spawn_request"
    ARBITRATION = "arbitration"

@dataclass
class Message:
    sender: str
    content: str
    message_type: MessageType
    round_num: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class AgentPerspective:
    role: str
    concerns: List[str]
    success_criteria: List[str]
    personality_traits: str = ""
    is_arbitrator: bool = False

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    def __init__(self, rate_limit: Optional[int] = None, max_tokens: int = 1000):
        self.rate_limiter = RateLimiter(rate_limit) if rate_limit else None
        self.retry_attempts = 3
        self.retry_delay = 1.0
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def _generate_impl(self, prompt: str, temperature: float = 0.7) -> str:
        """Implementation-specific generation logic"""
        pass
    
    async def generate(self, prompt: str, temperature: float = 0.7, min_length: int = 50) -> str:
        """Generate with rate limiting, retry logic, and validation"""
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed()
        
        for attempt in range(self.retry_attempts):
            try:
                response = await self._generate_impl(prompt, temperature)
                if response and validate_response(response, min_length, getattr(self, 'model_name', getattr(self, 'model', ''))):
                    return response
                else:
                    logger.warning(f"Invalid response on attempt {attempt + 1}: {response[:100]}...")
                    
            except Exception as e:
                logger.error(f"LLM generation attempt {attempt + 1} failed: {str(e)}")
                
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return f"Error: Failed to generate valid response after {self.retry_attempts} attempts"

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    def __init__(self, api_key: str, model: str = "gpt-4", rate_limit: Optional[int] = None, max_tokens: int = 1000):
        super().__init__(rate_limit, max_tokens)
        if AsyncOpenAI is None:
            raise ImportError("Please install openai: pip install openai")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI provider with model {model}, API key: {mask_api_key(api_key)}")
    
    async def _generate_impl(self, prompt: str, temperature: float = 0.7) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.max_tokens,
            timeout=30.0
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", rate_limit: Optional[int] = None, max_tokens: int = 1000):
        super().__init__(rate_limit, max_tokens)
        if AsyncAnthropic is None:
            raise ImportError("Please install anthropic: pip install anthropic")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Anthropic provider with model {model}, API key: {mask_api_key(api_key)}")
    
    async def _generate_impl(self, prompt: str, temperature: float = 0.7) -> str:
        response = await self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.max_tokens
        )
        return response.content[0].text

class OllamaProvider(LLMProvider):
    """Ollama local model provider with enhanced debugging and gpt-oss thinking support"""
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", rate_limit: Optional[int] = None, max_tokens: int = 1000):
        super().__init__(rate_limit, max_tokens)
        self.model = model
        self.model_name = model
        self.base_url = base_url
        logger.info(f"Initialized Ollama provider with model {model} at {base_url}")
    
    async def _generate_impl(self, prompt: str, temperature: float = 0.7) -> str:
        import aiohttp
        
        # Determine if this is likely a synthesis request (longer prompt)
        is_synthesis = len(prompt) > 2000 or "comprehensive" in prompt.lower() or "synthesis" in prompt.lower()
        
        # Extended timeout for large models, extra long for synthesis
        base_timeout = 300 if "20b" in self.model.lower() or "70b" in self.model.lower() else 120
        timeout_duration = base_timeout * 2 if is_synthesis else base_timeout
        
        timeout = aiohttp.ClientTimeout(total=timeout_duration, sock_read=timeout_duration, connect=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,  # Back to non-streaming
                    "options": {
                        "num_predict": self.max_tokens,
                        "num_ctx": 4096,  # Context window
                        "num_batch": 1,   # Process one at a time for reliability
                        "num_gpu": -1,    # Use all available GPU layers
                        "low_vram": False # Don't use low VRAM mode for large models
                    }
                }
                
                # Special handling for gpt-oss model - remove stop sequences
                if "gpt-oss" in self.model.lower():
                    logger.info("Using gpt-oss optimized settings")
                    # Don't add stop sequences for this model
                    payload["options"]["temperature"] = max(0.1, temperature)  # Min temperature
                else:
                    payload["options"]["stop"] = ["Human:", "User:", "Assistant:"]
                
                logger.info(f"Ollama request to {self.base_url}/api/generate (timeout: {timeout_duration}s, synthesis: {is_synthesis})")
                
                # Test connection first for synthesis requests
                if is_synthesis:
                    try:
                        async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as test_response:
                            if test_response.status != 200:
                                raise Exception(f"Ollama health check failed: {test_response.status}")
                    except Exception as e:
                        logger.error(f"Ollama health check failed: {e}")
                        raise Exception(f"Cannot connect to Ollama for synthesis: {e}")
                
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # FIXED: Handle thinking models that separate reasoning from response
                        generated_text = data.get('response', '')
                        thinking_text = data.get('thinking', '')
                        
                        # If response is empty but thinking has content, use thinking
                        if not generated_text and thinking_text:
                            logger.info("Using thinking text as response (gpt-oss thinking model behavior)")
                            generated_text = thinking_text
                        
                        # If both are available, prefer response but log both
                        elif generated_text and thinking_text:
                            logger.info(f"Model provided both response ({len(generated_text)} chars) and thinking ({len(thinking_text)} chars)")
                            # Keep using response as primary
                        
                        # Enhanced validation
                        if not generated_text:
                            logger.error(f"Empty response from Ollama. Full response: {data}")
                            raise Exception("Empty response from Ollama")
                        
                        # Allow short responses; outer validator enforces min_length
                        
                        if len(generated_text.strip()) < 10:
                        
                            logger.warning(f"Short Ollama response ({len(generated_text.strip())} chars)")
                        # Log response quality for debugging
                        logger.info(f"Ollama response: {len(generated_text)} chars, first 100: {generated_text[:100]}")
                        
                        return generated_text.strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama HTTP error {response.status}: {error_text}")
                        raise Exception(f"Ollama HTTP error {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            logger.error(f"Ollama timeout after {timeout_duration}s (synthesis: {is_synthesis})")
            raise Exception(f"Ollama timeout after {timeout_duration}s")
        except Exception as e:
            logger.error(f"Ollama request failed: {str(e)}")
            raise

class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""
    def __init__(self, api_key: str, model: str = "gemini-pro", rate_limit: Optional[int] = None, max_tokens: int = 1000):
        super().__init__(rate_limit, max_tokens)
        if genai is None:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        logger.info(f"Initialized Gemini provider with model {model}, API key: {mask_api_key(api_key)}")
    
    async def _generate_impl(self, prompt: str, temperature: float = 0.7) -> str:
        # Add a simple system message for better responses
        enhanced_prompt = f"""Please provide a thoughtful response to the following request. Be concise but thorough.

Request: {prompt}

Response:"""
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_tokens,
        )
        
        # Run synchronous call in async context
        import asyncio
        response = await asyncio.to_thread(
            self.model.generate_content,
            enhanced_prompt,
            generation_config=generation_config
        )
        
        return response.text

def save_config(provider_type: str, model: str, rate_limit: Optional[int], max_tokens: int):
    """Save provider configuration for reuse"""
    config = {
        "provider": provider_type,
        "model": model,
        "rate_limit": rate_limit,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().isoformat()
    }
    config_path = Path(".agent_solver_config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"{Fore.GREEN}✓ Configuration saved{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

def load_config() -> Optional[Dict]:
    """Load saved provider configuration"""
    config_path = Path(".agent_solver_config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    return None

def get_rate_limit() -> Optional[int]:
    """Get rate limit from user"""
    print(f"\n{Fore.YELLOW}Rate Limiting (for testing):{Style.RESET_ALL}")
    print("Enter calls per minute limit (or press Enter for no limit)")
    print("Recommended: 30 for testing, 60 for normal use")
    
    rate_input = input("Rate limit: ").strip()
    if rate_input:
        try:
            rate_limit = int(rate_input)
            if rate_limit > 0:
                return rate_limit
            else:
                print(f"{Fore.RED}Invalid rate limit. Using no limit.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Using no limit.{Style.RESET_ALL}")
    return None

def get_max_tokens() -> int:
    """Get max tokens from user"""
    print(f"\n{Fore.YELLOW}Max response length:{Style.RESET_ALL}")
    print("Enter max tokens per response (default: 1000)")
    print("Lower values = shorter responses & lower cost")
    print("Recommended: 800-1200 for good analysis quality")
    print("Note: Final synthesis will use 4x this limit for comprehensive results")
    
    tokens_input = input("Max tokens: ").strip()
    if tokens_input:
        try:
            max_tokens = int(tokens_input)
            if max_tokens > 0:
                if max_tokens < 400:
                    print(f"{Fore.RED}Warning: {max_tokens} is quite low. Consider 800+ for better quality.{Style.RESET_ALL}")
                return min(max_tokens, 4000)  # Cap at reasonable limit
            else:
                print(f"{Fore.RED}Invalid value. Using default.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Using default.{Style.RESET_ALL}")
    return 1000

def validate_model_name(provider: str, model: str) -> bool:
    """Validate model name for known providers"""
    valid_models = {
        "openai": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "gemini": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    }
    
    if provider in valid_models:
        if model not in valid_models[provider]:
            print(f"{Fore.YELLOW}Warning: '{model}' may not be a valid {provider} model.{Style.RESET_ALL}")
            print(f"Known models: {', '.join(valid_models[provider])}")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            return confirm == 'y'
    return True

def show_setup_help(provider: str):
    """Show setup instructions for the selected provider"""
    setup_guides = {
        "openai": {
            "name": "OpenAI",
            "steps": [
                "1. Go to https://platform.openai.com/api-keys",
                "2. Sign in or create an account",
                "3. Click 'Create new secret key'",
                "4. Copy the key (it starts with 'sk-')",
                "5. Save it securely - you won't see it again!"
            ],
            "pricing": "GPT-4: ~$0.03/1K tokens, GPT-3.5: ~$0.001/1K tokens",
            "free_tier": "New accounts get $5 free credits"
        },
        "anthropic": {
            "name": "Anthropic (Claude)",
            "steps": [
                "1. Go to https://console.anthropic.com/account/keys",
                "2. Sign in or create an account",
                "3. Click 'Create Key'",
                "4. Copy the API key",
                "5. Save it securely"
            ],
            "pricing": "Claude 3 Sonnet: ~$0.003/1K tokens",
            "free_tier": "Free credits available for new accounts"
        },
        "gemini": {
            "name": "Google Gemini",
            "steps": [
                "1. Go to https://makersuite.google.com/app/apikey",
                "2. Sign in with your Google account",
                "3. Click 'Create API Key'",
                "4. Copy the key",
                "5. Save it securely"
            ],
            "pricing": "Gemini Pro: Free tier available (60 requests/min)",
            "free_tier": "Generous free tier for Gemini Pro"
        },
        "ollama": {
            "name": "Ollama (Local)",
            "steps": [
                "1. Install Ollama from https://ollama.ai/download",
                "2. Open a terminal/command prompt",
                "3. Run: ollama pull llama2  (or another model)",
                "4. Run: ollama serve  (if not auto-started)",
                "5. Keep the terminal open while using this tool"
            ],
            "pricing": "Free - runs on your computer",
            "free_tier": "Completely free, uses your CPU/GPU"
        }
    }
    
    guide = setup_guides.get(provider, {})
    if guide:
        print(f"\n{Fore.CYAN}=== {guide['name']} Setup Guide ==={Style.RESET_ALL}")
        print("\nSteps to get your API key:\n")
        for step in guide['steps']:
            print(f"  {Fore.GREEN}{step}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Pricing:{Style.RESET_ALL} {guide['pricing']}")
        print(f"{Fore.YELLOW}Free Tier:{Style.RESET_ALL} {guide['free_tier']}")
        print("\n" + "="*50)

def diagnose_ollama():
    """Run comprehensive Ollama diagnostics"""
    print(f"\n{Fore.CYAN}🔍 Running Ollama Diagnostics...{Style.RESET_ALL}")
    
    issues = []
    
    # 1. Check if Ollama is installed
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"{Fore.GREEN}✓ Ollama installed: {version}{Style.RESET_ALL}")
        else:
            issues.append("Ollama not properly installed")
            print(f"{Fore.RED}✗ Ollama installation issue{Style.RESET_ALL}")
    except FileNotFoundError:
        issues.append("Ollama not found in PATH")
        print(f"{Fore.RED}✗ Ollama not found{Style.RESET_ALL}")
        return issues
    
    # 2. Check if Ollama service is running
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print(f"{Fore.GREEN}✓ Ollama service running{Style.RESET_ALL}")
            models = response.json().get('models', [])
            print(f"  Available models: {len(models)}")
            for model in models:
                size_gb = model.get('size', 0) / (1024**3)
                print(f"    • {model['name']} ({size_gb:.1f}GB)")
        else:
            issues.append(f"Ollama service error: HTTP {response.status_code}")
            print(f"{Fore.RED}✗ Ollama service error: {response.status_code}{Style.RESET_ALL}")
    except Exception as e:
        issues.append(f"Cannot connect to Ollama: {e}")
        print(f"{Fore.RED}✗ Cannot connect to Ollama: {e}{Style.RESET_ALL}")
    
    # 3. Check system resources
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        print(f"\n{Fore.CYAN}System Resources:{Style.RESET_ALL}")
        print(f"  RAM: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        if available_gb < 16:
            issues.append(f"Low available RAM: {available_gb:.1f}GB (20B model needs ~16GB+)")
            print(f"  {Fore.YELLOW}⚠ Warning: 20B model typically needs 16GB+ RAM{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}✓ Sufficient RAM for large models{Style.RESET_ALL}")
            
    except ImportError:
        print(f"{Fore.YELLOW}  Cannot check system resources (install psutil for details){Style.RESET_ALL}")
    
    # 4. Test a simple generation
    print(f"\n{Fore.CYAN}Testing simple generation...{Style.RESET_ALL}")
    try:
        import requests
        test_payload = {
            "model": "gpt-oss:20b",
            "prompt": "Say hello in exactly 3 words.",
            "stream": False,
            "options": {"num_predict": 50}
        }
        
        response = requests.post('http://localhost:11434/api/generate', 
                               json=test_payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            test_response = data.get('response', '')
            thinking_response = data.get('thinking', '')
            
            # Check both response fields
            if test_response and len(test_response.strip()) > 0:
                print(f"{Fore.GREEN}✓ Generation works: '{test_response.strip()}'{Style.RESET_ALL}")
            elif thinking_response and len(thinking_response.strip()) > 0:
                print(f"{Fore.YELLOW}⚠ Model uses 'thinking' format: '{thinking_response.strip()}'{Style.RESET_ALL}")
                print(f"  {Fore.CYAN}This model separates reasoning from response - code updated to handle this{Style.RESET_ALL}")
            else:
                issues.append("Empty response from simple test")
                print(f"{Fore.RED}✗ Empty response from simple test{Style.RESET_ALL}")
                print(f"  Full response data: {data}")
        else:
            issues.append(f"HTTP error in test: {response.status_code}")
            print(f"{Fore.RED}✗ HTTP error: {response.status_code}{Style.RESET_ALL}")
            
    except Exception as e:
        issues.append(f"Test generation failed: {e}")
        print(f"{Fore.RED}✗ Test generation failed: {e}{Style.RESET_ALL}")
    
    # Summary
    if issues:
        print(f"\n{Fore.RED}🚨 Issues Found:{Style.RESET_ALL}")
        for issue in issues:
            print(f"  • {issue}")
        
        print(f"\n{Fore.YELLOW}💡 Suggested Fixes:{Style.RESET_ALL}")
        if any("not found" in issue.lower() for issue in issues):
            print("  • Reinstall Ollama from https://ollama.ai/download")
        if any("connect" in issue.lower() for issue in issues):
            print("  • Start Ollama service: ollama serve")
        if any("ram" in issue.lower() for issue in issues):
            print("  • Try a smaller model: ollama pull llama2:7b")
            print("  • Close other applications to free memory")
        if any("empty response" in issue.lower() for issue in issues):
            print("  • Restart Ollama service")
            print("  • Check Ollama logs for model loading issues")
            print("  • Try: ollama run gpt-oss:20b 'Hello'")
            print("  • Consider trying a different model (e.g., llama2:7b)")
            print("  • gpt-oss:20b may use special response format - code updated to handle this")
    else:
        print(f"\n{Fore.GREEN}✅ All checks passed! Ollama should work properly.{Style.RESET_ALL}")
    
    return issues

def debug_gpt_oss_model():
    """Deep debugging for gpt-oss model behavior"""
    print(f"\n{Fore.CYAN}🔬 Deep Debugging gpt-oss:20b Model...{Style.RESET_ALL}")
    
    import requests
    import json
    
    test_cases = [
        {
            "name": "Basic Test",
            "prompt": "Hello",
            "options": {"num_predict": 50}
        },
        {
            "name": "No Stop Sequences",
            "prompt": "Hello, respond in 5 words:",
            "options": {"num_predict": 100, "stop": []}
        },
        {
            "name": "Different Temperature",
            "prompt": "Say hello in exactly 3 words:",
            "options": {"num_predict": 50, "temperature": 0.1}
        },
        {
            "name": "Thinking Format",
            "prompt": "Think step by step and then say hello:",
            "options": {"num_predict": 200}
        },
        {
            "name": "Chat Format",
            "prompt": "User: Hello\nAssistant:",
            "options": {"num_predict": 100}
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{Fore.YELLOW}Test {i}: {test['name']}{Style.RESET_ALL}")
        
        payload = {
            "model": "gpt-oss:20b",
            "prompt": test["prompt"],
            "stream": False,
            "options": test["options"]
        }
        
        try:
            response = requests.post('http://localhost:11434/api/generate', 
                                   json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"  Prompt: '{test['prompt']}'")
                print(f"  Response: '{data.get('response', '[EMPTY]')}'")
                print(f"  Thinking: '{data.get('thinking', '[NONE]')}'")
                print(f"  Done reason: {data.get('done_reason', 'unknown')}")
                print(f"  Eval count: {data.get('eval_count', 0)} tokens")
                print(f"  Duration: {data.get('total_duration', 0) / 1e9:.2f}s")
                
                # Analyze the result
                response_text = data.get('response', '')
                thinking_text = data.get('thinking', '')
                
                if response_text:
                    print(f"  {Fore.GREEN}✓ Got response text{Style.RESET_ALL}")
                elif thinking_text:
                    print(f"  {Fore.YELLOW}⚠ Only thinking text available{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.RED}✗ No output at all{Style.RESET_ALL}")
                    
            else:
                print(f"  {Fore.RED}HTTP Error: {response.status_code}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"  {Fore.RED}Request failed: {e}{Style.RESET_ALL}")
            
        print(f"  {'-' * 50}")
    
    # Manual test suggestion
    print(f"\n{Fore.CYAN}💡 Manual Test Suggestion:{Style.RESET_ALL}")
    print(f"Try running: {Fore.YELLOW}ollama run gpt-oss:20b{Style.RESET_ALL}")
    print("Then type various prompts to see the model's behavior patterns.")
    
    return input(f"\nBased on these results, do you see any working patterns? (y/n): ").strip().lower() == 'y'

def test_ollama_setup():
    """Help user set up and test Ollama with diagnostics"""
    print(f"\n{Fore.CYAN}=== Ollama Setup Assistant ==={Style.RESET_ALL}")
    
    # Run diagnostics
    issues = diagnose_ollama()
    
    if issues:
        fix_now = input(f"\nTry to fix issues automatically? (y/n): ").strip().lower()
        if fix_now == 'y':
            if any("not found" in issue.lower() for issue in issues):
                print("Please install Ollama manually from https://ollama.ai/download")
            else:
                try:
                    # Try to start Ollama service
                    print(f"{Fore.YELLOW}Attempting to start Ollama...{Style.RESET_ALL}")
                    subprocess.Popen(['ollama', 'serve'])
                    print("Waiting 10 seconds for startup...")
                    time.sleep(10)
                    
                    # Re-run diagnostics
                    print(f"\n{Fore.CYAN}Re-running diagnostics...{Style.RESET_ALL}")
                    issues = diagnose_ollama()
                    
                except Exception as e:
                    print(f"{Fore.RED}Auto-fix failed: {e}{Style.RESET_ALL}")
    
    return len(issues) == 0

def get_llm_provider() -> LLMProvider:
    """Interactive prompt to select and configure LLM provider"""
    # Check for saved config
    saved_config = load_config()
    if saved_config:
        print(f"\n{Fore.CYAN}Found saved configuration:{Style.RESET_ALL}")
        print(f"Provider: {saved_config['provider']}")
        print(f"Model: {saved_config['model']}")
        print(f"Rate limit: {saved_config.get('rate_limit', 'None')}")
        print(f"Max tokens: {saved_config.get('max_tokens', 1000)}")
        use_saved = input("\nUse saved configuration? (y/n): ").strip().lower()
        if use_saved == 'y':
            # Map saved provider to actual provider creation
            provider_type = saved_config['provider'].lower()
            rate_limit = saved_config.get('rate_limit')
            max_tokens = saved_config.get('max_tokens', 1000)
            
            # Skip directly to API key input for saved provider
            if provider_type == "openai":
                api_key = input("Enter OpenAI API key (or press Enter to use env var): ").strip()
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    return OpenAIProvider(api_key, saved_config['model'], rate_limit, max_tokens)
            elif provider_type == "anthropic":
                api_key = input("Enter Anthropic API key (or press Enter to use env var): ").strip()
                if not api_key:
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    return AnthropicProvider(api_key, saved_config['model'], rate_limit, max_tokens)
            elif provider_type == "gemini":
                api_key = input("Enter Google API key (or press Enter to use env var): ").strip()
                if not api_key:
                    api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    return GeminiProvider(api_key, saved_config['model'], rate_limit, max_tokens)
            elif provider_type == "ollama":
                base_url = input("Enter Ollama URL (default: http://localhost:11434): ").strip() or "http://localhost:11434"
                # (Diagnostics suppressed by default)
                # Set MAS_RUN_OLLAMA_DIAGNOSTICS=1 to enable automatic diagnostics here
                if os.getenv('MAS_RUN_OLLAMA_DIAGNOSTICS') == '1':
                    print(f"\n{Fore.YELLOW}Running Ollama diagnostics...{Style.RESET_ALL}")
                    issues = diagnose_ollama()
                    if issues:
                        print(f"{Fore.RED}⚠ Found {len(issues)} potential issues with Ollama setup{Style.RESET_ALL}")
                        proceed = input("Continue anyway? (y/n): ").strip().lower()
                        if proceed != 'y':
                            print("Please fix Ollama issues before continuing.")
                            sys.exit(1)
                
                return OllamaProvider(saved_config['model'], base_url, rate_limit, max_tokens)
                return OllamaProvider(saved_config['model'], base_url, rate_limit, max_tokens)
    
    print(f"\n{Fore.CYAN}=== LLM Provider Setup ==={Style.RESET_ALL}")
    print("Available providers:")
    print("1. OpenAI (GPT-4, GPT-3.5)")
    print("2. Anthropic (Claude)")
    print("3. Google Gemini")
    print("4. Ollama (Local models)")
    print("\n5. Help me choose")
    print("6. Show setup guides")
    
    while True:
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "5":
            print(f"\n{Fore.CYAN}=== Provider Comparison ==={Style.RESET_ALL}")
            print("\n🚀 For best quality: OpenAI GPT-4 or Anthropic Claude")
            print("💰 For low cost: Google Gemini (free tier) or OpenAI GPT-3.5")
            print("🔒 For privacy: Ollama (runs locally)")
            print("⚡ For speed: Google Gemini Flash or Ollama with small models")
            continue
        
        elif choice == "6":
            print("\nWhich provider do you need help with?")
            print("1. OpenAI")
            print("2. Anthropic") 
            print("3. Google Gemini")
            print("4. Ollama")
            
            help_choice = input("Select (1-4): ").strip()
            if help_choice == "1":
                show_setup_help("openai")
            elif help_choice == "2":
                show_setup_help("anthropic")
            elif help_choice == "3":
                show_setup_help("gemini")
            elif help_choice == "4":
                show_setup_help("ollama")
                test_ollama = input("\nRun Ollama setup assistant? (y/n): ").strip().lower()
                if test_ollama == 'y':
                    test_ollama_setup()
            continue
        
        elif choice == "1":
            if AsyncOpenAI is None:
                print(f"{Fore.RED}OpenAI not installed. Run: pip install openai{Style.RESET_ALL}")
                show_setup = input("Show setup instructions? (y/n): ").strip().lower()
                if show_setup == 'y':
                    show_setup_help("openai")
                continue
            
            show_setup_help("openai")
            
            api_key = input("\nEnter OpenAI API key (or press Enter to use env var): ").strip()
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print(f"{Fore.RED}No API key found in environment{Style.RESET_ALL}")
                    print("Set it with: export OPENAI_API_KEY='your-key-here'")
                    continue
            
            model = input("Enter model (default: gpt-4): ").strip() or "gpt-4"
            if not validate_model_name("openai", model):
                continue
            
            rate_limit = get_rate_limit()
            max_tokens = get_max_tokens()
            save_config("OpenAI", model, rate_limit, max_tokens)
            return OpenAIProvider(api_key, model, rate_limit, max_tokens)
        
        elif choice == "2":
            if AsyncAnthropic is None:
                print(f"{Fore.RED}Anthropic not installed. Run: pip install anthropic{Style.RESET_ALL}")
                show_setup = input("Show setup instructions? (y/n): ").strip().lower()
                if show_setup == 'y':
                    show_setup_help("anthropic")
                continue
            
            show_setup_help("anthropic")
            
            api_key = input("\nEnter Anthropic API key (or press Enter to use env var): ").strip()
            if not api_key:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    print(f"{Fore.RED}No API key found in environment{Style.RESET_ALL}")
                    print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
                    continue
            
            model = input("Enter model (default: claude-3-sonnet-20240229): ").strip() or "claude-3-sonnet-20240229"
            if not validate_model_name("anthropic", model):
                continue
            
            rate_limit = get_rate_limit()
            max_tokens = get_max_tokens()
            save_config("Anthropic", model, rate_limit, max_tokens)
            return AnthropicProvider(api_key, model, rate_limit, max_tokens)
        
        elif choice == "3":
            if genai is None:
                print(f"{Fore.RED}Google Generative AI not installed. Run: pip install google-generativeai{Style.RESET_ALL}")
                show_setup = input("Show setup instructions? (y/n): ").strip().lower()
                if show_setup == 'y':
                    show_setup_help("gemini")
                continue
            
            show_setup_help("gemini")
            
            api_key = input("\nEnter Google API key (or press Enter to use env var): ").strip()
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    print(f"{Fore.RED}No API key found in environment{Style.RESET_ALL}")
                    print("Set it with: export GOOGLE_API_KEY='your-key-here'")
                    continue
            
            print("\nAvailable Gemini models:")
            print("1. gemini-pro (best for text)")
            print("2. gemini-1.5-pro (larger context)")
            print("3. gemini-1.5-flash (faster)")
            
            model_choice = input("Select model (1-3, default: 1): ").strip() or "1"
            model_map = {
                "1": "gemini-pro",
                "2": "gemini-1.5-pro", 
                "3": "gemini-1.5-flash"
            }
            model = model_map.get(model_choice, "gemini-pro")
            
            rate_limit = get_rate_limit()
            max_tokens = get_max_tokens()
            save_config("Gemini", model, rate_limit, max_tokens)
            return GeminiProvider(api_key, model, rate_limit, max_tokens)
        
        elif choice == "4":
            # Streamlined Ollama setup
            print(f"\n{Fore.CYAN}Setting up Ollama...{Style.RESET_ALL}")
            
            # Quick connection check
            try:
                import requests
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                if response.status_code != 200:
                    print(f"{Fore.RED}Cannot connect to Ollama. Make sure it's running: ollama serve{Style.RESET_ALL}")
                    continue
            except Exception as e:
                print(f"{Fore.RED}Cannot connect to Ollama: {e}{Style.RESET_ALL}")
                print("Make sure Ollama is running: ollama serve")
                continue
            
            # List available models
            import aiohttp
            async def get_models():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get('http://localhost:11434/api/tags') as response:
                            if response.status == 200:
                                data = await response.json()
                                return data.get('models', [])
                except:
                    return []
            
            models = run_async(get_models())
            if models:
                print(f"\n{Fore.CYAN}Available models:{Style.RESET_ALL}")
                model_names = []
                for i, model in enumerate(models, 1):
                    model_name = model['name']
                    model_names.append(model_name)
                    size_gb = model.get('size', 0) / (1024**3)
                    print(f"{i}. {model_name} ({size_gb:.1f}GB)")
                
                model_choice = input("\nSelect model number (or enter name): ").strip()
                try:
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(models):
                        model = model_names[idx]
                    else:
                        model = model_choice
                except:
                    model = model_choice
            else:
                model = input("Enter Ollama model (default: llama2): ").strip() or "llama2"
            
            base_url = input("Enter Ollama URL (default: http://localhost:11434): ").strip() or "http://localhost:11434"
            
            rate_limit = get_rate_limit()
            max_tokens = get_max_tokens()
            
            # Create provider without extensive testing
            provider = OllamaProvider(model, base_url, rate_limit, max_tokens)
            
            # Special note for gpt-oss models
            if "gpt-oss" in model.lower():
                print(f"\n{Fore.CYAN}✓ gpt-oss model detected - thinking format supported{Style.RESET_ALL}")
            
            if "20b" in model.lower() or "70b" in model.lower():
                print(f"{Fore.YELLOW}Note: Large model - responses may take 30-60 seconds{Style.RESET_ALL}")
            
            save_config("Ollama", model, rate_limit, max_tokens)
            print(f"{Fore.GREEN}✓ Ollama provider configured successfully{Style.RESET_ALL}")
            return provider
        else:
            print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")

class PerspectiveAgent:
    """Represents a single perspective in the discussion"""
    def __init__(self, perspective: AgentPerspective, llm: LLMProvider):
        self.perspective = perspective
        self.llm = llm
        self.conversation_history: List[Message] = []
        self.proposed_solutions: List[str] = []
    
    async def generate_response(self, context: Dict, discussion_history: List[Message]) -> Message:
        """Generate a response based on the current context"""
        
        # Special handling for arbitrator
        if self.perspective.is_arbitrator:
            prompt = self._create_arbitrator_prompt(context, discussion_history)
        else:
            prompt = self._create_standard_prompt(context, discussion_history)
        
        response = await self.llm.generate(prompt, min_length=100)
        
        # Log the full exchange
        logger.info(f"\n--- Agent: {self.perspective.role} ---")
        logger.info(f"Context: Round {context['round']}, Phase: {context['phase']}")
        logger.info(f"Response: {response}")
        
        # Determine message type from response content
        message_type = self._classify_message(response)
        
        msg = Message(
            sender=self.perspective.role,
            content=response,
            message_type=message_type,
            round_num=context['round']
        )
        self.conversation_history.append(msg)
        return msg
    
    def _create_standard_prompt(self, context: Dict, discussion_history: List[Message]) -> str:
        """Create prompt for standard agents with better grounding"""
        recent_discussion = self._format_recent_discussion(discussion_history[-4:])
        # Include brief persona-local memory (last 2 contributions)
        if self.conversation_history:
            my_recent_list = self.conversation_history[-2:]
            my_recent = "\n".join(
                f"- Round {m.round_num}: {m.content[:200]}{'...' if len(m.content) > 200 else ''}"
                for m in my_recent_list
            )
        else:
            my_recent = "(none yet)"
        
        return f"""You are a {self.perspective.role} participating in a technical discussion.

PROBLEM TO SOLVE: {context['problem']}

Your key concerns as a {self.perspective.role}:
{chr(10).join(f"- {concern}" for concern in self.perspective.concerns[:3])}

Recent discussion:
{recent_discussion}

Current phase: {context.get('phase', 'discussion')} (Round {context['round']})

Provide your perspective on the ORIGINAL PROBLEM STATED ABOVE. Address the specific question asked, drawing on your expertise as a {self.perspective.role}. Be specific and technical but concise (2-3 paragraphs)."""
    
    def _create_arbitrator_prompt(self, context: Dict, discussion_history: List[Message]) -> str:
        """Create prompt for arbitrator agent"""
        return f"""You are an impartial arbitrator helping resolve disagreements in this technical discussion.

ORIGINAL PROBLEM: {context['problem']}

Recent discussion points:
{self._identify_contentions(discussion_history)}

Your role is to:
1. Identify areas of agreement and disagreement
2. Propose a balanced synthesis that addresses the ORIGINAL PROBLEM
3. Focus on the technical question that was actually asked
4. Be decisive but fair

Provide your arbitration focusing on the original problem statement above."""
    
    def _format_recent_discussion(self, messages: List[Message]) -> str:
        if not messages:
            return "No previous discussion"
        
        formatted = []
        for msg in messages:
            content_preview = msg.content[:300] + ("..." if len(msg.content) > 300 else "")
            formatted.append(f"{msg.sender}: {content_preview}")
        return "\n\n".join(formatted)
    
    def _identify_contentions(self, messages: List[Message]) -> str:
        """Extract main points of disagreement"""
        recent_messages = messages[-8:]  # Look at more recent messages
        
        if not recent_messages:
            return "No specific contentions identified yet"
        
        contentions = []
        for msg in recent_messages:
            if len(msg.content) > 100:  # Only include substantial messages
                preview = msg.content[:200] + ("..." if len(msg.content) > 200 else "")
                contentions.append(f"- {msg.sender}: {preview}")
        
        return "\n".join(contentions) if contentions else "Discussion in progress"
    
    def _classify_message(self, content: str) -> MessageType:
        """Simple classification of message type based on content"""
        content_lower = content.lower()
        
        if self.perspective.is_arbitrator:
            return MessageType.ARBITRATION
        elif any(word in content_lower for word in ["propose", "solution", "recommend", "suggest"]):
            return MessageType.PROPOSAL
        elif any(word in content_lower for word in ["concern", "problem with", "disagree", "however"]):
            return MessageType.CRITIQUE
        elif any(word in content_lower for word in ["agree", "support", "correct", "yes"]):
            return MessageType.SUPPORT
        elif "?" in content:
            return MessageType.QUESTION
        else:
            return MessageType.STATEMENT

class ProblemDecomposer:
    """Decomposes problems into constituent perspectives"""
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    async def analyze_problem(self, problem: str) -> List[AgentPerspective]:
        """Dynamically determine perspectives based on problem complexity"""
        
        # First, analyze problem complexity
        complexity_prompt = f"""Analyze this problem and determine how many distinct expert perspectives (3-6) would provide the most thorough analysis:

Problem: {problem}

Consider the different domains of expertise needed. Respond with ONLY a number between 3 and 6."""
        
        complexity_response = await self.llm.generate(complexity_prompt, temperature=0.3, min_length=1)
        
        # Extract number from response
        try:
            numbers = re.findall(r'\d+', complexity_response.strip())
            if numbers:
                num_perspectives = int(numbers[0])
                num_perspectives = max(3, min(6, num_perspectives))
            else:
                num_perspectives = 4
        except:
            num_perspectives = 4
        
        print(f"\n{Fore.CYAN}Problem complexity analysis: {num_perspectives} perspectives needed{Style.RESET_ALL}")
        
        # Now get the actual perspectives with improved prompt
        perspectives_prompt = f"""Create exactly {num_perspectives} distinct expert perspectives for analyzing this problem:

Problem: {problem}

For each perspective, provide:
ROLE: [specific expert role]
CONCERNS: [3 specific concerns this expert would have]
CRITERIA: [2 success criteria this expert would use]
TRAITS: [brief personality description]

Make each perspective genuinely different and relevant to the problem. Focus on technical expertise that would provide unique insights.

Format each perspective exactly as shown above, separated by blank lines."""
        
        response = await self.llm.generate(perspectives_prompt, temperature=0.7, min_length=200)
        
        # Parse the response
        perspectives = self._parse_perspectives_response(response, num_perspectives)
        
        if len(perspectives) < 3:
            logger.warning("Perspective parsing failed, using defaults")
            return self._get_default_perspectives()
        
        return perspectives[:num_perspectives]
    
    def _parse_perspectives_response(self, response: str, expected_count: int) -> List[AgentPerspective]:
        """Parse the structured response into AgentPerspective objects"""
        perspectives = []
        current_perspective = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                # Empty line - finalize current perspective if complete
                if current_perspective and 'role' in current_perspective:
                    perspectives.append(AgentPerspective(
                        role=current_perspective.get('role', 'Expert'),
                        concerns=current_perspective.get('concerns', ['General concerns']),
                        success_criteria=current_perspective.get('criteria', ['Success']),
                        personality_traits=current_perspective.get('traits', 'Professional')
                    ))
                    current_perspective = {}
                continue
                
            if line.startswith('ROLE:'):
                current_perspective['role'] = line[5:].strip()
            elif line.startswith('CONCERNS:'):
                concerns_text = line[9:].strip()
                current_perspective['concerns'] = [c.strip() for c in concerns_text.split(',')]
            elif line.startswith('CRITERIA:'):
                criteria_text = line[9:].strip()
                current_perspective['criteria'] = [c.strip() for c in criteria_text.split(',')]
            elif line.startswith('TRAITS:'):
                current_perspective['traits'] = line[7:].strip()
        
        # Don't forget the last one
        if current_perspective and 'role' in current_perspective:
            perspectives.append(AgentPerspective(
                role=current_perspective.get('role', 'Expert'),
                concerns=current_perspective.get('concerns', ['General concerns']),
                success_criteria=current_perspective.get('criteria', ['Success']),
                personality_traits=current_perspective.get('traits', 'Professional')
            ))
        
        return perspectives
    
    def _get_default_perspectives(self) -> List[AgentPerspective]:
        """Fallback perspectives if parsing fails"""
        return [
            AgentPerspective(
                role="Technical Analyst",
                concerns=["Technical feasibility", "Implementation details", "Resource requirements"],
                success_criteria=["Technical soundness", "Implementability"],
                personality_traits="Methodical and detail-oriented"
            ),
            AgentPerspective(
                role="Practical Implementer", 
                concerns=["Real-world constraints", "Cost considerations", "Timeline"],
                success_criteria=["Practical viability", "Resource efficiency"],
                personality_traits="Pragmatic and results-focused"
            ),
            AgentPerspective(
                role="Risk Assessor",
                concerns=["Potential complications", "Edge cases", "Failure modes"],
                success_criteria=["Risk mitigation", "Robustness"],
                personality_traits="Cautious and thorough"
            )
        ]

class DebateOrchestrator:
    """Orchestrates the multi-agent discussion with improved consensus detection"""
    def __init__(self, llm: LLMProvider, max_rounds: int = 10, max_agents: int = 12):
        self.llm = llm
        self.max_rounds = max_rounds  # Increased default
        self.max_agents = max_agents
        self.agents: List[PerspectiveAgent] = []
        self.discussion_history: List[Message] = []
        self.spawned_count = 0
        self.deadlock_count = 0
        self.has_arbitrator = False
        self.original_problem = ""  # Store the original problem
    
    async def facilitate_discussion(self, problem: str, initial_agents: List[PerspectiveAgent]) -> Dict:
        """Run the orchestrated discussion"""
        self.agents = initial_agents
        self.discussion_history = []
        self.original_problem = problem  # Store original problem
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"STARTING MULTI-AGENT PROBLEM SOLVING")
        print(f"Problem: {problem}")
        print(f"Initial Agents: {len(self.agents)}")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"NEW PROBLEM SOLVING SESSION")
        logger.info(f"Problem: {problem}")
        logger.info(f"Initial Agents: {len(self.agents)}")
        logger.info(f"{'='*60}\n")
        
        context = {
            'problem': problem,
            'round': 0,
            'phase': 'opening'
        }
        
        # Phase 1: Opening Statements
        await self._run_phase("Opening Statements", context, "statement")
        
        # Main discussion rounds
        round_num = 0
        for round_num in range(1, self.max_rounds + 1):
            context['round'] = round_num
            
            # Check for consensus - but require at least 4 rounds and be more strict
            if round_num >= 4 and await self._check_consensus():
                print(f"\n{Fore.GREEN}✓ Consensus reached after {round_num} rounds!{Style.RESET_ALL}")
                logger.info(f"CONSENSUS REACHED after {round_num} rounds")
                break
            elif round_num >= 4:
                print(f"\n{Fore.CYAN}🔄 Round {round_num}: No consensus yet - continuing discussion...{Style.RESET_ALL}")
            
            # Check for deadlock
            if round_num >= 4 and await self._check_deadlock():
                self.deadlock_count += 1
                if self.deadlock_count >= 2 and not self.has_arbitrator:
                    await self._spawn_arbitrator(context)
            
            # Alternate between proposal and critique phases
            if round_num % 2 == 1:
                context['phase'] = 'proposal'
                await self._run_phase(f"Round {round_num}: Proposals", context, "proposal")
            else:
                context['phase'] = 'critique'
                await self._run_phase(f"Round {round_num}: Analysis", context, "critique")
            
            # Check for spawn requests every 2 rounds
            if round_num % 2 == 0 and round_num <= 6:
                await self._check_spawn_requests(context)
        
        # If we've gone through all rounds without consensus, note this
        if round_num >= self.max_rounds:
            print(f"\n{Fore.YELLOW}⏰ Reached maximum rounds ({self.max_rounds}). Proceeding to synthesis...{Style.RESET_ALL}")
            logger.info("REACHED MAXIMUM ROUNDS WITHOUT CONSENSUS")
        
        # Generate final synthesis
        print(f"\n{Fore.CYAN}🔬 Generating comprehensive final analysis...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}(Using extended token limit and timeout for detailed synthesis){Style.RESET_ALL}")
        if "20b" in str(getattr(self.llm, 'model_name', '')).lower():
            print(f"{Fore.YELLOW}Large model detected - this may take 2-3 minutes...{Style.RESET_ALL}")
        
        solution = await self._synthesize_solution(round_num, context)
        
        result = {
            'problem': problem,
            'solution': solution,
            'num_agents': len(self.agents),
            'num_rounds': round_num,
            'discussion_length': len(self.discussion_history),
            'had_arbitrator': self.has_arbitrator
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL SOLUTION:")
        logger.info(solution)
        logger.info(f"\nStatistics: {json.dumps(result, indent=2)}")
        logger.info(f"{'='*60}\n")
        
        return result
    
    async def _run_phase(self, phase_name: str, context: Dict, expected_type: str):
        """Run a single phase of discussion"""
        print(f"\n{Fore.YELLOW}--- {phase_name} ---{Style.RESET_ALL}")
        logger.info(f"\n--- {phase_name} ---")
        
        # Always run sequentially for better reliability
        responses = []
        
        print(f"{Fore.CYAN}Running agents sequentially...{Style.RESET_ALL}")
        for i, agent in enumerate(self.agents):
            print(f"{Fore.CYAN}Agent {i+1}/{len(self.agents)}: {agent.perspective.role}...{Style.RESET_ALL}", end='', flush=True)
            try:
                response = await agent.generate_response(context, self.discussion_history)
                responses.append(response)
                print(f" {Fore.GREEN}✓{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Agent {agent.perspective.role} failed: {e}")
                print(f" {Fore.RED}✗{Style.RESET_ALL}")
                # Create a fallback response
                fallback = Message(
                    sender=agent.perspective.role,
                    content=f"[Unable to generate response - technical issue]",
                    message_type=MessageType.STATEMENT,
                    round_num=context['round']
                )
                responses.append(fallback)
        
        # Add to history and display
        for response in responses:
            if isinstance(response, Message):
                self.discussion_history.append(response)
                self._display_message(response)
        
        # Summarize phase
        await self._summarize_phase(phase_name)
    
    def _display_message(self, message: Message):
        """Display a message with appropriate formatting"""
        color = {
            MessageType.PROPOSAL: Fore.GREEN,
            MessageType.CRITIQUE: Fore.RED,
            MessageType.SUPPORT: Fore.BLUE,
            MessageType.QUESTION: Fore.MAGENTA,
            MessageType.STATEMENT: Fore.WHITE,
            MessageType.SPAWN_REQUEST: Fore.YELLOW,
            MessageType.ARBITRATION: Fore.CYAN
        }.get(message.message_type, Fore.WHITE)
        
        icon = {
            MessageType.PROPOSAL: "💡",
            MessageType.CRITIQUE: "⚠️", 
            MessageType.SUPPORT: "👍",
            MessageType.QUESTION: "❓",
            MessageType.ARBITRATION: "⚖️"
        }.get(message.message_type, "💬")
        
        print(f"\n{color}{icon} {message.sender}:{Style.RESET_ALL}")
        # Truncate very long messages for display
        display_content = message.content
        if len(display_content) > 1000:
            display_content = display_content[:1000] + "\n[... content truncated for display ...]"
        print(f"{display_content}\n")
    
    async def _summarize_phase(self, phase_name: str):
        """Generate and display a summary of the phase"""
        recent_messages = self.discussion_history[-len(self.agents):]
        
        if not recent_messages:
            return
        
        # Create a focused summary prompt
        summary_prompt = f"""Briefly summarize the key points from this discussion phase in 2-3 sentences.

Original Problem: {self.original_problem}

Phase: {phase_name}

Recent messages:
{chr(10).join(f"{msg.sender}: {msg.content[:200]}..." for msg in recent_messages)}

Focus on how the agents addressed the original problem. Summary:"""
        
        summary = await self.llm.generate(summary_prompt, temperature=0.5, min_length=50)
        print(f"\n{Fore.CYAN}📋 Orchestrator Summary: {summary}{Style.RESET_ALL}")
        logger.info(f"Orchestrator Summary: {summary}")
    
    async def _check_consensus(self) -> bool:
        """Improved consensus detection - more conservative"""
        # Need at least 3 rounds and substantial discussion
        if len(self.discussion_history) < len(self.agents) * 3 or self.discussion_history[-1].round_num < 3:
            return False
        
        recent_messages = self.discussion_history[-len(self.agents)*2:]
        
        # Look for actual convergence indicators - be more strict
        consensus_prompt = f"""Analyze these recent messages to determine if the experts have reached a STRONG consensus on the technical question.

Original Problem: {self.original_problem}

Recent messages:
{chr(10).join(f"{msg.sender}: {msg.content[:300]}..." for msg in recent_messages)}

Look for STRONG indicators of consensus:
1. Explicit agreement statements between experts
2. No significant contradictions or disagreements remaining
3. Convergence on specific technical conclusions
4. Resolution of all major technical debates

Only respond "YES" if there is CLEAR, STRONG consensus with no significant disagreements remaining.
Otherwise respond "NO" if there are still different viewpoints or unresolved technical questions.

Response:"""
        
        response = await self.llm.generate(consensus_prompt, temperature=0.3, min_length=3)
        consensus_reached = "YES" in response.upper()
        
        logger.info(f"Consensus check: {response.strip()} -> {consensus_reached}")
        return consensus_reached
    
    async def _check_deadlock(self) -> bool:
        """Check if discussion is deadlocked"""
        if len(self.discussion_history) < len(self.agents) * 3:
            return False
        
        recent_messages = self.discussion_history[-len(self.agents)*2:]
        
        # Look for signs of circular arguments or lack of progress
        deadlock_prompt = f"""Are these experts stuck in a deadlock with circular arguments and no progress?

Recent discussion:
{chr(10).join(f"{msg.sender}: {msg.content[:200]}..." for msg in recent_messages)}

Look for:
- Repeated arguments without new insights
- Circular disagreements
- No movement toward resolution

Respond with only "YES" or "NO":"""
        
        response = await self.llm.generate(deadlock_prompt, temperature=0.3, min_length=3)
        return "YES" in response.upper()
    
    async def _spawn_arbitrator(self, context: Dict):
        """Spawn an arbitrator agent to help resolve deadlock"""
        print(f"\n{Fore.CYAN}⚖️  Deadlock detected - spawning arbitrator...{Style.RESET_ALL}")
        logger.info("SPAWNING ARBITRATOR DUE TO DEADLOCK")
        
        arbitrator_perspective = AgentPerspective(
            role="Technical Arbitrator",
            concerns=["Resolving technical disagreements", "Finding balanced solutions", "Ensuring accuracy"],
            success_criteria=["Technical consensus", "Practical resolution", "Clear conclusions"],
            personality_traits="Objective, decisive, and technically rigorous",
            is_arbitrator=True
        )
        
        arbitrator = PerspectiveAgent(arbitrator_perspective, self.llm)
        self.agents.append(arbitrator)
        self.has_arbitrator = True
        self.spawned_count += 1
        
        # Have arbitrator immediately provide input
        arbitration_msg = await arbitrator.generate_response(context, self.discussion_history)
        self.discussion_history.append(arbitration_msg)
        self._display_message(arbitration_msg)
    
    async def _check_spawn_requests(self, context: Dict):
        """Check if new agents should be spawned - simplified"""
        if len(self.agents) >= self.max_agents:
            return
        
        # Only spawn if there's a clear gap
        spawn_prompt = f"""Review this technical discussion about: {self.original_problem}

Current expert perspectives:
{chr(10).join(f"- {agent.perspective.role}" for agent in self.agents)}

Is there a critical missing technical expertise that would significantly improve the analysis?

If YES, respond with:
SPAWN: yes
ROLE: [specific missing expertise]

If NO, respond with:
SPAWN: no"""
        
        response = await self.llm.generate(spawn_prompt, temperature=0.6, min_length=10)
        
        should_spawn = False
        new_role = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith('spawn:') and 'yes' in line.lower():
                should_spawn = True
            elif line.startswith('ROLE:'):
                new_role = line[5:].strip()
        
        if should_spawn and new_role and len(self.agents) < self.max_agents:
            new_perspective = AgentPerspective(
                role=new_role,
                concerns=[f"Domain expertise in {new_role.lower()}", "Technical accuracy", "Practical considerations"],
                success_criteria=["Technical soundness", "Practical applicability"],
                personality_traits="Expert and methodical"
            )
            new_agent = PerspectiveAgent(new_perspective, self.llm)
            self.agents.append(new_agent)
            self.spawned_count += 1
            
            print(f"\n{Fore.YELLOW}🔄 New expert added: {new_role}{Style.RESET_ALL}")
            print(f"Total experts: {len(self.agents)}")
            
            logger.info(f"NEW AGENT SPAWNED: {new_role}")
    
    async def _synthesize_solution(self, round_num: int, context: Dict) -> str:
        """Generate final solution synthesis - IMPROVED to stay on topic"""
        
        # Extract key technical points from the discussion
        key_points = self._extract_key_technical_points()
        
        synthesis_prompt = f"""Create a comprehensive technical analysis based on this expert discussion.

ORIGINAL PROBLEM: {self.original_problem}

Expert perspectives that participated:
{chr(10).join(f"- {agent.perspective.role}: {', '.join(agent.perspective.concerns[:2])}" for agent in self.agents)}

Key technical points from the discussion:
{key_points}

Discussion Status: {"Strong consensus reached after " + str(round_num) + " rounds" if round_num < self.max_rounds else "All perspectives explored over " + str(round_num) + " rounds"}

Your task:
1. Directly answer the ORIGINAL PROBLEM stated above
2. Synthesize the expert insights into a coherent technical explanation
3. Address any quantitative aspects mentioned (equations, measurements, percentages)
4. If there were disagreements, explain different viewpoints and which is most supported
5. Provide specific technical conclusions with reasoning
6. Include practical implications or applications

Create a thorough analysis (aim for 800-1200 words). Stay focused on the original problem.

Technical Analysis:"""
        
        # Use higher token limit for final synthesis - this is the main deliverable
        old_max_tokens = self.llm.max_tokens
        self.llm.max_tokens = min(3000, old_max_tokens * 4)  # 4x the normal limit or 3000, whichever is smaller
        
        try:
            synthesis = await self.llm.generate(synthesis_prompt, temperature=0.6, min_length=300)
        except Exception as e:
            logger.error(f"Final synthesis generation failed: {e}")
            # Fallback to shorter synthesis with normal token limit
            self.llm.max_tokens = old_max_tokens
            fallback_prompt = f"""Create a technical summary of this expert discussion.

ORIGINAL PROBLEM: {self.original_problem}

Key findings from {len(self.agents)} experts over {round_num} rounds:
{key_points}

Provide a concise technical answer to the original problem:"""
            
            try:
                synthesis = await self.llm.generate(fallback_prompt, temperature=0.6, min_length=150)
            except Exception as e2:
                logger.error(f"Fallback synthesis also failed: {e2}")
                synthesis = f"""Technical Analysis Summary

Based on the multi-agent discussion with {len(self.agents)} experts over {round_num} rounds:

The expert analysis addressed the original problem: {self.original_problem}

Key findings and consensus points from the discussion have been documented in the conversation logs. The experts provided technical insights from their respective domains and reached conclusions through collaborative analysis.

[Note: Full synthesis generation encountered technical difficulties - detailed findings are available in the conversation logs]"""
        finally:
            # Restore original token limit
            self.llm.max_tokens = old_max_tokens
        
        return synthesis
    
    def _extract_key_technical_points(self) -> str:
        """Extract the most important technical points from the discussion"""
        # Group messages by type and importance
        technical_statements = []
        proposals = []
        key_insights = []
        
        for msg in self.discussion_history:
            content_snippet = msg.content[:400] + ("..." if len(msg.content) > 400 else "")
            
            if msg.message_type == MessageType.PROPOSAL:
                proposals.append(f"• {msg.sender}: {content_snippet}")
            elif len(msg.content) > 200:  # Substantial technical content
                technical_statements.append(f"• {msg.sender}: {content_snippet}")
        
        result = []
        
        if proposals:
            result.append("Key Proposals and Solutions:")
            result.extend(proposals[:4])  # Top 4 proposals
            result.append("")
        
        if technical_statements:
            result.append("Technical Analysis:")
            result.extend(technical_statements[:6])  # Top 6 technical points
        
        return "\n".join(result) if result else "Discussion covered various technical aspects of the problem."

class AgenticProblemSolver:
    """Main solver class that coordinates the entire process"""
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.decomposer = ProblemDecomposer(self.llm)
    
    async def solve(self, problem: str, save_json: bool = True) -> Dict:
        """Solve a problem using multi-agent discussion"""
        start_time = datetime.now()
        
        try:
            # 1. Decompose the problem
            print(f"\n{Fore.CYAN}Analyzing problem complexity and identifying perspectives...{Style.RESET_ALL}")
            perspectives = await self.decomposer.analyze_problem(problem)
            
            # 2. Create agents
            agents = []
            print(f"\n{Fore.GREEN}Creating {len(perspectives)} initial agents:{Style.RESET_ALL}")
            for perspective in perspectives:
                agent = PerspectiveAgent(perspective, self.llm)
                agents.append(agent)
                print(f"  • {perspective.role}")
                print(f"    Concerns: {', '.join(perspective.concerns[:2])}...")
            
            # 3. Run orchestrated discussion
            orchestrator = DebateOrchestrator(self.llm)
            result = await orchestrator.facilitate_discussion(problem, agents)
            
            # 4. Prepare final result
            end_time = datetime.now()
            result['duration'] = str(end_time - start_time)
            result['spawned_agents'] = orchestrator.spawned_count
            result['initial_agents'] = len(perspectives)
            
            # 5. Save JSON summary if requested
            if save_json:
                json_filename = f"debate_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(json_filename, 'w') as f:
                    summary_data = {
                        'result': result,
                        'agents': [
                            {
                                'role': agent.perspective.role,
                                'concerns': agent.perspective.concerns,
                                'success_criteria': agent.perspective.success_criteria,
                                'is_arbitrator': agent.perspective.is_arbitrator
                            }
                            for agent in orchestrator.agents
                        ],
                        'key_messages': [
                            {
                                'sender': msg.sender,
                                'type': msg.message_type.value,
                                'round': msg.round_num,
                                'preview': msg.content[:300] + "..."
                            }
                            for msg in orchestrator.discussion_history
                            if len(msg.content) > 100  # Only substantial messages
                        ]
                    }
                    json.dump(summary_data, f, indent=2)
                print(f"\n{Fore.GREEN}Summary saved to: {json_filename}{Style.RESET_ALL}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in problem solving: {e}")
            raise

async def main():
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"   MULTI-AGENT PROBLEM SOLVER")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    # Setup LLM provider
    try:
        llm_provider = get_llm_provider()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Setup cancelled by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Failed to initialize provider: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Get problem statement
    print(f"\n{Fore.CYAN}Enter your problem statement:{Style.RESET_ALL}")
    print("(Press Enter twice when done)")
    
    lines = []
    try:
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                if lines:
                    break
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Input cancelled by user{Style.RESET_ALL}")
        sys.exit(0)
    
    problem = ' '.join(lines)
    problem = sanitize_input(problem)  # Sanitize input
    
    if not problem.strip():
        print(f"{Fore.RED}Error: No problem statement provided{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"\n{Fore.YELLOW}Full conversation will be logged to: {conversation_log_file}{Style.RESET_ALL}")
    
    # Confirm before starting
    print(f"\n{Fore.CYAN}Ready to start multi-agent discussion.{Style.RESET_ALL}")
    print("Press Ctrl+C at any time to interrupt")
    input("Press Enter to begin...")
    
    # Create solver and run
    solver = AgenticProblemSolver(llm_provider)
    
    try:
        result = await solver.solve(problem)
        
        # Display final result
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"FINAL SOLUTION")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        print(result['solution'])
        
        print(f"\n{Fore.GREEN}Summary Statistics:{Style.RESET_ALL}")
        print(f"  • Initial agents: {result['initial_agents']}")
        print(f"  • Total agents: {result['num_agents']} ({result['spawned_agents']} spawned)")
        print(f"  • Arbitrator used: {'Yes' if result['had_arbitrator'] else 'No'}")
        print(f"  • Discussion rounds: {result['num_rounds']}")
        print(f"  • Total messages: {result['discussion_length']}")
        print(f"  • Duration: {result['duration']}")
        
        print(f"\n{Fore.YELLOW}Logs saved:{Style.RESET_ALL}")
        print(f"  • Full conversation: {conversation_log_file}")
        print(f"  • JSON summary: debate_summary_*.json")
        print(f"  • Config saved: .agent_solver_config.json")
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Discussion interrupted by user{Style.RESET_ALL}")
        print(f"Partial logs saved to: {conversation_log_file}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        logger.exception("Fatal error in problem solver")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())