import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import requests
import json

class LLMInterface:
    """
    Interface for interacting with different language models.
    Supports OpenAI API and local models like Ollama.
    """
    
    def __init__(self, 
                 model_type: str = "openai",
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the LLM interface.
        
        Args:
            model_type: Type of model ('openai', 'ollama', or 'local')
            model_name: Name of the specific model to use
            api_key: API key for OpenAI (if using OpenAI)
            base_url: Base URL for local models (if using local)
            temperature: Sampling temperature for generation
            max_tokens: Maximum number of tokens to generate
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        # Initialize the appropriate client
        if self.model_type == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                self.logger.warning("No OpenAI API key provided. Using fallback local model.")
                self.model_type = "local"
                self._init_local_client()
            else:
                self.client = OpenAI(api_key=self.api_key)
                self.logger.info(f"Initialized OpenAI client with model: {model_name}")
        
        elif self.model_type == "ollama":
            self.base_url = base_url or "http://localhost:11434"
            self._init_ollama_client()
            
        else:  # local/fallback model
            self._init_local_client()
    
    def _init_ollama_client(self):
        """Initialize Ollama client and check if model is available."""
        try:
            # Test connection to Ollama
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = response.json().get('models', [])
                model_names = [model['name'] for model in available_models]
                
                if self.model_name not in model_names:
                    self.logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                    # Use first available model as fallback
                    if model_names:
                        self.model_name = model_names[0]
                        self.logger.info(f"Using fallback model: {self.model_name}")
                    else:
                        raise Exception("No models available in Ollama")
                
                self.logger.info(f"Initialized Ollama client with model: {self.model_name}")
            else:
                raise Exception("Ollama server not accessible")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {str(e)}")
            self.logger.info("Falling back to local model")
            self._init_local_client()
    
    def _init_local_client(self):
        """Initialize local/fallback model client."""
        self.model_type = "local"
        self.model_name = "local_fallback"
        self.logger.info("Using local fallback model (rule-based responses)")
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         system_message: Optional[str] = None) -> str:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: User prompt/question
            context: Additional context to include
            system_message: System message to guide the model
            
        Returns:
            Generated response
        """
        try:
            if self.model_type == "openai":
                return self._generate_openai_response(prompt, context, system_message)
            elif self.model_type == "ollama":
                return self._generate_ollama_response(prompt, context, system_message)
            else:
                return self._generate_local_response(prompt, context)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(prompt, context)
    
    def _generate_openai_response(self, 
                                 prompt: str, 
                                 context: Optional[str] = None,
                                 system_message: Optional[str] = None) -> str:
        """Generate response using OpenAI API."""
        messages = []
        
        # Add system message
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({
                "role": "system", 
                "content": "You are a knowledgeable herbalist and expert in traditional medicine. Provide helpful, accurate information about herbs and natural remedies. Always remind users to consult healthcare professionals for medical conditions."
            })
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system", 
                "content": f"Use the following information from a herbalism reference book to answer the user's question:\n\n{context}"
            })
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_ollama_response(self, 
                                 prompt: str, 
                                 context: Optional[str] = None,
                                 system_message: Optional[str] = None) -> str:
        """Generate response using Ollama."""
        # Construct the full prompt
        full_prompt = ""
        
        if system_message:
            full_prompt += f"System: {system_message}\n\n"
        else:
            full_prompt += "System: You are a knowledgeable herbalist and expert in traditional medicine. Provide helpful, accurate information about herbs and natural remedies. Always remind users to consult healthcare professionals for medical conditions.\n\n"
        
        if context:
            full_prompt += f"Reference Information:\n{context}\n\n"
        
        full_prompt += f"Question: {prompt}\n\nAnswer:"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def _generate_local_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using local rule-based system."""
        # Simple rule-based responses for common herbalism queries
        prompt_lower = prompt.lower()
        
        # Knowledge base of basic herbalism information
        herb_info = {
            'chamomile': "Chamomile is known for its calming properties and is often used to promote sleep and reduce anxiety. It can be prepared as a tea by steeping dried flowers in hot water.",
            'echinacea': "Echinacea is commonly used to support immune system function and may help reduce the duration of cold symptoms. It's available as teas, tinctures, and supplements.",
            'ginger': "Ginger is excellent for digestive issues, nausea, and motion sickness. Fresh ginger can be made into tea or added to foods for its medicinal benefits.",
            'turmeric': "Turmeric contains curcumin, which has anti-inflammatory properties. It's traditionally used for joint health and digestive support.",
            'lavender': "Lavender is known for its relaxing and calming effects. It's often used in aromatherapy and can help with sleep and stress relief.",
            'peppermint': "Peppermint is excellent for digestive issues, headaches, and respiratory problems. Peppermint tea is a common preparation.",
            'garlic': "Garlic has antimicrobial properties and supports cardiovascular health. It can be used fresh in cooking or as supplements.",
            'willow bark': "Willow bark contains salicin, a compound similar to aspirin. It's traditionally used for pain relief and reducing inflammation."
        }
        
        # Check for herb-specific queries
        for herb, info in herb_info.items():
            if herb in prompt_lower:
                response = f"Based on traditional herbalism knowledge: {info}"
                if context:
                    response += f"\n\nAdditional information from the reference text:\n{context[:500]}..."
                response += "\n\n⚠️ Please consult with a healthcare professional before using herbs for medical conditions."
                return response
        
        # General responses for common query types
        if any(word in prompt_lower for word in ['sleep', 'insomnia', 'rest']):
            response = "For sleep issues, traditional herbs include chamomile, lavender, valerian root, and passionflower. These can be prepared as teas or used in aromatherapy."
        elif any(word in prompt_lower for word in ['digestion', 'stomach', 'nausea']):
            response = "For digestive issues, consider ginger, peppermint, chamomile, or fennel. Ginger tea is particularly effective for nausea."
        elif any(word in prompt_lower for word in ['headache', 'pain', 'ache']):
            response = "For headaches and pain, traditional remedies include willow bark, feverfew, and peppermint. Cold or warm compresses with essential oils can also help."
        elif any(word in prompt_lower for word in ['immune', 'cold', 'flu']):
            response = "For immune support, consider echinacea, elderberry, garlic, and ginger. Honey and lemon in warm water is also beneficial."
        else:
            response = "I'd be happy to help with your herbalism question! Could you be more specific about which herbs or conditions you're interested in learning about?"
        
        if context:
            response += f"\n\nRelevant information from the herbalism reference:\n{context[:500]}..."
        
        response += "\n\n⚠️ This information is for educational purposes only. Always consult with healthcare professionals before using herbal remedies for medical conditions."
        return response
    
    def _generate_fallback_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a fallback response when all other methods fail."""
        response = "I apologize, but I'm currently experiencing technical difficulties with the language model. "
        
        if context:
            response += f"However, I found some relevant information from the herbalism reference that might help:\n\n{context[:600]}..."
        else:
            response += "Please try rephrasing your question or check back later."
        
        response += "\n\n⚠️ For any health-related concerns, please consult with qualified healthcare professionals."
        return response
    
    def check_model_availability(self) -> bool:
        """
        Check if the configured model is available and working.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            test_response = self.generate_response("Hello", system_message="Respond with 'Available'")
            return len(test_response) > 0
        except Exception as e:
            self.logger.error(f"Model availability check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'available': self.check_model_availability()
        }
    
    def set_parameters(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Update model parameters.
        
        Args:
            temperature: New temperature value
            max_tokens: New max tokens value
        """
        if temperature is not None:
            self.temperature = temperature
            self.logger.info(f"Updated temperature to: {temperature}")
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.logger.info(f"Updated max_tokens to: {max_tokens}")