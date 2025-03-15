import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class ArithmeticCoder:
    def __init__(self, model_name="EleutherAI/pythia-160m", device="cpu", precision=12):
        """
        Initialize the arithmetic coder with a small language model.
        
        Args:
            model_name: HuggingFace model name (smaller models work better on Raspberry Pi)
            device: 'cpu' or 'cuda'
            precision: Number of bits of precision for the arithmetic coder (higher = more accurate but more compute)
        """
        self.device = device
        self.precision = precision
        self.max_val = 2**precision
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        ).to(device).eval()
        
        # Apply 4-bit quantization (further optimization for resource-limited environments)
        if device == "cpu":
            self.quantize_model()
    
    def quantize_model(self):
        """Apply dynamic quantization to the model for CPU efficiency."""
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("Model quantized to 8-bit for CPU inference")
        except Exception as e:
            print(f"Quantization failed: {e}. Using model as-is.")
    
    def text_to_bits(self, text):
        """Convert text to a list of bits."""
        # Convert text to bytes, then to bits
        bytes_data = text.encode('utf-8')
        bits = []
        for b in bytes_data:
            for i in range(8):
                bits.append((b >> i) & 1)
        return bits
    
    def bits_to_text(self, bits):
        """Convert a list of bits back to text."""
        # Convert bits to bytes
        bytes_data = bytearray()
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = 0
                for j in range(8):
                    if bits[i + j]:
                        byte |= 1 << j
                bytes_data.append(byte)
        
        # Convert bytes to text
        try:
            return bytes_data.decode('utf-8')
        except UnicodeDecodeError:
            # Handle potential incomplete UTF-8 sequences
            return bytes_data.decode('utf-8', errors='replace')
    
    def bits_to_int(self, bits):
        """Convert a list of bits to an integer."""
        val = 0
        for bit in bits:
            val = (val << 1) | bit
        return val
    
    def int_to_bits(self, value, num_bits):
        """Convert an integer to a list of bits with specified length."""
        bits = []
        for _ in range(num_bits):
            bits.append(value & 1)
            value >>= 1
        return list(reversed(bits))  # MSB first
    
    def num_same_from_beginning(self, bits1, bits2):
        """Count the number of matching bits from the beginning of two lists."""
        count = 0
        for b1, b2 in zip(bits1, bits2):
            if b1 == b2:
                count += 1
            else:
                break
        return count
    
    def encode(self, message_bits, prompt, temp=0.8, max_tokens=200):
        """
        Encode a binary message into text using arithmetic coding.
        
        Args:
            message_bits: List of bits to encode
            prompt: Starting text prompt to condition the LM
            temp: Temperature for sampling (lower = more deterministic, higher = more random)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text with encoded message, statistics
        """
        if isinstance(message_bits, str):
            message_bits = self.text_to_bits(message_bits)
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Initialize arithmetic coding state
        cur_interval = [0, self.max_val]  # [low, high) - high is exclusive
        
        generated = input_ids[0].tolist()
        
        i = 0  # Index into message_bits
        stats = {
            "bits_encoded": 0,
            "tokens_generated": 0,
            "bits_per_token": 0
        }
        
        # Generate text token by token
        with torch.no_grad():
            past = None
            
            while i < len(message_bits) and len(generated) - len(input_ids[0]) < max_tokens:
                # Get model predictions for next token
                if past is None:
                    outputs = self.model(input_ids)
                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values
                else:
                    # More efficient generation using past key values
                    current_token = torch.tensor([[generated[-1]]]).to(self.device)
                    outputs = self.model(current_token, past_key_values=past)
                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values
                
                # Apply temperature
                logits = logits / temp
                
                # Get probabilities for next token
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                
                # Calculate current interval range
                cur_range = cur_interval[1] - cur_interval[0]
                
                # Sort tokens by probability (descending)
                sorted_indices = np.argsort(-probs)
                sorted_probs = probs[sorted_indices]
                
                # Keep only tokens with non-negligible probability
                threshold = 1.0 / cur_range
                k = min(100, max(2, np.sum(sorted_probs > threshold)))
                
                # Scale and quantize probabilities to integers
                p = sorted_probs[:k]
                p = p / np.sum(p) * cur_range
                p = np.round(p).astype(np.int64)
                
                # Ensure probabilities sum to current range
                if np.sum(p) > cur_range:
                    # If we rounded up too much, reduce highest probabilities
                    while np.sum(p) > cur_range:
                        idx = np.argmax(p)
                        p[idx] -= 1
                
                if np.sum(p) < cur_range:
                    # If we rounded down too much, add to highest probability
                    p[0] += cur_range - np.sum(p)
                
                # Calculate cumulative probabilities
                cum_p = np.cumsum(p)
                
                # Determine which subinterval to choose based on message bits
                message_val = 0
                for j in range(min(self.precision, len(message_bits) - i)):
                    message_val = (message_val << 1) | message_bits[i + j]
                    
                message_val = message_val << (self.precision - min(self.precision, len(message_bits) - i))
                target = cur_interval[0] + message_val
                
                # Find which interval contains our target
                token_idx = np.searchsorted(cum_p + cur_interval[0], target, side='right')
                if token_idx == 0:  # Handle edge case
                    token_idx = 1
                    
                # The token we'll output
                next_token = sorted_indices[token_idx - 1]
                
                # Calculate new interval
                new_low = cur_interval[0] + (0 if token_idx == 1 else cum_p[token_idx - 2])
                new_high = cur_interval[0] + cum_p[token_idx - 1]
                
                # Calculate how many bits we can encode with this choice
                low_bits = self.int_to_bits(new_low, self.precision)
                high_bits = self.int_to_bits(new_high - 1, self.precision)  # -1 because high is exclusive
                
                bits_encoded = self.num_same_from_beginning(low_bits, high_bits)
                i += bits_encoded
                stats["bits_encoded"] += bits_encoded
                
                # Update interval
                cur_interval[0] = new_low
                cur_interval[1] = new_high
                
                # Add token to generated text
                generated.append(int(next_token))
                stats["tokens_generated"] += 1
                
                # Check if we've reached an end token
                if next_token == self.tokenizer.eos_token_id:
                    break
        
        stats["bits_per_token"] = stats["bits_encoded"] / max(1, stats["tokens_generated"])
        generated_text = self.tokenizer.decode(generated[len(input_ids[0]):])
        
        return generated_text, stats
    
    def decode(self, encoded_text, prompt, temp=0.8):
        """
        Decode a message hidden in text using arithmetic coding.
        
        Args:
            encoded_text: Text containing the hidden message
            prompt: The same prompt used during encoding
            temp: Same temperature used during encoding
            
        Returns:
            Extracted bit sequence, statistics
        """
        # Tokenize prompt and encoded text
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        encoded_ids = self.tokenizer.encode(encoded_text, return_tensors="pt").to(self.device)[0].tolist()
        
        # Initialize arithmetic coding state
        cur_interval = [0, self.max_val]
        
        extracted_bits = []
        stats = {
            "bits_decoded": 0,
            "tokens_processed": 0,
            "bits_per_token": 0
        }
        
        # Process tokens one by one
        with torch.no_grad():
            past = None
            
            for i, token_id in enumerate(encoded_ids):
                # Get model predictions for this position
                if i == 0:
                    outputs = self.model(prompt_ids)
                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values
                else:
                    current_token = torch.tensor([[encoded_ids[i-1]]]).to(self.device)
                    outputs = self.model(current_token, past_key_values=past)
                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values
                
                # Apply temperature
                logits = logits / temp
                
                # Get probabilities for next token
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                
                # Calculate current interval range
                cur_range = cur_interval[1] - cur_interval[0]
                
                # Sort tokens by probability (descending)
                sorted_indices = np.argsort(-probs)
                sorted_probs = probs[sorted_indices]
                
                # Keep only tokens with non-negligible probability
                threshold = 1.0 / cur_range
                k = min(100, max(2, np.sum(sorted_probs > threshold)))
                
                # Scale and quantize probabilities to integers
                p = sorted_probs[:k]
                p = p / np.sum(p) * cur_range
                p = np.round(p).astype(np.int64)
                
                # Ensure probabilities sum to current range
                if np.sum(p) > cur_range:
                    while np.sum(p) > cur_range:
                        idx = np.argmax(p)
                        p[idx] -= 1
                
                if np.sum(p) < cur_range:
                    p[0] += cur_range - np.sum(p)
                
                # Calculate cumulative probabilities
                cum_p = np.cumsum(p)
                
                # Find the rank of the actual token
                token_rank = np.where(sorted_indices == token_id)[0]
                if len(token_rank) == 0 or token_rank[0] >= k:
                    # Token is outside our pruned set - this is an error
                    print(f"Warning: Token {token_id} not in top-k. Decoding may be incorrect.")
                    token_rank = 0
                else:
                    token_rank = token_rank[0]
                
                # Calculate new interval based on the observed token
                new_low = cur_interval[0] + (0 if token_rank == 0 else cum_p[token_rank - 1])
                new_high = cur_interval[0] + cum_p[token_rank]
                
                # Extract bits that are now determined
                low_bits = self.int_to_bits(new_low, self.precision)
                high_bits = self.int_to_bits(new_high - 1, self.precision)
                
                bits_decoded = self.num_same_from_beginning(low_bits, high_bits)
                
                # The first identical bits are now fixed - extract them
                for j in range(bits_decoded):
                    extracted_bits.append(low_bits[j])
                
                stats["bits_decoded"] += bits_decoded
                stats["tokens_processed"] += 1
                
                # Update interval for next iteration
                cur_interval[0] = new_low
                cur_interval[1] = new_high
                
                # Stop if we hit the end token
                if token_id == self.tokenizer.eos_token_id:
                    break
        
        stats["bits_per_token"] = stats["bits_decoded"] / max(1, stats["tokens_processed"])
        
        return extracted_bits, stats
                
    def encode_message(self, message, prompt, **kwargs):
        """Encode a text message into steganographic text."""
        message_bits = self.text_to_bits(message)
        return self.encode(message_bits, prompt, **kwargs)
    
    def decode_message(self, encoded_text, prompt, expected_bits=None, **kwargs):
        """Decode a message from steganographic text."""
        bits, stats = self.decode(encoded_text, prompt, **kwargs)
        
        # If we know how many bits to expect, truncate
        if expected_bits:
            bits = bits[:expected_bits]
            
        # Convert bits back to text
        message = self.bits_to_text(bits)
        return message, stats

# Example usage
def example():
    # Initialize with a small model suitable for Raspberry Pi
    coder = ArithmeticCoder(model_name="EleutherAI/pythia-160m", precision=10)
    
    # Message to hide
    message = "This is a secret message!"
    
    # Prompt to start generation
    prompt = "Today I learned something interesting about deep learning:"
    
    # Encode the message
    print(f"Encoding message: {message}")
    encoded_text, encode_stats = coder.encode_message(message, prompt, temp=0.8, max_tokens=100)
    print(f"Generated text: {encoded_text}")
    print(f"Encoding stats: {encode_stats}")
    
    # Decode the message
    decoded_message, decode_stats = coder.decode_message(encoded_text, prompt, temp=0.8)
    print(f"Decoded message: {decoded_message}")
    print(f"Decoding stats: {decode_stats}")
    
    # Verify success
    print(f"Success: {message == decoded_message}")

if __name__ == "__main__":
    example()
