import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class SimpleStego:
    """A reliable steganography system for resource-constrained environments."""

    def __init__(self, model_name="gpt2", device="cpu"):
        """Initialize with a small language model."""
        self.device = device

        # Load tokenizer
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        print(f"Loading model {model_name}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(device).eval()
        except:
            # Fall back to simpler loading if needed
            print("Falling back to simpler model loading...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None
            ).to(device).eval()

    def text_to_bits(self, text):
        """Convert text to a list of bits."""
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
            return bytes_data.decode('utf-8', errors='replace')

    def encode(self, message_bits, prompt, temp=1.0, top_k=10, max_tokens=200):
        """
        Encode a binary message into text using a simple binary partition approach.
        Each token encodes exactly 1 bit.
        """
        if isinstance(message_bits, str):
            message_bits = self.text_to_bits(message_bits)

        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        generated = input_ids[0].tolist()
        stats = {"bits_encoded": 0, "tokens_generated": 0}

        # Generate text token by token
        with torch.no_grad():
            past = None

            for i in range(min(len(message_bits), max_tokens)):
                # Get model predictions for next token
                if past is None:
                    outputs = self.model(input_ids)
                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values
                else:
                    current_token = torch.tensor([[generated[-1]]]).to(self.device)
                    outputs = self.model(current_token, past_key_values=past)
                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values

                # Apply temperature
                logits = logits / temp

                # Get probabilities and sort them
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                sorted_indices = np.argsort(-probs)

                # Take top k tokens to ensure text still makes sense
                top_tokens = sorted_indices[:top_k]

                # Simply split the top tokens in half
                mid_point = len(top_tokens) // 2

                # Choose token based on current bit
                bit = message_bits[i]
                if bit == 0:
                    # Use first half
                    token_pool = top_tokens[:mid_point]
                else:
                    # Use second half
                    token_pool = top_tokens[mid_point:]

                # Sample from the appropriate pool
                token_probs = probs[token_pool]
                token_probs = token_probs / token_probs.sum()  # Renormalize
                next_token_idx = np.random.choice(len(token_pool), p=token_probs)
                next_token = token_pool[next_token_idx]

                # Add token to generated text
                generated.append(int(next_token))
                stats["bits_encoded"] += 1
                stats["tokens_generated"] += 1

                # Check if we've reached an end token
                if next_token == self.tokenizer.eos_token_id:
                    break

        stats["bits_per_token"] = 1.0  # Always 1 bit per token in this approach
        generated_text = self.tokenizer.decode(generated[len(input_ids[0]):])

        return generated_text, stats

    def decode(self, encoded_text, prompt, top_k=10):
        """
        Decode a message hidden in text using the simple binary partition approach.
        """
        # Tokenize prompt and encoded text
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        encoded_ids = self.tokenizer.encode(encoded_text, return_tensors="pt").to(self.device)[0].tolist()

        extracted_bits = []
        stats = {"bits_decoded": 0, "tokens_processed": 0}

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

                # Get probabilities and sort them
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                sorted_indices = np.argsort(-probs)

                # Take top k tokens
                top_tokens = sorted_indices[:top_k]

                # Use the same deterministic split as during encoding
                mid_point = len(top_tokens) // 2
                first_half = set(top_tokens[:mid_point].tolist())

                # Determine which set the actual token belongs to
                if token_id in first_half:
                    extracted_bits.append(0)
                else:
                    # Check if token exists in second half, otherwise make an educated guess
                    if token_id in top_tokens[mid_point:]:
                        extracted_bits.append(1)
                    else:
                        # Token not in top k - use a heuristic
                        # Assign to group with closest token by index
                        closest_dist = float('inf')
                        closest_is_first_half = True

                        for idx, t in enumerate(top_tokens):
                            dist = abs(int(t) - int(token_id))
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_is_first_half = idx < mid_point

                        extracted_bits.append(0 if closest_is_first_half else 1)

                stats["bits_decoded"] += 1
                stats["tokens_processed"] += 1

                # Stop if we hit the end token
                if token_id == self.tokenizer.eos_token_id:
                    break

        stats["bits_per_token"] = 1.0

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
    # Initialize with GPT-2 Small
    stego = SimpleStego(model_name="gpt2")

    # Message to hide
    message = "This is a secret message!"

    # Better prompt for more natural text
    prompt = "Sergio's Italian Grille is the newest restaurant in downtown Austin."

    # Encode the message
    print(f"Encoding message: {message}")
    encoded_text, encode_stats = stego.encode_message(
        message,
        prompt,
        temp=1.0,  # Higher temperature for more natural text
        top_k=10,  # Smaller top_k for higher reliability
        max_tokens=200
    )
    print(f"Generated text: {encoded_text}")
    print(f"Encoding stats: {encode_stats}")

    # Calculate expected bits
    expected_bits = len(stego.text_to_bits(message))
    print(f"Expected bits: {expected_bits}")

    # Decode the message
    decoded_message, decode_stats = stego.decode_message(
        encoded_text,
        prompt,
        expected_bits=expected_bits,
        top_k=10  # Same top_k as encoding
    )
    print(f"Decoded message: {decoded_message}")
    print(f"Decoding stats: {decode_stats}")

    # Verify success
    print(f"Success: {message == decoded_message}")

if __name__ == "__main__":
    example()
