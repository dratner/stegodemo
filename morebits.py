import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import math

class MultiBitStego:
    """A steganography system that encodes multiple bits per token for better efficiency."""

    def __init__(self, model_name="gpt2", device="cpu", bits_per_token=2):
        """
        Initialize with a small language model.

        Args:
            model_name: The name of the model to use
            device: The device to run the model on
            bits_per_token: How many bits to encode per token (1-3 recommended)
        """
        self.device = device
        self.bits_per_token = bits_per_token

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

    def bits_to_partition(self, bits, num_partitions):
        """Convert a group of bits to a partition index."""
        # Convert binary bits to an integer
        value = 0
        for i, bit in enumerate(bits):
            value |= bit << i

        # Ensure the value is within range
        return value % num_partitions

    def partition_to_bits(self, partition, num_bits):
        """Convert a partition index back to bits."""
        bits = []
        for i in range(num_bits):
            bits.append((partition >> i) & 1)
        return bits

    def encode(self, message_bits, prompt, temp=1.0, top_k=32, max_tokens=200):
        """
        Encode a binary message into text using multiple bits per token.

        Args:
            message_bits: List of bits to encode
            prompt: Starting text prompt
            temp: Temperature for sampling
            top_k: Number of top tokens to consider
            max_tokens: Maximum number of tokens to generate
        """
        if isinstance(message_bits, str):
            message_bits = self.text_to_bits(message_bits)

        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        generated = input_ids[0].tolist()
        stats = {"bits_encoded": 0, "tokens_generated": 0}

        # Calculate number of partitions based on bits per token
        num_partitions = 2 ** self.bits_per_token

        # Generate text token by token
        with torch.no_grad():
            past = None

            # Process message_bits in chunks of bits_per_token
            bit_index = 0
            while bit_index < len(message_bits) and len(generated) - len(input_ids[0]) < max_tokens:
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

                # Determine how many bits we can encode with this token
                bits_to_encode = min(self.bits_per_token, len(message_bits) - bit_index)
                if bits_to_encode < self.bits_per_token:
                    # We're at the end of the message, adjust partitions
                    num_current_partitions = 2 ** bits_to_encode
                else:
                    num_current_partitions = num_partitions

                # Get current bits to encode
                current_bits = message_bits[bit_index:bit_index + bits_to_encode]

                # Convert bits to partition index
                partition_idx = self.bits_to_partition(current_bits, num_current_partitions)

                # Divide the top_k tokens into num_partitions groups
                partition_size = len(top_tokens) // num_current_partitions

                # Select the partition corresponding to our bits
                start_idx = partition_idx * partition_size
                end_idx = (partition_idx + 1) * partition_size if partition_idx < num_current_partitions - 1 else len(top_tokens)
                token_pool = top_tokens[start_idx:end_idx]

                # Sample from the appropriate pool
                token_probs = probs[token_pool]
                token_probs = token_probs / token_probs.sum()  # Renormalize
                next_token_idx = np.random.choice(len(token_pool), p=token_probs)
                next_token = token_pool[next_token_idx]

                # Add token to generated text
                generated.append(int(next_token))

                # Update stats
                stats["bits_encoded"] += bits_to_encode
                stats["tokens_generated"] += 1
                bit_index += bits_to_encode

                # Check if we've reached an end token
                if next_token == self.tokenizer.eos_token_id:
                    break

        stats["bits_per_token"] = stats["bits_encoded"] / max(1, stats["tokens_generated"])
        generated_text = self.tokenizer.decode(generated[len(input_ids[0]):])

        return generated_text, stats

    def decode(self, encoded_text, prompt, expected_bits=None):
        """
        Decode a message hidden in text using multiple bits per token.
        """
        # Tokenize prompt and encoded text
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        encoded_ids = self.tokenizer.encode(encoded_text, return_tensors="pt").to(self.device)[0].tolist()

        # Calculate number of partitions based on bits per token
        num_partitions = 2 ** self.bits_per_token

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

                # Determine how many bits we can decode from this token
                if expected_bits and len(extracted_bits) + self.bits_per_token > expected_bits:
                    # We're at the end of the message
                    bits_remaining = expected_bits - len(extracted_bits)
                    num_current_partitions = 2 ** bits_remaining
                else:
                    bits_remaining = self.bits_per_token
                    num_current_partitions = num_partitions

                # Take top k tokens
                top_k = min(32, len(sorted_indices))  # Make sure we don't exceed the size
                top_tokens = sorted_indices[:top_k]

                # Find which partition the token belongs to
                partition_size = len(top_tokens) // num_current_partitions

                # Find the token's position
                token_pos = -1
                for j, t in enumerate(top_tokens):
                    if t.item() == token_id:
                        token_pos = j
                        break

                if token_pos == -1:
                    # Token not in top tokens - use closest match
                    distances = np.abs(sorted_indices - token_id)
                    closest_idx = np.argmin(distances)
                    token_pos = closest_idx if closest_idx < len(top_tokens) else 0

                # Determine which partition it belongs to
                partition_idx = min(token_pos // partition_size, num_current_partitions - 1)

                # Convert partition to bits
                partition_bits = self.partition_to_bits(partition_idx, bits_remaining)
                extracted_bits.extend(partition_bits)

                # Update stats
                stats["bits_decoded"] += bits_remaining
                stats["tokens_processed"] += 1

                # Stop if we've decoded all expected bits
                if expected_bits and len(extracted_bits) >= expected_bits:
                    break

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
        bits, stats = self.decode(encoded_text, prompt, expected_bits, **kwargs)

        # If we know how many bits to expect, truncate
        if expected_bits:
            bits = bits[:expected_bits]

        # Convert bits back to text
        message = self.bits_to_text(bits)
        return message, stats

# Example usage
def example():
    # Initialize with GPT-2 Small and 2 bits per token
    stego = MultiBitStego(model_name="gpt2", bits_per_token=4)

    # Message to hide
    message = "This is a secret message!"

    # Better prompt for more natural text
    prompt = "Sergio's Italian Grille is the newest restaurant in downtown Austin."

    # Encode the message
    print(f"Encoding message: {message}")
    encoded_text, encode_stats = stego.encode_message(
        message,
        prompt,
        temp=1.1,  # Higher temperature for more natural text
        top_k=32,  # More tokens for better partitioning
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
        expected_bits=expected_bits
    )
    print(f"Decoded message: {decoded_message}")
    print(f"Decoding stats: {decode_stats}")

    # Verify success
    print(f"Success: {message == decoded_message}")

if __name__ == "__main__":
    example()
