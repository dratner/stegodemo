# StegoGPT - Steganography with GPT-2

This project implements a steganography system that can hide encrypted text within language model output. It uses a lightweight GPT-2 implementation in C++ to encode and decode hidden messages within natural-looking text.

## Features

- Hide messages in natural-looking text using GPT-2
- Configurable bits-per-token for efficiency/quality tradeoff
- Pure C++ implementation with no external dependencies
- Works on CPU
- No Python or external dependencies required

## Building from Source

### Prerequisites

- CMake (3.14 or newer)
- C++17 compatible compiler (gcc, clang)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/stego_cpp.git
cd stego_cpp

# Option 1: Use the setup script
chmod +x setup.sh
./setup.sh

# Option 2: Manual build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Running the Application

```bash
# Run the application with default settings (2 bits per token)
./build/bin/stego_app

# Run with custom bits per token
./build/bin/stego_app 3  # Use 3 bits per token
```

## Usage

The application takes one optional argument:

```
Usage: ./stego_app [bits_per_token]
Example: ./stego_app 2
```

Where:
- `bits_per_token` is the number of bits to encode per token (default: 2)

When run, the application will:
1. Ask you to enter a message to hide
2. Ask you to enter a prompt text
3. Encode your message into generated text
4. Decode the message back from the generated text
5. Show success rate and statistics

Example output:
```
Initializing StegoGPT with 2 bits per token...
Enter a message to hide (press Enter to use default): Hello world!
Enter a prompt (press Enter to use default): 
Encoding message: Hello world!
Generated text: when back than many when on if been...
Encoding stats:
bits_encoded: 88
tokens_generated: 44
bits_per_token: 2
Expected bits: 88
Decoded message: Hello world!
Decoding stats:
tokens_processed: 44
bits_per_token: 2
bits_decoded: 88
Success: Yes
```

## How It Works

The steganography algorithm works by:

1. Converting the input message to binary bits
2. For each generated token:
   - Getting the language model's predictions for the next token
   - Taking the top-k most likely tokens (e.g., top 32)
   - Dividing these tokens into partitions (e.g., 4 partitions for 2 bits)
   - Selecting the partition that corresponds to the bits we want to encode
   - Sampling from that partition to choose the final token
3. When decoding, the process is reversed:
   - For each token in the generated text, we identify which partition it belongs to
   - The partition index is converted back to bits
   - Bits are assembled into bytes and then into the original message

## Technical Implementation

- Implements a lightweight version of the GPT-2 tokenizer
- Uses a simplified language model with embedded vocabulary
- No external dependencies or Python required
- Implementation details based on the original MultiBitStego Python code

## License

[MIT License](LICENSE)

## Acknowledgments

- Based on the original MultiBitStego Python implementation