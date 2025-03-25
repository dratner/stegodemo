#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

// Minimal GPT-2 model implementation
// Contains only the pieces needed for the steganography algorithm

struct GPT2Vocab {
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int> token_to_id;
    
    // Special tokens
    int eos_token_id = 50256; // Default for GPT-2
    
    bool load(const std::string& vocab_path);
    int encode_token(const std::string& token) const;
    std::string decode_token(int token_id) const;
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
};

struct GPT2Model {
    // Model parameters
    int n_vocab = 50257;  // GPT-2 vocabulary size
    int n_ctx = 1024;     // Maximum context size
    int n_embd = 768;     // Embedding dimension (GPT-2 small)
    int n_layer = 12;     // Number of layers (GPT-2 small)
    int n_head = 12;      // Number of attention heads (GPT-2 small)
    
    // Model weights and state
    GPT2Vocab vocab;
    std::vector<float> token_embeddings;  // [n_vocab, n_embd]
    
    // Model layers (simplified representation)
    struct {
        // We only need the basic structures to generate embeddings and logits
        std::vector<float> ln_1_weight; // Layer normalization weights
        std::vector<float> ln_1_bias;   // Layer normalization biases
        std::vector<float> ln_2_weight; // Layer normalization weights 
        std::vector<float> ln_2_bias;   // Layer normalization biases
    } layers[12]; // Assuming GPT-2 small with 12 layers
    
    // Final layer normalization
    std::vector<float> ln_f_weight;
    std::vector<float> ln_f_bias;
    
    bool load(const std::string& model_path);
    
    // Forward pass to get logits for next token prediction
    std::vector<float> forward(const std::vector<int>& input_ids);
};

// Token Encoder-Decoder using GPT-2 tokenization rules
class GPT2Tokenizer {
public:
    GPT2Tokenizer();
    
    bool load_from_file(const std::string& vocab_path);
    
    // Simple tokenization functions
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
    
    // Helper methods
    int token_to_id(const std::string& token);
    std::string id_to_token(int id);
    
    int eos_token_id() const { return 50256; } // End of sequence token
    
private:
    GPT2Vocab vocab;
    
    // Byte-pair encoding helpers
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<int, std::string> decoder;
    
    // Simplified tokenization algorithm for GPT-2
    std::vector<std::string> tokenize(const std::string& text);
};

// Simple model loader that creates a minimal representation of GPT-2
// suitable for our steganography needs
class GPT2ModelLoader {
public:
    static bool load_embedded_model(GPT2Model& model);
    
    // We'll embed a tiny pre-trained model in the source code
    // that's enough for the steganography algorithm
    static const std::vector<float> generate_dummy_embeddings(int n_vocab, int n_embd);
};