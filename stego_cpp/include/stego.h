#pragma once

#include "gpt2_model.h"
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <unordered_map>

class StegoGPT {
public:
    StegoGPT(int bits_per_token = 2);
    ~StegoGPT();

    // Initialize the model
    bool initialize();

    // Encode a message into steganographic text
    std::pair<std::string, std::unordered_map<std::string, int>> 
    encode_message(const std::string& message, const std::string& prompt, 
                  float temp = 1.0, int top_k = 32, int max_tokens = 200);

    // Decode a message from steganographic text
    std::pair<std::string, std::unordered_map<std::string, int>> 
    decode_message(const std::string& encoded_text, const std::string& prompt, 
                  int expected_bits = -1);

    // Helper function for testing
    std::vector<bool> text_to_bits(const std::string& text);

private:
    // Convert text to bits and vice versa
    std::string bits_to_text(const std::vector<bool>& bits);

    // Convert bits to partition index and vice versa
    int bits_to_partition(const std::vector<bool>& bits, int num_partitions);
    std::vector<bool> partition_to_bits(int partition, int num_bits);

    // Internal encode/decode methods
    std::pair<std::string, std::unordered_map<std::string, int>> 
    encode(const std::vector<bool>& message_bits, const std::string& prompt, 
           float temp, int top_k, int max_tokens);
    
    std::pair<std::vector<bool>, std::unordered_map<std::string, int>> 
    decode(const std::string& encoded_text, const std::string& prompt, int expected_bits);

    // Model parameters
    int bits_per_token_;
    
    // GPT-2 model and tokenizer
    GPT2Model model_;
    GPT2Tokenizer tokenizer_;
    
    // Random number generator
    std::mt19937 rng_;
};