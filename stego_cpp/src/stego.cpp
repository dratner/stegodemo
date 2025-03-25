#include "../include/stego.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>

StegoGPT::StegoGPT(int bits_per_token)
    : bits_per_token_(bits_per_token) {
    // Initialize random number generator with time-based seed
    rng_.seed(static_cast<unsigned int>(std::time(nullptr)));
}

StegoGPT::~StegoGPT() {
    // Clean up resources if needed
}

bool StegoGPT::initialize() {
    // Initialize GPT-2 model with embedded weights
    return GPT2ModelLoader::load_embedded_model(model_);
}

std::vector<bool> StegoGPT::text_to_bits(const std::string& text) {
    std::vector<bool> bits;
    for (unsigned char c : text) {
        for (int i = 0; i < 8; ++i) {
            bits.push_back((c >> i) & 1);
        }
    }
    return bits;
}

std::string StegoGPT::bits_to_text(const std::vector<bool>& bits) {
    std::string text;
    for (size_t i = 0; i < bits.size(); i += 8) {
        if (i + 8 <= bits.size()) {
            unsigned char byte = 0;
            for (int j = 0; j < 8; ++j) {
                if (bits[i + j]) {
                    byte |= (1 << j);
                }
            }
            text.push_back(byte);
        }
    }
    return text;
}

int StegoGPT::bits_to_partition(const std::vector<bool>& bits, int num_partitions) {
    int value = 0;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) {
            value |= (1 << i);
        }
    }
    return value % num_partitions;
}

std::vector<bool> StegoGPT::partition_to_bits(int partition, int num_bits) {
    std::vector<bool> bits(num_bits);
    for (int i = 0; i < num_bits; ++i) {
        bits[i] = (partition >> i) & 1;
    }
    return bits;
}

std::pair<std::string, std::unordered_map<std::string, int>> 
StegoGPT::encode(const std::vector<bool>& message_bits, const std::string& prompt, 
                float temp, int top_k, int max_tokens) {
    std::unordered_map<std::string, int> stats = {
        {"bits_encoded", 0},
        {"tokens_generated", 0}
    };
    
    // Tokenize the prompt
    std::vector<int> input_tokens = tokenizer_.encode(prompt);
    if (input_tokens.empty()) {
        std::cerr << "Failed to tokenize prompt" << std::endl;
        return {"", stats};
    }
    
    std::vector<int> generated_tokens = input_tokens;
    
    // Calculate number of partitions based on bits per token
    const int num_partitions = 1 << bits_per_token_;
    
    // Process message_bits in chunks of bits_per_token
    size_t bit_index = 0;
    
    while (bit_index < message_bits.size() && 
           (generated_tokens.size() - input_tokens.size()) < static_cast<size_t>(max_tokens)) {
        
        // Get logits for the next token based on context
        std::vector<float> logits = model_.forward(generated_tokens);
        
        // Copy and apply temperature
        std::vector<float> probs = logits;
        for (auto& p : probs) {
            p /= temp;
        }
        
        // Apply softmax
        float max_logit = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (auto& p : probs) {
            p = std::exp(p - max_logit);
            sum += p;
        }
        for (auto& p : probs) {
            p /= sum;
        }
        
        // Create index->probability pairs
        std::vector<std::pair<int, float>> token_probs(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            token_probs[i] = {i, probs[i]};
        }
        
        // Sort by probability (descending)
        std::sort(token_probs.begin(), token_probs.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Take top_k tokens
        if (token_probs.size() > static_cast<size_t>(top_k)) {
            token_probs.resize(top_k);
        }
        
        // Determine how many bits we can encode with this token
        int bits_to_encode = std::min(bits_per_token_, static_cast<int>(message_bits.size() - bit_index));
        int num_current_partitions = 1 << bits_to_encode;
        
        // Get current bits to encode
        std::vector<bool> current_bits(message_bits.begin() + bit_index, 
                                      message_bits.begin() + bit_index + bits_to_encode);
        
        // Convert bits to partition index
        int partition_idx = bits_to_partition(current_bits, num_current_partitions);
        
        // Divide the top_k tokens into num_partitions groups
        int partition_size = token_probs.size() / num_current_partitions;
        if (partition_size == 0) partition_size = 1;  // Ensure at least one token per partition
        
        // Select the partition corresponding to our bits
        int start_idx = partition_idx * partition_size;
        int end_idx = (partition_idx < num_current_partitions - 1) ? 
                     (partition_idx + 1) * partition_size : token_probs.size();
        
        // Make sure we have valid indices
        start_idx = std::min(start_idx, static_cast<int>(token_probs.size()) - 1);
        end_idx = std::min(end_idx, static_cast<int>(token_probs.size()));
        if (start_idx >= end_idx) end_idx = start_idx + 1;
        
        // Extract the token pool for this partition
        std::vector<std::pair<int, float>> partition_tokens(
            token_probs.begin() + start_idx, token_probs.begin() + end_idx);
        
        // Renormalize probabilities in the selected partition
        float partition_sum = 0.0f;
        for (const auto& pair : partition_tokens) {
            partition_sum += pair.second;
        }
        for (auto& pair : partition_tokens) {
            pair.second /= partition_sum;
        }
        
        // Sample from the partition
        int next_token_idx = 0;
        if (partition_tokens.size() > 1) {
            // Create weights vector for distribution
            std::vector<float> weights;
            weights.reserve(partition_tokens.size());
            for (const auto& pair : partition_tokens) {
                weights.push_back(pair.second);
            }
            
            std::discrete_distribution<> dist(weights.begin(), weights.end());
            next_token_idx = dist(rng_);
        }
        int next_token = partition_tokens[next_token_idx].first;
        
        // Add token to generated text
        generated_tokens.push_back(next_token);
        
        // Update stats
        stats["bits_encoded"] += bits_to_encode;
        stats["tokens_generated"]++;
        bit_index += bits_to_encode;
        
        // Check if we've reached an end token (EOS)
        if (next_token == tokenizer_.eos_token_id()) {
            break;
        }
    }
    
    // Calculate bits per token stat
    stats["bits_per_token"] = stats["bits_encoded"] / std::max(1, stats["tokens_generated"]);
    
    // Detokenize the generated text (excluding the prompt)
    std::vector<int> output_tokens(generated_tokens.begin() + input_tokens.size(), generated_tokens.end());
    std::string generated_text = tokenizer_.decode(output_tokens);
    
    return {generated_text, stats};
}

std::pair<std::vector<bool>, std::unordered_map<std::string, int>> 
StegoGPT::decode(const std::string& encoded_text, const std::string& prompt, int expected_bits) {
    std::vector<bool> extracted_bits;
    std::unordered_map<std::string, int> stats = {
        {"bits_decoded", 0},
        {"tokens_processed", 0}
    };
    
    // Tokenize the prompt and encoded text
    std::vector<int> prompt_tokens = tokenizer_.encode(prompt);
    std::vector<int> encoded_tokens = tokenizer_.encode(encoded_text);
    
    if (prompt_tokens.empty() || encoded_tokens.empty()) {
        std::cerr << "Failed to tokenize input" << std::endl;
        return {extracted_bits, stats};
    }
    
    // Calculate number of partitions based on bits per token
    const int num_partitions = 1 << bits_per_token_;
    
    // Build context from prompt tokens
    std::vector<int> context = prompt_tokens;
    
    // Process each token in the encoded text
    for (size_t i = 0; i < encoded_tokens.size(); ++i) {
        // Get logits for the current position
        std::vector<float> logits = model_.forward(context);
        
        // Apply softmax
        std::vector<float> probs = logits;
        float max_logit = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (auto& p : probs) {
            p = std::exp(p - max_logit);
            sum += p;
        }
        for (auto& p : probs) {
            p /= sum;
        }
        
        // Create index->probability pairs
        std::vector<std::pair<int, float>> token_probs(probs.size());
        for (size_t j = 0; j < probs.size(); ++j) {
            token_probs[j] = {j, probs[j]};
        }
        
        // Sort by probability (descending)
        std::sort(token_probs.begin(), token_probs.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Determine how many bits we can decode from this token
        int bits_remaining;
        int num_current_partitions;
        
        if (expected_bits > 0 && static_cast<int>(extracted_bits.size()) + bits_per_token_ > expected_bits) {
            // We're at the end of the message
            bits_remaining = expected_bits - extracted_bits.size();
            num_current_partitions = 1 << bits_remaining;
        } else {
            bits_remaining = bits_per_token_;
            num_current_partitions = num_partitions;
        }
        
        // Take top tokens
        const int top_k = std::min(32, static_cast<int>(token_probs.size()));
        token_probs.resize(top_k);
        
        // Find which partition the token belongs to
        int token_pos = -1;
        for (int j = 0; j < top_k; ++j) {
            if (token_probs[j].first == encoded_tokens[i]) {
                token_pos = j;
                break;
            }
        }
        
        // If token not in top tokens, find closest match
        if (token_pos == -1) {
            int closest_idx = 0;
            int min_distance = std::abs(token_probs[0].first - encoded_tokens[i]);
            
            for (int j = 1; j < top_k; ++j) {
                int distance = std::abs(token_probs[j].first - encoded_tokens[i]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_idx = j;
                }
            }
            
            token_pos = closest_idx;
        }
        
        // Partition size
        int partition_size = top_k / num_current_partitions;
        if (partition_size == 0) partition_size = 1;  // Ensure at least one token per partition
        
        // Determine which partition the token belongs to
        int partition_idx = std::min(token_pos / partition_size, num_current_partitions - 1);
        
        // Convert partition to bits
        std::vector<bool> partition_bits = partition_to_bits(partition_idx, bits_remaining);
        extracted_bits.insert(extracted_bits.end(), partition_bits.begin(), partition_bits.end());
        
        // Update stats
        stats["bits_decoded"] += bits_remaining;
        stats["tokens_processed"]++;
        
        // Update context for next token prediction
        context.push_back(encoded_tokens[i]);
        
        // Stop if we've decoded all expected bits
        if (expected_bits > 0 && static_cast<int>(extracted_bits.size()) >= expected_bits) {
            break;
        }
        
        // Stop if we hit the end token
        if (encoded_tokens[i] == tokenizer_.eos_token_id()) {
            break;
        }
    }
    
    // Calculate bits per token stat
    stats["bits_per_token"] = stats["bits_decoded"] / std::max(1, stats["tokens_processed"]);
    
    return {extracted_bits, stats};
}

std::pair<std::string, std::unordered_map<std::string, int>> 
StegoGPT::encode_message(const std::string& message, const std::string& prompt, 
                        float temp, int top_k, int max_tokens) {
    std::vector<bool> message_bits = text_to_bits(message);
    
    // Ensure we have enough tokens to encode the full message
    int required_tokens = (message_bits.size() + bits_per_token_ - 1) / bits_per_token_;
    if (max_tokens < required_tokens) {
        std::cout << "Warning: Increasing max_tokens from " << max_tokens 
                  << " to " << required_tokens << " to ensure full message encoding" << std::endl;
        max_tokens = required_tokens + 10; // Add some margin
    }
    
    return encode(message_bits, prompt, temp, top_k, max_tokens);
}

std::pair<std::string, std::unordered_map<std::string, int>> 
StegoGPT::decode_message(const std::string& encoded_text, const std::string& prompt, 
                        int expected_bits) {
    // If expected_bits is not provided, calculate it based on message length
    if (expected_bits < 0) {
        // This is just a default - in real usage, the caller should provide
        // the expected bits or store the length in the message itself
        expected_bits = 8 * 20;  // Assume a default message length of 20 bytes
    }
    
    auto [bits, stats] = decode(encoded_text, prompt, expected_bits);
    
    // Truncate to expected bits if necessary
    if (expected_bits > 0 && static_cast<int>(bits.size()) > expected_bits) {
        bits.resize(expected_bits);
    }
    
    // Convert bits to text
    std::string message = bits_to_text(bits);
    
    return {message, stats};
}