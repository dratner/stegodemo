#include "../include/gpt2_model.h"
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

// GPT-2 core vocabulary (simplified for implementation)
// This contains just a tiny subset of the full GPT-2 vocabulary
// for demonstration purposes
const std::vector<std::string> GPT2_CORE_VOCAB = {
    "<|endoftext|>", "the", "and", "a", "in", "to", "of", "is", 
    "that", "it", "with", "for", "as", "was", "on", "are", "by", 
    "this", "be", "not", "or", "have", "from", "at", "but", "an", 
    "they", "you", "were", "their", "one", "all", "we", "can", 
    "has", "which", "about", "when", "what", "there", "his", "her", 
    "who", "will", "more", "no", "if", "out", "so", "up", "time", 
    "other", "people", "than", "into", "just", "its", "some", "two", 
    "these", "may", "like", "then", "would", "first", "because", "see", 
    "any", "been", "now", "could", "after", "over", "only", "many", 
    "most", "such", "where", "through", "way", "each", "before", "how", 
    "well", "years", "our", "very", "between", "him", "should", "must", 
    "day", "even", "here", "said", "back", "did", "under", "made", 
    "down", "them", "she", "new", "world", "also", "work", "good", 
    "same", "year", "three", "use", "being", "still", "those", "both", 
    "never", "life", "while", "last", "right", "off", "get", "us", 
    "might", "come", "state", "know", "take", "make", "long", "say", 
    "part", "great", "since", "against", "place", "own", "too", "go", 
    "used", "man", "around", "however", "home", "small", "found", "went", 
    "thought", "called", "few", "without", "general", "much", "every", 
    "another", "look", "give", "high", "put", "does", "got", "old", "end",
    ".", ",", "?", "!", "'", "\"", ":", ";", "-", "(", ")", "[", "]", "{", "}"
};

// Simplified implementation of GPT-2 tokenizer
GPT2Tokenizer::GPT2Tokenizer() {
    // Initialize with the minimal embedded vocabulary
    vocab.tokens = GPT2_CORE_VOCAB;
    
    // Build token_to_id map
    for (size_t i = 0; i < vocab.tokens.size(); i++) {
        vocab.token_to_id[vocab.tokens[i]] = i;
    }
}

bool GPT2Tokenizer::load_from_file(const std::string& vocab_path) {
    // In a real implementation, this would load vocabulary and BPE merges
    // from files, but for this simplified version we'll use the embedded vocab
    return true;
}

std::vector<int> GPT2Tokenizer::encode(const std::string& text) {
    // Very simplified tokenization that just splits on spaces
    // In a full implementation, this would use BPE encoding
    std::vector<int> result;
    std::string current;
    bool in_token = false;
    
    // Simple character-by-character tokenization
    for (char c : text) {
        if (std::isspace(c)) {
            if (!current.empty()) {
                // Try to find the token in vocabulary
                auto it = vocab.token_to_id.find(current);
                if (it != vocab.token_to_id.end()) {
                    result.push_back(it->second);
                } else {
                    // Unknown token, use character by character
                    for (char ch : current) {
                        std::string char_str(1, ch);
                        auto char_it = vocab.token_to_id.find(char_str);
                        if (char_it != vocab.token_to_id.end()) {
                            result.push_back(char_it->second);
                        } else {
                            // Fall back to a common token if not found
                            result.push_back(0); // Use <|endoftext|> as fallback
                        }
                    }
                }
                current.clear();
            }
        } else {
            // Handle punctuation as separate tokens
            if (std::ispunct(c)) {
                if (!current.empty()) {
                    auto it = vocab.token_to_id.find(current);
                    if (it != vocab.token_to_id.end()) {
                        result.push_back(it->second);
                    }
                    current.clear();
                }
                
                std::string punct_str(1, c);
                auto punct_it = vocab.token_to_id.find(punct_str);
                if (punct_it != vocab.token_to_id.end()) {
                    result.push_back(punct_it->second);
                }
            } else {
                current += c;
            }
        }
    }
    
    // Handle any remaining token
    if (!current.empty()) {
        auto it = vocab.token_to_id.find(current);
        if (it != vocab.token_to_id.end()) {
            result.push_back(it->second);
        } else {
            // Unknown token, use character by character
            for (char ch : current) {
                std::string char_str(1, ch);
                auto char_it = vocab.token_to_id.find(char_str);
                if (char_it != vocab.token_to_id.end()) {
                    result.push_back(char_it->second);
                } else {
                    // Fall back to a common token if not found
                    result.push_back(0);
                }
            }
        }
    }
    
    return result;
}

std::string GPT2Tokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < static_cast<int>(vocab.tokens.size())) {
            std::string token = vocab.tokens[id];
            // Special case for end of text token
            if (token == "<|endoftext|>") {
                break;
            }
            
            // Add space for word tokens but not for punctuation
            if (!result.empty() && !std::ispunct(token[0]) && 
                !std::ispunct(result.back())) {
                result += ' ';
            }
            
            result += token;
        }
    }
    return result;
}

int GPT2Tokenizer::token_to_id(const std::string& token) {
    auto it = vocab.token_to_id.find(token);
    if (it != vocab.token_to_id.end()) {
        return it->second;
    }
    return -1; // Unknown token
}

std::string GPT2Tokenizer::id_to_token(int id) {
    if (id >= 0 && id < static_cast<int>(vocab.tokens.size())) {
        return vocab.tokens[id];
    }
    return ""; // Unknown ID
}

// GPT-2 Model implementation
bool GPT2Model::load(const std::string& model_path) {
    // In a real implementation, this would load weights from a file
    // but for the minimal version, we'll use random weights
    
    // Initialize model with embedded vocabulary
    vocab.tokens = GPT2_CORE_VOCAB;
    for (size_t i = 0; i < vocab.tokens.size(); i++) {
        vocab.token_to_id[vocab.tokens[i]] = i;
    }
    
    // For demonstration purposes, create random embeddings
    n_vocab = vocab.tokens.size();
    token_embeddings = GPT2ModelLoader::generate_dummy_embeddings(n_vocab, n_embd);
    
    // Set up each layer with random weights
    for (int i = 0; i < n_layer; i++) {
        layers[i].ln_1_weight.resize(n_embd, 1.0f);
        layers[i].ln_1_bias.resize(n_embd, 0.0f);
        layers[i].ln_2_weight.resize(n_embd, 1.0f);
        layers[i].ln_2_bias.resize(n_embd, 0.0f);
    }
    
    // Final layer normalization
    ln_f_weight.resize(n_embd, 1.0f);
    ln_f_bias.resize(n_embd, 0.0f);
    
    return true;
}

std::vector<float> GPT2Model::forward(const std::vector<int>& input_ids) {
    // This is a simplified forward pass that generates deterministic
    // but plausible logits based on context and vocabulary
    
    // Generate logits that favor common tokens and create coherent sequences
    std::vector<float> logits(n_vocab);
    
    // Seed based on context for deterministic but context-sensitive generation
    uint64_t seed = 42;
    if (!input_ids.empty()) {
        // Use last few tokens as seed
        for (size_t i = std::max(0, static_cast<int>(input_ids.size()) - 3); i < input_ids.size(); i++) {
            seed = seed * 31 + input_ids[i];
        }
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Context-sensitive logit generation
    for (int i = 0; i < n_vocab; i++) {
        // Common tokens (first 100) get higher probability
        if (i < 100) {
            logits[i] = 5.0f + dist(rng) * 0.5f;  // Higher baseline with less variance
        } else {
            logits[i] = dist(rng) * 0.2f;  // Low probability for rare tokens
        }
        
        // Semantic coherence - if this token appeared in context, increase probability
        if (!input_ids.empty()) {
            // Check the last 5 tokens (or fewer if context is shorter)
            size_t context_start = input_ids.size() > 5 ? input_ids.size() - 5 : 0;
            for (size_t j = context_start; j < input_ids.size(); j++) {
                if (input_ids[j] == i) {
                    // Recent tokens get higher boost
                    float recency_boost = 1.0f + (j - context_start) / 5.0f;
                    logits[i] += 3.0f * recency_boost;
                    break;
                }
            }
            
            // Favor continuing sequences and common word pairs
            if (input_ids.size() >= 2) {
                int last_token = input_ids.back();
                
                // Simple grammar rules:
                // After "the", favor nouns
                auto the_it = vocab.token_to_id.find("the");
                if (the_it != vocab.token_to_id.end() && last_token == the_it->second) {
                    // Increase probability of nouns (using token indices as proxy)
                    if (i >= 10 && i < 30) {
                        logits[i] += 4.0f;
                    }
                }
                
                // After punctuation, favor capitalized words or common sentence starters
                if (last_token < vocab.tokens.size()) {
                    const std::string& last_token_str = vocab.tokens[last_token];
                    if (!last_token_str.empty() && std::ispunct(last_token_str[0])) {
                        if (i >= 30 && i < 50) { // Approximate range for capitalized words
                            logits[i] += 3.0f;
                        }
                    }
                }
            }
        }
    }
    
    return logits;
}

// Model loader implementation
const std::vector<float> GPT2ModelLoader::generate_dummy_embeddings(int n_vocab, int n_embd) {
    std::vector<float> embeddings(n_vocab * n_embd);
    
    // Initialize with random values for demonstration
    std::default_random_engine rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    for (int i = 0; i < n_vocab * n_embd; i++) {
        embeddings[i] = dist(rng);
    }
    
    return embeddings;
}

bool GPT2ModelLoader::load_embedded_model(GPT2Model& model) {
    // Generate a dummy model with random weights
    // In a real implementation, this would load weights from files
    
    // Set up model parameters
    model.n_vocab = GPT2_CORE_VOCAB.size();
    model.n_embd = 768;   // Embedding dimension for GPT-2 small
    model.n_layer = 12;   // Number of layers for GPT-2 small
    model.n_head = 12;    // Number of attention heads for GPT-2 small
    
    // Load vocabulary
    model.vocab.tokens = GPT2_CORE_VOCAB;
    for (size_t i = 0; i < model.vocab.tokens.size(); i++) {
        model.vocab.token_to_id[model.vocab.tokens[i]] = i;
    }
    
    // Generate random embeddings
    model.token_embeddings = generate_dummy_embeddings(model.n_vocab, model.n_embd);
    
    // Set up each layer with random weights
    for (int i = 0; i < model.n_layer; i++) {
        model.layers[i].ln_1_weight.resize(model.n_embd, 1.0f);
        model.layers[i].ln_1_bias.resize(model.n_embd, 0.0f);
        model.layers[i].ln_2_weight.resize(model.n_embd, 1.0f);
        model.layers[i].ln_2_bias.resize(model.n_embd, 0.0f);
    }
    
    // Final layer normalization
    model.ln_f_weight.resize(model.n_embd, 1.0f);
    model.ln_f_bias.resize(model.n_embd, 0.0f);
    
    return true;
}