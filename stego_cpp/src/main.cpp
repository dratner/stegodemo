#include "../include/stego.h"
#include <iostream>
#include <string>
#include <unordered_map>

void print_stats(const std::unordered_map<std::string, int>& stats) {
    for (const auto& [key, value] : stats) {
        std::cout << key << ": " << value << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Check arguments
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [bits_per_token]" << std::endl;
        std::cerr << "Example: " << argv[0] << " 2" << std::endl;
        return 1;
    }

    int bits_per_token = (argc > 1) ? std::stoi(argv[1]) : 2;

    std::cout << "Initializing StegoGPT with " << bits_per_token << " bits per token..." << std::endl;
    StegoGPT stego(bits_per_token);

    if (!stego.initialize()) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }

    // Get message from user
    std::string message = "This is a secret message!";
    std::cout << "Enter a message to hide (press Enter to use default): ";
    std::string user_message;
    std::getline(std::cin, user_message);
    if (!user_message.empty()) {
        message = user_message;
    }
    
    // Get prompt from user
    std::string prompt = "Sergio's Italian Grille is the newest restaurant in downtown Austin.";
    std::cout << "Enter a prompt (press Enter to use default): ";
    std::string user_prompt;
    std::getline(std::cin, user_prompt);
    if (!user_prompt.empty()) {
        prompt = user_prompt;
    }

    // Encode the message
    std::cout << "Encoding message: " << message << std::endl;
    auto [encoded_text, encode_stats] = stego.encode_message(
        message,
        prompt,
        1.1f,  // Higher temperature for more natural text
        32,    // More tokens for better partitioning
        200    // Max tokens to generate
    );
    
    std::cout << "Generated text: " << encoded_text << std::endl;
    std::cout << "Encoding stats:" << std::endl;
    print_stats(encode_stats);

    // Calculate expected bits for the message
    int expected_bits = stego.text_to_bits(message).size();
    std::cout << "Expected bits: " << expected_bits << std::endl;

    // Decode the message
    auto [decoded_message, decode_stats] = stego.decode_message(
        encoded_text,
        prompt,
        expected_bits
    );
    
    std::cout << "Decoded message: " << decoded_message << std::endl;
    std::cout << "Decoding stats:" << std::endl;
    print_stats(decode_stats);

    // Verify success
    std::cout << "Success: " << (message == decoded_message ? "Yes" : "No") << std::endl;
    
    // If success is not exact, print how many bytes were correct
    if (message != decoded_message) {
        int correct_bytes = 0;
        int min_length = std::min(message.length(), decoded_message.length());
        
        for (int i = 0; i < min_length; i++) {
            if (message[i] == decoded_message[i]) {
                correct_bytes++;
            }
        }
        
        std::cout << "Partially correct: " << correct_bytes << "/" << message.length() 
                  << " bytes (" << (100.0 * correct_bytes / message.length()) << "%)" << std::endl;
    }

    return 0;
}