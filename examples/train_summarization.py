#!/usr/bin/env python3
"""
Example: Training FLAN-T5 for Summarization Task
This script demonstrates how to finetune FLAN-T5 for text summarization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flan_t5_finetune import FLANT5Finetuner
from data_utils import DataProcessor
import json

def create_summarization_data():
    """Create sample summarization data"""
    summarization_data = [
        {
            "input": "Summarize this text: Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem solving. AI has been used in various fields including healthcare, finance, transportation, and entertainment.",
            "output": "AI is a computer science field creating intelligent machines that mimic human behavior, used in healthcare, finance, transportation, and entertainment."
        },
        {
            "input": "Summarize this text: Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures.",
            "output": "Climate change involves long-term temperature and weather shifts, primarily driven by human fossil fuel burning since the 1800s, creating greenhouse gas emissions that trap heat."
        },
        {
            "input": "Summarize this text: Renewable energy comes from natural sources that are constantly replenished, such as sunlight, wind, rain, tides, waves, and geothermal heat. Unlike fossil fuels, which are finite and contribute to climate change, renewable energy sources are sustainable and have minimal environmental impact. Solar power, wind energy, hydropower, and biomass are among the most common forms of renewable energy used today.",
            "output": "Renewable energy comes from naturally replenished sources like sunlight and wind, offering sustainable alternatives to finite fossil fuels with minimal environmental impact."
        },
        {
            "input": "Summarize this text: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future.",
            "output": "Machine learning enables computers to learn from experience without explicit programming, using data to identify patterns and improve decision-making."
        },
        {
            "input": "Summarize this text: The Internet of Things (IoT) refers to the network of physical objects that are embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the internet. These devices range from ordinary household objects to sophisticated industrial tools. IoT has applications in smart homes, healthcare monitoring, industrial automation, and environmental monitoring.",
            "output": "IoT connects physical objects with sensors and software to exchange data over the internet, used in smart homes, healthcare, industrial automation, and environmental monitoring."
        },
        {
            "input": "Summarize this text: Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. Blockchain technology is the foundation of cryptocurrencies like Bitcoin, but it also has applications in supply chain management, voting systems, and digital identity verification.",
            "output": "Blockchain is a distributed ledger technology using cryptographically linked blocks, foundational for cryptocurrencies and applicable to supply chains, voting, and digital identity."
        },
        {
            "input": "Summarize this text: Virtual reality (VR) is a computer-generated simulation of a three-dimensional environment that can be interacted with in a seemingly real or physical way by a person using special electronic equipment, such as a helmet with a screen inside or gloves fitted with sensors. VR technology has applications in gaming, education, healthcare, military training, and architectural visualization.",
            "output": "VR creates interactive 3D computer simulations experienced through special equipment, used in gaming, education, healthcare, military training, and architecture."
        },
        {
            "input": "Summarize this text: Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information, extorting money from users, or interrupting normal business processes. Implementing effective cybersecurity measures is particularly challenging today because there are more devices than people, and attackers are becoming more innovative.",
            "output": "Cybersecurity protects systems from digital attacks aimed at accessing sensitive data, extorting money, or disrupting business, made challenging by increasing devices and innovative attackers."
        },
        {
            "input": "Summarize this text: Big data refers to extremely large data sets that may be analyzed computationally to reveal patterns, trends, and associations, especially relating to human behavior and interactions. Big data is characterized by the three V's: volume (amount of data), velocity (speed of data generation), and variety (types of data). It is used in business intelligence, scientific research, and government policy making.",
            "output": "Big data involves large datasets analyzed for patterns and trends, characterized by volume, velocity, and variety, used in business intelligence, research, and policy making."
        },
        {
            "input": "Summarize this text: Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the internet to offer faster innovation, flexible resources, and economies of scale. Users typically pay only for cloud services they use, helping lower operating costs, run infrastructure more efficiently, and scale as business needs change.",
            "output": "Cloud computing delivers computing services over the internet, offering flexible resources and cost efficiency through pay-per-use models."
        }
    ]
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save summarization data
    with open("data/summarization_data.json", "w", encoding="utf-8") as f:
        json.dump(summarization_data, f, indent=2, ensure_ascii=False)
    
    print("Summarization data created successfully!")
    return "data/summarization_data.json"

def train_summarization_model():
    """Train FLAN-T5 for summarization task"""
    
    # Create sample data
    data_path = create_summarization_data()
    
    # Initialize finetuner for summarization
    finetuner = FLANT5Finetuner(
        model_size="large",  # Larger model for better summarization
        device="auto",       # Automatically detect best device
        max_length=512,      # Longer sequences for summarization
        batch_size=2,        # Smaller batch size for larger model
        learning_rate=3e-5,  # Slightly lower learning rate
        num_epochs=8,        # More epochs for summarization
        warmup_steps=100,    # Longer warmup
        gradient_accumulation_steps=8,  # More gradient accumulation
        use_lora=True,       # Use LoRA for efficiency
        lora_r=32,           # Higher rank for summarization
        lora_alpha=64,       # Higher alpha
        lora_dropout=0.1,
        output_dir="./summarization_output",
        checkpoint_dir="./summarization_checkpoints",
        wandb_project="flan-t5-summarization"
    )
    
    print("Loading model and tokenizer...")
    finetuner.load_model_and_tokenizer()
    
    print("Preparing dataset...")
    train_dataset = finetuner.prepare_dataset(data_path)
    
    print("Setting up training...")
    finetuner.setup_training(train_dataset)
    
    print("Starting training...")
    finetuner.train()
    
    print("Training completed!")
    print(f"Model saved to: {finetuner.output_dir}")
    print(f"Checkpoints saved to: {finetuner.checkpoint_dir}")

def test_summarization_model():
    """Test the trained summarization model"""
    
    # Initialize finetuner
    finetuner = FLANT5Finetuner(
        model_size="large",
        device="auto",
        use_lora=True,
        output_dir="./summarization_output",
        checkpoint_dir="./summarization_checkpoints"
    )
    
    # Load the trained model
    try:
        finetuner.load_checkpoint("final_model")
        print("Loaded trained model successfully!")
    except:
        print("No trained model found. Loading base model...")
        finetuner.load_model_and_tokenizer()
    
    # Test summarizations
    test_inputs = [
        "Summarize this text: Artificial Intelligence (AI) is transforming industries across the globe. From healthcare to finance, AI applications are becoming increasingly sophisticated. Machine learning algorithms can now diagnose diseases, predict market trends, and automate complex tasks. However, this rapid advancement also raises important questions about job displacement, privacy, and ethical considerations. As AI continues to evolve, society must carefully consider both its benefits and potential risks.",
        "Summarize this text: Renewable energy sources are becoming increasingly important as the world seeks to reduce carbon emissions and combat climate change. Solar power, wind energy, and hydropower are leading the transition away from fossil fuels. These technologies are not only environmentally friendly but also becoming more cost-effective. Many countries are setting ambitious targets for renewable energy adoption, with some aiming for 100% renewable electricity by 2050.",
        "Summarize this text: The Internet of Things (IoT) is revolutionizing how we interact with technology. Smart homes, connected vehicles, and industrial sensors are creating vast networks of interconnected devices. This connectivity enables unprecedented levels of automation and data collection. However, it also introduces new security challenges as more devices become potential entry points for cyberattacks. The IoT ecosystem continues to grow rapidly, with billions of devices expected to be connected by 2030."
    ]
    
    print("\nTesting summarization:")
    print("=" * 60)
    
    for input_text in test_inputs:
        generated_text = finetuner.generate(
            input_text,
            max_length=150,
            num_beams=4
        )
        print(f"Input: {input_text[:100]}...")
        print(f"Summary: {generated_text}")
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FLAN-T5 for summarization")
    parser.add_argument("--mode", choices=["train", "test", "both"], 
                       default="both", help="Mode to run")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        print("Training summarization model...")
        train_summarization_model()
    
    if args.mode in ["test", "both"]:
        print("\nTesting summarization model...")
        test_summarization_model() 