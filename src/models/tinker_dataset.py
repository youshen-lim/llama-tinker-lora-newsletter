# ============================================================================
# READY-TO-USE CODE CELLS FOR TINKER TRAINING WITH news_content DATA
# ============================================================================
#
# These cells fix the issue where Tinker was training on the "no_robots" 
# dataset instead of your news_content data.
#
# USAGE:
# 1. Copy Cell 1 and paste it BEFORE "Build sl_basic Training Config"
# 2. Replace the "Build sl_basic Training Config" cell with Cell 2
# 3. Run both cells
# 4. Then run "Run Tinker Training with nest_asyncio Fix"
#
# ============================================================================

# ============================================================================
# CELL 1: Create Custom Dataset Builder for news_content Data
# ============================================================================
# 📍 INSERT THIS CELL BEFORE "Build sl_basic Training Config"

from tinker_cookbook.recipes import sl_basic
from pathlib import Path
import json

print("🔧 Creating custom dataset builder for news_content data...")
print("="*80)

# Path to your training data
base_dir = "/content/drive/MyDrive/AI_Projects/news_content_FineTuning/training_data"
train_file_path = f"{base_dir}/news_content_train_data.jsonl"

# Verify file exists
if not Path(train_file_path).exists():
    print(f"❌ Training file not found: {train_file_path}")
    print("💡 Please run the train/test split code first!")
    print("\n⚠️ Cannot proceed without training data!")
else:
    print(f"✅ Found training file: {train_file_path}")
    
    # Count examples and show sample
    with open(train_file_path, 'r') as f:
        lines = [line for line in f if line.strip()]
        num_examples = len(lines)
        
        # Show first example to verify it's news_content data
        if lines:
            first_example = json.loads(lines[0])
            print(f"\n📊 Training examples: {num_examples}")
            print(f"\n📋 First example preview:")
            if 'messages' in first_example:
                user_msg = first_example['messages'][0]['content'][:200]
                print(f"  User: {user_msg}...")
            else:
                print(f"  Keys: {list(first_example.keys())}")
    
    # Create dataset builder using FromConversationFileBuilder
    dataset_builder = sl_basic.FromConversationFileBuilder(
        conversation_file_path=train_file_path,
        common_config=sl_basic.ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL_NAME_TINKER,
            renderer_name='llama3',  # Use llama3 renderer for Llama models
            max_length=32768,
            batch_size=4,  # Match your batch size
            train_on_what=sl_basic.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
    )
    
    print(f"\n✅ Custom dataset builder created!")
    print(f"  • File: {train_file_path}")
    print(f"  • Examples: {num_examples}")
    print(f"  • Model: {MODEL_NAME_TINKER}")
    print(f"  • Renderer: llama3")
    print(f"  • Batch size: 4")
    print(f"  • Training on: ALL_ASSISTANT_MESSAGES")
    
    print("\n🎯 Dataset builder ready!")
    print("   Next: Run the config creation cell to build training config")
    
print("="*80)


# ============================================================================
# CELL 2: Build sl_basic Training Config with Custom Dataset
# ============================================================================
# 📍 REPLACE the existing "Build sl_basic Training Config" cell with this

from tinker_cookbook.recipes import sl_basic
from tinker_cookbook.hyperparam_utils import get_lr
import chz

print("⚙️ Building sl_basic training configuration...")
print("="*80)

# Verify dataset_builder exists
if 'dataset_builder' not in globals():
    print("❌ ERROR: dataset_builder not found!")
    print("💡 Please run the 'Create Custom Dataset Builder' cell first!")
    print("="*80)
else:
    # Get recommended learning rate
    model_name = MODEL_NAME_TINKER
    recommended_lr = get_lr(model_name)
    
    print(f"\n📋 Training Parameters:")
    print(f"  • Model: {model_name}")
    print(f"  • Recommended LR: {recommended_lr:.6f}")
    print(f"  • LoRA rank: 32")
    print(f"  • Epochs: 1")
    print(f"  • Batch size: 4")
    print(f"  • Log path: /tmp/tinker-examples/sl_basic")
    
    # Create config blueprint
    blueprint = sl_basic.build_config_blueprint()
    
    # Apply custom settings INCLUDING the custom dataset builder
    config = blueprint.apply({
        'model_name': model_name,
        'learning_rate': recommended_lr,
        'lora_rank': 32,
        'num_epochs': 1,
        'save_every': 20,
        'log_path': '/tmp/tinker-examples/sl_basic',
        'dataset_builder': dataset_builder,  # ✅ USE CUSTOM DATASET BUILDER
    }).make()
    
    print(f"\n✅ Config created with CUSTOM dataset builder!")
    print(f"\n📋 Final Configuration:")
    print(f"  • Model: {config.model_name}")
    print(f"  • Learning rate: {config.learning_rate:.6f}")
    print(f"  • LoRA rank: {config.lora_rank}")
    print(f"  • Num epochs: {config.num_epochs}")
    print(f"  • Save every: {config.save_every} steps")
    print(f"  • Log path: {config.log_path}")
    print(f"  • Dataset builder: {type(config.dataset_builder).__name__}")
    
    # Verify it's using the custom builder
    if hasattr(config.dataset_builder, 'conversation_file_path'):
        print(f"\n✅ VERIFIED: Using custom dataset from:")
        print(f"   {config.dataset_builder.conversation_file_path}")
    else:
        print(f"\n⚠️ WARNING: Dataset builder type: {type(config.dataset_builder).__name__}")
        print(f"   Expected: FromConversationFileBuilder")
    
    print("\n🎯 Config ready!")
    print("   Next: Run 'Run Tinker Training with nest_asyncio Fix' to start training")
    
    print("="*80)


# ============================================================================
# VERIFICATION CELL (Optional - Run after training starts)
# ============================================================================
# 📍 Use this to verify training is using the correct dataset

import json

print("🔍 Verifying training dataset...")
print("="*80)

# Check the config
if 'config' in globals():
    print(f"\n📋 Config Dataset Builder:")
    print(f"  • Type: {type(config.dataset_builder).__name__}")
    
    if hasattr(config.dataset_builder, 'conversation_file_path'):
        file_path = config.dataset_builder.conversation_file_path
        print(f"  • File: {file_path}")
        
        # Count examples in the file
        with open(file_path, 'r') as f:
            num_examples = sum(1 for line in f if line.strip())
        print(f"  • Examples: {num_examples}")
        
        # Show first example
        with open(file_path, 'r') as f:
            first_line = next(line for line in f if line.strip())
            first_example = json.loads(first_line)
            
        print(f"\n📋 First Training Example:")
        if 'messages' in first_example:
            for msg in first_example['messages']:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:150]
                print(f"  {role}: {content}...")
        
        print(f"\n✅ Training will use YOUR news_content data!")
        print(f"   NOT the 'no_robots' dataset")
    else:
        print(f"\n⚠️ WARNING: Using default dataset builder")
        print(f"   This will train on 'no_robots' dataset, not your news_content data!")
        print(f"   Please run the custom dataset builder cell first")
else:
    print("❌ Config not found. Please run the config creation cell first.")

print("="*80)


# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================

"""
🔧 TROUBLESHOOTING:

1. **Error: "dataset_builder not found"**
   - Make sure you run CELL 1 before CELL 2
   - Check that CELL 1 completed successfully
   - Verify the training file exists at the path

2. **Training shows "no_robots" examples**
   - The config is using the default dataset builder
   - Re-run CELL 1 to create custom dataset_builder
   - Re-run CELL 2 to create config with custom builder
   - Then re-run training

3. **Error: "Training file not found"**
   - Run the train/test split code first
   - This creates news_content_train_data.jsonl
   - Verify the file exists in Google Drive

4. **Training examples don't match news_content data**
   - Check the file path in CELL 1
   - Verify news_content_train_data.jsonl contains your data
   - Run the verification cell to check what's being loaded

5. **Want to use different model or parameters**
   - Modify the parameters in CELL 2:
     - model_name: Change MODEL_NAME_TINKER
     - lora_rank: Default 32, can try 16 or 64
     - num_epochs: Default 1, can increase for more training
     - batch_size: Default 4, adjust based on GPU memory

📚 EXPECTED WORKFLOW:

1. ✅ Run train/test split (creates news_content_train_data.jsonl)
2. ✅ Run CELL 1 (creates custom dataset_builder)
3. ✅ Run CELL 2 (creates config with custom dataset)
4. ✅ Run training cell (trains on YOUR news_content data)
5. ✅ Verify logged examples match your news_content data

🎯 SUCCESS INDICATORS:

- CELL 1 shows: "✅ Custom dataset builder created!"
- CELL 2 shows: "✅ VERIFIED: Using custom dataset from: ..."
- Training logs show YOUR news_content examples, not coffee/Texas/Calico stories
- Training completes with checkpoints saved

💡 REMEMBER:

The key fix is creating a FromConversationFileBuilder that points to your
news_content_train_data.jsonl file, instead of using the default NoRobotsBuilder
that loads the "no_robots" dataset.
"""

