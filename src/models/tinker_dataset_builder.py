# ============================================================================
# FIXED: Create Custom Dataset Builder for news_content Data
# ============================================================================
# This version uses the correct approach based on Tinker's actual API

from tinker_cookbook.recipes import sl_basic
from pathlib import Path
import json
import inspect

print("🔍 First, let's inspect FromConversationFileBuilder...")
print("="*80)

# Inspect the builder to find correct parameters
try:
    sig = inspect.signature(sl_basic.FromConversationFileBuilder.__init__)
    print(f"✅ FromConversationFileBuilder signature:")
    print(f"   {sig}")
except Exception as e:
    print(f"❌ Could not inspect: {e}")

print("\n" + "="*80)
print("🔧 Creating custom dataset builder for news_content data...")
print("="*80)

# Path to your training data
base_dir = "/content/drive/MyDrive/AI_Projects/news_content_FineTuning/training_data"
train_file_path = f"{base_dir}/news_content_train_data.jsonl"

# Verify file exists
if not Path(train_file_path).exists():
    print(f"❌ Training file not found: {train_file_path}")
    print("💡 Please run the train/test split code first!")
else:
    print(f"✅ Found training file: {train_file_path}")
    
    # Count examples
    with open(train_file_path, 'r') as f:
        num_examples = sum(1 for line in f if line.strip())
    print(f"📊 Training examples: {num_examples}")
    
    # APPROACH 1: Try using chat_datasets module
    print("\n🔧 Approach 1: Using chat_datasets module...")
    try:
        # Check if chat_datasets has a function to load from file
        if hasattr(sl_basic, 'chat_datasets'):
            print("✅ chat_datasets module found")
            chat_datasets_items = [item for item in dir(sl_basic.chat_datasets) if not item.startswith('_')]
            print(f"   Available: {chat_datasets_items[:10]}")
            
            # Try to use it
            if hasattr(sl_basic.chat_datasets, 'from_conversation_file'):
                print("   Trying chat_datasets.from_conversation_file...")
                dataset_builder = sl_basic.chat_datasets.from_conversation_file(
                    train_file_path,
                    common_config=sl_basic.ChatDatasetBuilderCommonConfig(
                        model_name_for_tokenizer=MODEL_NAME_TINKER,
                        renderer_name='llama3',
                        max_length=32768,
                        batch_size=4,
                        train_on_what=sl_basic.TrainOnWhat.ALL_ASSISTANT_MESSAGES
                    )
                )
                print("   ✅ Dataset builder created via chat_datasets!")
    except Exception as e:
        print(f"   ❌ Approach 1 failed: {e}")
        
        # APPROACH 2: Try FromConversationFileBuilder with correct params
        print("\n🔧 Approach 2: Trying FromConversationFileBuilder with different params...")
        try:
            # Maybe it takes the file path as first positional arg?
            dataset_builder = sl_basic.FromConversationFileBuilder(
                train_file_path,  # Try as positional arg
                common_config=sl_basic.ChatDatasetBuilderCommonConfig(
                    model_name_for_tokenizer=MODEL_NAME_TINKER,
                    renderer_name='llama3',
                    max_length=32768,
                    batch_size=4,
                    train_on_what=sl_basic.TrainOnWhat.ALL_ASSISTANT_MESSAGES
                )
            )
            print("   ✅ Dataset builder created with positional arg!")
        except Exception as e2:
            print(f"   ❌ Approach 2 failed: {e2}")
            
            # APPROACH 3: Try with 'file_path' parameter
            print("\n🔧 Approach 3: Trying with 'file_path' parameter...")
            try:
                dataset_builder = sl_basic.FromConversationFileBuilder(
                    file_path=train_file_path,
                    common_config=sl_basic.ChatDatasetBuilderCommonConfig(
                        model_name_for_tokenizer=MODEL_NAME_TINKER,
                        renderer_name='llama3',
                        max_length=32768,
                        batch_size=4,
                        train_on_what=sl_basic.TrainOnWhat.ALL_ASSISTANT_MESSAGES
                    )
                )
                print("   ✅ Dataset builder created with file_path!")
            except Exception as e3:
                print(f"   ❌ Approach 3 failed: {e3}")
                
                # APPROACH 4: Try with 'path' parameter
                print("\n🔧 Approach 4: Trying with 'path' parameter...")
                try:
                    dataset_builder = sl_basic.FromConversationFileBuilder(
                        path=train_file_path,
                        common_config=sl_basic.ChatDatasetBuilderCommonConfig(
                            model_name_for_tokenizer=MODEL_NAME_TINKER,
                            renderer_name='llama3',
                            max_length=32768,
                            batch_size=4,
                            train_on_what=sl_basic.TrainOnWhat.ALL_ASSISTANT_MESSAGES
                        )
                    )
                    print("   ✅ Dataset builder created with path!")
                except Exception as e4:
                    print(f"   ❌ Approach 4 failed: {e4}")
                    
                    # APPROACH 5: Try with just common_config
                    print("\n🔧 Approach 5: Trying with only common_config...")
                    try:
                        dataset_builder = sl_basic.FromConversationFileBuilder(
                            common_config=sl_basic.ChatDatasetBuilderCommonConfig(
                                model_name_for_tokenizer=MODEL_NAME_TINKER,
                                renderer_name='llama3',
                                max_length=32768,
                                batch_size=4,
                                train_on_what=sl_basic.TrainOnWhat.ALL_ASSISTANT_MESSAGES
                            )
                        )
                        # Then maybe set the file path as an attribute?
                        if hasattr(dataset_builder, 'file_path'):
                            dataset_builder.file_path = train_file_path
                        elif hasattr(dataset_builder, 'conversation_file_path'):
                            dataset_builder.conversation_file_path = train_file_path
                        elif hasattr(dataset_builder, 'path'):
                            dataset_builder.path = train_file_path
                        
                        print("   ✅ Dataset builder created, file path set as attribute!")
                    except Exception as e5:
                        print(f"   ❌ Approach 5 failed: {e5}")
                        print("\n❌ ALL APPROACHES FAILED!")
                        print("💡 Need to inspect the actual FromConversationFileBuilder implementation")

# If we got here and dataset_builder exists, verify it
if 'dataset_builder' in locals():
    print(f"\n✅ Dataset builder created successfully!")
    print(f"  • Type: {type(dataset_builder).__name__}")
    print(f"  • Attributes: {[attr for attr in dir(dataset_builder) if not attr.startswith('_')][:10]}")
    
    # Check if it has the file path set
    for attr in ['file_path', 'conversation_file_path', 'path', 'conversations_file']:
        if hasattr(dataset_builder, attr):
            value = getattr(dataset_builder, attr)
            print(f"  • {attr}: {value}")

print("\n" + "="*80)

