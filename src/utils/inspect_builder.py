# ============================================================================
# Inspect FromConversationFileBuilder to Find Correct Parameters
# ============================================================================

from tinker_cookbook.recipes import sl_basic
import inspect

print("🔍 Inspecting FromConversationFileBuilder...")
print("="*80)

# Get the class
builder_class = sl_basic.FromConversationFileBuilder

# Check signature
print("\n📋 FromConversationFileBuilder signature:")
try:
    sig = inspect.signature(builder_class.__init__)
    print(f"  {sig}")
except Exception as e:
    print(f"  ❌ Could not get signature: {e}")

# Check docstring
print("\n📖 Docstring:")
if builder_class.__doc__:
    print(f"  {builder_class.__doc__}")
else:
    print("  No docstring available")

# Check attributes
print("\n📋 Class attributes:")
attrs = [attr for attr in dir(builder_class) if not attr.startswith('_')]
for attr in attrs[:20]:
    print(f"  • {attr}")

# Try to inspect the source
print("\n📄 Source code (first 50 lines):")
try:
    source = inspect.getsource(builder_class)
    lines = source.split('\n')[:50]
    for i, line in enumerate(lines, 1):
        print(f"{i:3}: {line}")
except Exception as e:
    print(f"  ❌ Could not get source: {e}")

# Check if it's a dataclass or has __annotations__
print("\n🔍 Type annotations:")
if hasattr(builder_class, '__annotations__'):
    for name, type_hint in builder_class.__annotations__.items():
        print(f"  • {name}: {type_hint}")
else:
    print("  No annotations found")

# Try to create an instance with minimal args to see what's required
print("\n🧪 Testing instantiation...")
try:
    # Try with no args
    test = builder_class()
    print("  ✅ No args required!")
except TypeError as e:
    print(f"  ❌ Error with no args: {e}")
    
    # Try to parse the error message to find required args
    error_msg = str(e)
    print(f"\n💡 Error message suggests:")
    print(f"  {error_msg}")

print("\n" + "="*80)

