# ============================================================================
# FIXED ANNOTATION WIDGET - Complete Code with Append/Merge Fix
# ============================================================================
# 
# This is the complete annotation widget code with the critical fix applied
# to the on_save_all_click() function. The fix ensures that new annotations
# are APPENDED to existing annotations instead of OVERWRITING them.
#
# KEY FIX: Lines 237-268 - Changed from 'w' (overwrite) to merge logic
# ============================================================================

# ============================================================================
# Load Training Data for Annotation
# ============================================================================
import json
import os

DRIVE_PATH = "/content/drive/MyDrive/AI_Projects/news_content_FineTuning"
training_path = f"{DRIVE_PATH}/training_data/news_content_training_data.jsonl"

print("📂 Loading training data for annotation...")
print("="*80)

if not os.path.exists(training_path):
    print(f"❌ Training data not found at: {training_path}")
    print("💡 Please check the file path")
    loaded_examples = []
else:
    loaded_examples = []
    with open(training_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    loaded_examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠ Warning: Skipping invalid JSON line: {e}")

    print(f"✅ Loaded {len(loaded_examples)} examples for annotation")

    # Check if annotated file exists to filter out already annotated examples
    annotated_path = f"{DRIVE_PATH}/training_data/news_content_training_annotated.jsonl"

    if os.path.exists(annotated_path):
        print(f"\n📋 Checking for already annotated examples...")
        annotated_examples = []
        with open(annotated_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        annotated_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        print(f"✅ Found {len(annotated_examples)} already annotated examples")

        # Filter out annotated examples
        annotated_contents = {
            json.dumps(ex['messages'], sort_keys=True)
            for ex in annotated_examples
        }

        unannotated_examples = []
        for ex in loaded_examples:
            ex_content = json.dumps(ex['messages'], sort_keys=True)
            if ex_content not in annotated_contents:
                unannotated_examples.append(ex)

        loaded_examples = unannotated_examples
        print(f"✅ Filtered to {len(loaded_examples)} unannotated examples")
    else:
        print(f"\n📋 No existing annotations found - all {len(loaded_examples)} examples are unannotated")

    # Show first example structure
    if loaded_examples:
        print(f"\n📋 First example structure:")
        print(f"  • Keys: {list(loaded_examples[0].keys())}")
        if 'messages' in loaded_examples[0]:
            print(f"  • Number of messages: {len(loaded_examples[0]['messages'])}")
            print(f"  • Message roles: {[msg.get('role') for msg in loaded_examples[0]['messages']]}")

print("="*80)

# ============================================================================
# Initialize Annotation System
# ============================================================================
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

print("\n🎨 Setting up annotation interface...")

# Check that we have data to annotate
if not loaded_examples:
    print("❌ No training data loaded! Please run the 'Load Training Data' cell first.")
else:
    print(f"✅ Ready to annotate {len(loaded_examples)} examples")

    # Annotation state
    annotation_state = {
        'current_index': 0,
        'total_examples': len(loaded_examples),
        'annotations': {},  # Store annotations by index
        'approved': set(),
        'corrected': set(),
        'skipped': set()
    }

    print("✅ Annotation system initialized")

# ============================================================================
# Create Annotation Widgets
# ============================================================================

# Navigation
nav_label = widgets.HTML(value=f"<h2 style='font-size: 24px;'>📝 Example 1 of {annotation_state['total_examples']}</h2>")
prev_button = widgets.Button(description='← Previous', button_style='info', disabled=True)
next_button = widgets.Button(description='Next →', button_style='info')

# Content display
content_area = widgets.HTML(value="<p style='font-size: 16px;'>Loading...</p>")

# Analysis fields with larger font
relevance_slider = widgets.IntSlider(
    value=5, min=1, max=10, description='Relevance Score:',
    style={'description_width': '180px', 'font_size': '16px'},
    layout=widgets.Layout(width='600px')
)

topics_text = widgets.Textarea(
    placeholder='Enter key topics, one per line',
    description='Key Topics:',
    rows=3,
    style={'description_width': '180px', 'font_size': '16px'},
    layout=widgets.Layout(width='600px', font_size='16px')
)

companies_text = widgets.Textarea(
    placeholder='Enter company names, one per line',
    description='Companies:',
    rows=3,
    style={'description_width': '180px', 'font_size': '16px'},
    layout=widgets.Layout(width='600px', font_size='16px')
)

summary_text = widgets.Textarea(
    placeholder='Brief summary of the news_content item',
    description='Summary:',
    rows=2,
    style={'description_width': '180px', 'font_size': '16px'},
    layout=widgets.Layout(width='600px', font_size='16px')
)

# Quality rating
quality_slider = widgets.IntSlider(
    value=7, min=1, max=10, description='Quality Rating:',
    style={'description_width': '180px', 'font_size': '16px'},
    layout=widgets.Layout(width='600px')
)

notes_text = widgets.Textarea(
    placeholder='Add annotation notes or corrections made',
    description='Notes:',
    rows=2,
    style={'description_width': '180px', 'font_size': '16px'},
    layout=widgets.Layout(width='600px', font_size='16px')
)

# Action buttons
approve_button = widgets.Button(description='✅ Approve', button_style='success', layout=widgets.Layout(height='40px'))
correct_button = widgets.Button(description='✏️ Save Corrections', button_style='warning', layout=widgets.Layout(height='40px'))
skip_button = widgets.Button(description='⏭️ Skip', button_style='danger', layout=widgets.Layout(height='40px'))
save_all_button = widgets.Button(description='💾 Save All Annotations', button_style='primary', layout=widgets.Layout(height='40px'))

# Progress display
progress_bar = widgets.IntProgress(
    value=0, min=0, max=annotation_state['total_examples'],
    description='Progress:', layout=widgets.Layout(width='600px')
)
stats_label = widgets.HTML(value="<p style='font-size: 16px;'>No annotations yet</p>")

# Store in global dict
annotation_widgets = {
    'nav_label': nav_label,
    'prev_button': prev_button,
    'next_button': next_button,
    'content_area': content_area,
    'relevance_slider': relevance_slider,
    'topics_text': topics_text,
    'companies_text': companies_text,
    'summary_text': summary_text,
    'quality_slider': quality_slider,
    'notes_text': notes_text,
    'approve_button': approve_button,
    'correct_button': correct_button,
    'skip_button': skip_button,
    'save_all_button': save_all_button,
    'progress_bar': progress_bar,
    'stats_label': stats_label
}

print("✅ Widgets created")

# ============================================================================
# Annotation Logic (Dark Theme with Large Fonts)
# ============================================================================

def load_example(index):
    """Load an example into the annotation interface with dark theme and large fonts"""
    if index < 0 or index >= annotation_state['total_examples']:
        return

    example = loaded_examples[index]
    annotation_state['current_index'] = index

    # Update navigation
    annotation_widgets['nav_label'].value = f"<h2 style='font-size: 24px;'>📝 Example {index + 1} of {annotation_state['total_examples']}</h2>"
    annotation_widgets['prev_button'].disabled = (index == 0)
    annotation_widgets['next_button'].disabled = (index == annotation_state['total_examples'] - 1)

    # Display content with dark theme and LARGER font
    user_content = example['messages'][0]['content']
    assistant_content = example['messages'][1]['content']

    content_html = f"""
    <div style="background: #2d2d2d; padding: 25px; border-radius: 5px; margin: 10px 0; border: 1px solid #404040;">
        <div style="color: #64B5F6;">
            <h3 style='font-size: 24px; margin-bottom: 15px;'>👤 User Prompt:</h3>
            <pre style="white-space: pre-wrap; background: #1a1a1a; padding: 20px; border-left: 3px solid #64B5F6; border-radius: 3px; color: #e0e0e0; font-family: 'Courier New', monospace; font-size: 18px; line-height: 1.8;">{user_content}</pre>
        </div>

        <div style="color: #81C784; margin-top: 25px;">
            <h3 style='font-size: 24px; margin-bottom: 15px;'>🤖 Assistant Response:</h3>
            <pre style="white-space: pre-wrap; background: #1a1a1a; padding: 20px; border-left: 3px solid #81C784; border-radius: 3px; color: #e0e0e0; font-family: 'Courier New', monospace; font-size: 18px; line-height: 1.8;">{assistant_content}</pre>
        </div>
    </div>
    """

    annotation_widgets['content_area'].value = content_html

    # Load existing annotation if any
    if index in annotation_state['annotations']:
        ann = annotation_state['annotations'][index]
        annotation_widgets['relevance_slider'].value = ann.get('relevance_score', 5)
        annotation_widgets['topics_text'].value = '\n'.join(ann.get('topics', []))
        annotation_widgets['companies_text'].value = '\n'.join(ann.get('companies', []))
        annotation_widgets['summary_text'].value = ann.get('summary', '')
        annotation_widgets['quality_slider'].value = ann.get('quality', 7)
        annotation_widgets['notes_text'].value = ann.get('notes', '')
    else:
        # Reset fields
        annotation_widgets['relevance_slider'].value = 5
        annotation_widgets['topics_text'].value = ''
        annotation_widgets['companies_text'].value = ''
        annotation_widgets['summary_text'].value = ''
        annotation_widgets['quality_slider'].value = 7
        annotation_widgets['notes_text'].value = ''

    update_stats()

def save_annotation(status):
    """Save current annotation"""
    index = annotation_state['current_index']

    annotation = {
        'relevance_score': annotation_widgets['relevance_slider'].value,
        'topics': [t.strip() for t in annotation_widgets['topics_text'].value.split('\n') if t.strip()],
        'companies': [c.strip() for c in annotation_widgets['companies_text'].value.split('\n') if c.strip()],
        'summary': annotation_widgets['summary_text'].value.strip(),
        'quality': annotation_widgets['quality_slider'].value,
        'notes': annotation_widgets['notes_text'].value.strip(),
        'status': status
    }

    annotation_state['annotations'][index] = annotation

    # Update status sets
    annotation_state['approved'].discard(index)
    annotation_state['corrected'].discard(index)
    annotation_state['skipped'].discard(index)

    if status == 'approved':
        annotation_state['approved'].add(index)
    elif status == 'corrected':
        annotation_state['corrected'].add(index)
    elif status == 'skipped':
        annotation_state['skipped'].add(index)

    update_stats()

def update_stats():
    """Update progress statistics with dark theme and large fonts"""
    total = annotation_state['total_examples']
    annotated = len(annotation_state['annotations'])
    approved = len(annotation_state['approved'])
    corrected = len(annotation_state['corrected'])
    skipped = len(annotation_state['skipped'])

    annotation_widgets['progress_bar'].value = annotated

    stats_html = f"""
    <div style="background: #2d2d2d; padding: 20px; border-radius: 5px; border: 1px solid #404040; color: #e0e0e0; font-size: 16px; line-height: 1.8;">
        <strong style="color: #64B5F6; font-size: 18px;">📊 Annotation Statistics:</strong><br><br>
        <span style="color: #81C784; font-size: 16px;">✅ Approved: {approved}</span><br>
        <span style="color: #FFB74D; font-size: 16px;">✏️ Corrected: {corrected}</span><br>
        <span style="color: #E57373; font-size: 16px;">⏭️ Skipped: {skipped}</span><br>
        <span style="color: #64B5F6; font-size: 16px;">📝 Total Annotated: {annotated} / {total} ({100*annotated/total:.1f}%)</span>
    </div>
    """
    annotation_widgets['stats_label'].value = stats_html

# Button handlers
def on_prev_click(b):
    load_example(annotation_state['current_index'] - 1)

def on_next_click(b):
    load_example(annotation_state['current_index'] + 1)

def on_approve_click(b):
    save_annotation('approved')
    if annotation_state['current_index'] < annotation_state['total_examples'] - 1:
        load_example(annotation_state['current_index'] + 1)

def on_correct_click(b):
    save_annotation('corrected')
    if annotation_state['current_index'] < annotation_state['total_examples'] - 1:
        load_example(annotation_state['current_index'] + 1)

def on_skip_click(b):
    save_annotation('skipped')
    if annotation_state['current_index'] < annotation_state['total_examples'] - 1:
        load_example(annotation_state['current_index'] + 1)

# ============================================================================
# 🔧 CRITICAL FIX: Save All Annotations with Merge Logic (APPEND, NOT OVERWRITE)
# ============================================================================
def on_save_all_click(b):
    """Save all annotations to file (with merge to prevent duplicates and preserve existing)"""
    output_path = f"{DRIVE_PATH}/training_data/news_content_training_annotated.jsonl"

    # ✅ STEP 1: Load existing annotations from file
    existing_annotations = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        existing_annotations.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    print(f"\n📋 Found {len(existing_annotations)} existing annotations in file")

    # ✅ STEP 2: Create set of existing content for deduplication
    existing_contents = {
        json.dumps(ex['messages'], sort_keys=True)
        for ex in existing_annotations
    }

    # ✅ STEP 3: Collect new annotations from current session
    new_annotations = []
    for i, example in enumerate(loaded_examples):
        if i in annotation_state['annotations']:
            ann = annotation_state['annotations'][i]
            if ann['status'] != 'skipped':  # Don't include skipped examples
                example_copy = example.copy()
                example_copy['annotation'] = ann

                # Check if not already saved (prevent duplicates)
                ex_content = json.dumps(example_copy['messages'], sort_keys=True)
                if ex_content not in existing_contents:
                    new_annotations.append(example_copy)

    # ✅ STEP 4: Append only new annotations (CRITICAL: 'a' mode, not 'w')
    if new_annotations:
        with open(output_path, 'a', encoding='utf-8') as f:  # ✅ 'a' = APPEND MODE
            for ex in new_annotations:
                f.write(json.dumps(ex) + '\n')

        total_now = len(existing_annotations) + len(new_annotations)
        print(f"\n✅ Successfully added {len(new_annotations)} new annotated examples")
        print(f"📊 Total annotations now: {total_now}")
        print(f"📄 Location: {output_path}")
        print(f"\n💡 Progress: {total_now}/101 annotations complete ({100*total_now/101:.1f}%)")
    else:
        print(f"\n⚠️ No new annotations to save")
        print(f"   (All {len(annotation_state['annotations'])} current annotations already exist in file)")
        print(f"📊 Total annotations: {len(existing_annotations)}")

# Connect handlers
annotation_widgets['prev_button'].on_click(on_prev_click)
annotation_widgets['next_button'].on_click(on_next_click)
annotation_widgets['approve_button'].on_click(on_approve_click)
annotation_widgets['correct_button'].on_click(on_correct_click)
annotation_widgets['skip_button'].on_click(on_skip_click)
annotation_widgets['save_all_button'].on_click(on_save_all_click)

print("✅ Annotation logic connected")

# ============================================================================
# Display Annotation Interface (Dark Theme with Large Fonts)
# ============================================================================

# Dark theme styling with larger fonts
dark_headers = """
<style>
    .dark-header {
        color: #e0e0e0;
        background: #1e1e1e;
        padding: 10px 0;
        font-size: 28px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
    }
</style>
"""

# Layout
navigation = widgets.HBox([
    annotation_widgets['prev_button'],
    annotation_widgets['nav_label'],
    annotation_widgets['next_button']
])

analysis_fields = widgets.VBox([
    annotation_widgets['relevance_slider'],
    annotation_widgets['topics_text'],
    annotation_widgets['companies_text'],
    annotation_widgets['summary_text']
])

quality_section = widgets.VBox([
    annotation_widgets['quality_slider'],
    annotation_widgets['notes_text']
])

action_buttons = widgets.HBox([
    annotation_widgets['approve_button'],
    annotation_widgets['correct_button'],
    annotation_widgets['skip_button']
])

progress_section = widgets.VBox([
    annotation_widgets['progress_bar'],
    annotation_widgets['stats_label'],
    annotation_widgets['save_all_button']
])

interface = widgets.VBox([
    widgets.HTML(dark_headers + "<div class='dark-header'><h2>✏️ Interactive Annotation Interface</h2></div>"),
    navigation,
    annotation_widgets['content_area'],
    widgets.HTML("<hr style='border-color: #404040;'><h4 style='color: #64B5F6; font-size: 20px;'>📊 Analysis Fields</h4>"),
    analysis_fields,
    widgets.HTML("<hr style='border-color: #404040;'><h4 style='color: #FFB74D; font-size: 20px;'>⭐ Quality Assessment</h4>"),
    quality_section,
    widgets.HTML("<hr style='border-color: #404040;'><h4 style='color: #81C784; font-size: 20px;'>🎯 Actions</h4>"),
    action_buttons,
    widgets.HTML("<hr style='border-color: #404040;'><h4 style='color: #64B5F6; font-size: 20px;'>📈 Progress</h4>"),
    progress_section
])

# Load first example and display
load_example(0)
display(interface)

print("\n🎯 Dark-themed annotation interface with large fonts ready! Start reviewing your training examples.")
print("\n" + "="*80)
print("🔧 CRITICAL FIX APPLIED:")
print("   ✅ Save function now APPENDS to existing annotations (not overwrites)")
print("   ✅ Prevents duplicates automatically")
print("   ✅ Preserves all previously annotated examples")
print("   ✅ Safe to save in batches!")
print("="*80)

