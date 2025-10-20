# 📝 Data Preparation and Annotation

## Overview

This document describes the data preparation process for fine-tuning LLaMA 3.2-1B with LoRA adapters. The process involves collecting newsletter content, manual annotation, and formatting into JSONL for training.

---

## Dataset Overview

### **Final Dataset Statistics**

| Split | Examples | Purpose |
|-------|----------|---------|
| **Training** | 101 | Fine-tuning LoRA adapters |
| **Test** | 20 | Evaluation and comparison |
| **Total** | 121 | Complete annotated dataset |

### **Data Split Rationale**

- **80/20 split** (approximately)
- **Training set**: Sufficient for LoRA fine-tuning (50-200 examples recommended)
- **Test set**: Large enough for statistical significance
- **No validation set**: Small dataset size makes separate validation unnecessary

---

## Data Format

### **JSONL Structure**

Each line in the JSONL file represents one training example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze this newsletter:\n\n[Newsletter content here...]"
    },
    {
      "role": "assistant",
      "content": "{\"relevance_score\": 8, \"summary\": \"...\", \"insights\": [...]}"
    }
  ]
}
```

### **Message Format**

**User Message**:
- Prompt template: `"Analyze this newsletter:\n\n{newsletter_content}"`
- Content: Raw newsletter text (email body)
- Length: 500-3000 characters (truncated if longer)

**Assistant Message**:
- Format: JSON string
- Required fields:
  - `relevance_score`: Integer 1-10
  - `summary`: String (concise summary)
  - `insights`: Array of strings (key takeaways)

### **Example**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze this newsletter:\n\nSubject: AI Advances in 2024\n\nOpenAI released GPT-4 Turbo with improved performance..."
    },
    {
      "role": "assistant",
      "content": "{\"relevance_score\": 9, \"summary\": \"OpenAI announces GPT-4 Turbo with significant performance improvements and cost reductions.\", \"insights\": [\"GPT-4 Turbo is 3x cheaper than GPT-4\", \"128K context window enables longer conversations\", \"Improved instruction following\"]}"
    }
  ]
}
```

---

## Annotation Process

### **Step 1: Data Collection**

**Source**: Real newsletter emails from Gmail

**Collection Method**:
1. Export newsletters from Gmail
2. Extract email body content
3. Clean HTML formatting
4. Truncate to manageable length (500-3000 chars)

**Quality Criteria**:
- ✅ Diverse topics (tech, business, research)
- ✅ Varied newsletter sources
- ✅ Different content structures
- ✅ Mix of short and long newsletters

### **Step 2: Manual Annotation**

**Tool**: Custom Google Colab annotation widget

**Annotation Interface**:
```python
# Interactive widget with:
# - Newsletter content display
# - Relevance score slider (1-10)
# - Summary text area
# - Insights list (add/remove)
# - Save button
```

**Annotation Guidelines**:

1. **Relevance Score (1-10)**:
   - 1-3: Low relevance (spam, promotional)
   - 4-6: Medium relevance (general news)
   - 7-9: High relevance (actionable insights)
   - 10: Critical (urgent, high-impact)

2. **Summary**:
   - Length: 1-2 sentences
   - Focus: Main topic and key message
   - Style: Concise, informative
   - Example: "OpenAI announces GPT-4 Turbo with 3x cost reduction and 128K context window."

3. **Insights**:
   - Count: 2-5 insights per newsletter
   - Format: Bullet points
   - Content: Actionable takeaways, key facts, important details
   - Example: ["GPT-4 Turbo is 3x cheaper", "128K context window", "Improved instruction following"]

### **Step 3: Quality Control**

**Validation Checks**:
- ✅ JSON format validity
- ✅ Required fields present
- ✅ Relevance score in range (1-10)
- ✅ Summary not empty
- ✅ At least 2 insights

**Consistency Checks**:
- ✅ Similar newsletters have similar scores
- ✅ Summaries are concise (not copy-paste)
- ✅ Insights are distinct (not redundant)

### **Step 4: Data Splitting**

**Method**: Random split with stratification

```python
from sklearn.model_selection import train_test_split

# Split 80/20
train_data, test_data = train_test_split(
    annotated_data,
    test_size=0.2,
    random_state=42,
    stratify=relevance_scores  # Ensure balanced distribution
)
```

**Verification**:
- ✅ Training set: 101 examples
- ✅ Test set: 20 examples
- ✅ No overlap between sets
- ✅ Similar relevance score distribution

---

## Annotation Widget

### **Features**

1. **Newsletter Display**:
   - Shows full newsletter content
   - Syntax highlighting for readability
   - Scrollable for long content

2. **Annotation Controls**:
   - Relevance score slider (1-10)
   - Summary text area (auto-resize)
   - Insights list (add/remove dynamically)
   - Save button (validates and saves to JSONL)

3. **Progress Tracking**:
   - Shows current example number
   - Displays total examples
   - Progress bar

4. **Validation**:
   - Real-time JSON validation
   - Error messages for invalid input
   - Prevents saving incomplete annotations

### **Usage**

```python
# In Google Colab
from annotation_widget import AnnotationWidget

# Initialize widget
widget = AnnotationWidget(
    input_file='raw_newsletters.jsonl',
    output_file='annotated_newsletters.jsonl'
)

# Display widget
widget.display()

# Annotate each newsletter:
# 1. Read newsletter content
# 2. Set relevance score (1-10)
# 3. Write summary
# 4. Add insights
# 5. Click "Save"
# 6. Move to next newsletter
```

### **Code**

See `notebooks/JSONL_Annotation_Notebook_Final.ipynb` for complete implementation.

---

## Data Quality Metrics

### **Annotation Consistency**

| Metric | Value |
|--------|-------|
| **Inter-annotator agreement** | N/A (single annotator) |
| **Annotation time** | ~2-3 minutes per newsletter |
| **Total annotation time** | ~4-6 hours for 121 examples |

### **Content Diversity**

| Category | Count | Percentage |
|----------|-------|------------|
| **Technology** | 45 | 37% |
| **Business** | 32 | 26% |
| **Research** | 28 | 23% |
| **Other** | 16 | 13% |

### **Relevance Score Distribution**

| Score Range | Count | Percentage |
|-------------|-------|------------|
| **1-3 (Low)** | 12 | 10% |
| **4-6 (Medium)** | 48 | 40% |
| **7-9 (High)** | 55 | 45% |
| **10 (Critical)** | 6 | 5% |

---

## Data Augmentation

### **Techniques Considered**

1. **Paraphrasing**: Not used (risk of changing meaning)
2. **Back-translation**: Not used (small dataset, quality concerns)
3. **Synthetic generation**: Not used (prefer real newsletters)

### **Rationale**

- LoRA fine-tuning works well with 50-200 examples
- Real data preferred over synthetic for this use case
- Quality over quantity for semantic understanding tasks

---

## Data Storage

### **File Locations**

```
data/
└── processed/
    ├── newsletter_train_data.jsonl    # 101 training examples
    └── newsletter_test_data.jsonl     # 20 test examples
```

### **File Sizes**

| File | Size | Lines |
|------|------|-------|
| `newsletter_train_data.jsonl` | ~150 KB | 101 |
| `newsletter_test_data.jsonl` | ~30 KB | 20 |

### **Backup**

- ✅ Stored in Google Drive
- ✅ Version controlled in Git
- ✅ Backed up locally

---

## Lessons Learned

### **What Worked Well**

1. ✅ **Custom annotation widget** - Much faster than manual JSON editing
2. ✅ **Real newsletter data** - Better than synthetic examples
3. ✅ **Structured JSON output** - Easy to validate and parse
4. ✅ **Small dataset** - Sufficient for LoRA fine-tuning

### **Challenges**

1. ⚠️ **Time-consuming** - 4-6 hours for 121 examples
2. ⚠️ **Subjectivity** - Relevance scores can be subjective
3. ⚠️ **HTML cleaning** - Some newsletters had complex formatting

### **Future Improvements**

1. **Semi-automated annotation** - Use base model to suggest annotations
2. **Multi-annotator** - Get multiple annotations for inter-rater reliability
3. **Active learning** - Prioritize uncertain examples for annotation
4. **Expand dataset** - Aim for 200-500 examples for better coverage

---

## Reproducibility

### **Annotation Workflow**

1. **Collect newsletters** from Gmail (export as MBOX or use Gmail API)
2. **Clean and format** newsletter content (remove HTML, truncate)
3. **Load annotation widget** in Google Colab
4. **Annotate each newsletter** (relevance score, summary, insights)
5. **Validate annotations** (JSON format, required fields)
6. **Split data** (80/20 train/test)
7. **Save to JSONL** files

### **Tools Required**

- Google Colab (for annotation widget)
- Python 3.8+
- Libraries: `ipywidgets`, `jsonschema`, `sklearn`

### **Time Estimate**

- Data collection: 1-2 hours
- Annotation: 4-6 hours (121 examples)
- Validation and splitting: 30 minutes
- **Total**: 6-9 hours

---

## References

- **JSONL Format**: [JSON Lines](https://jsonlines.org/)
- **LoRA Data Requirements**: Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- **Annotation Best Practices**: Pustejovsky & Stubbs (2012). Natural Language Annotation for Machine Learning.

