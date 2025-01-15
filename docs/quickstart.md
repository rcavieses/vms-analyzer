## docs/quickstart.md
```markdown
# Quick Start Guide

## Basic Usage

```python
from pathlib import Path
from vms_analyzer import VMSAnalyzer

# Initialize analyzer
analyzer = VMSAnalyzer(base_path=Path("./analysis"))

# Run complete analysis
unified_file = analyzer.unify_files()
filtered_file = analyzer.filter_by_polygon(unified_file)
classified_file = analyzer.classify_fishing_activity(filtered_file)
results = analyzer.analyze_fishing_effort(classified_file)