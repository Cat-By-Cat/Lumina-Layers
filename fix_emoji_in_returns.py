#!/usr/bin/env python3
"""
Safely replace emojis ONLY in return statements.
"""

import re
import os

# Emoji replacements
REPLACEMENTS = {
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
}

def fix_file(filepath):
    """Fix emojis in return statements only."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    for i, line in enumerate(lines):
        # Only process lines that contain 'return' and an emoji
        if 'return' in line and any(emoji in line for emoji in REPLACEMENTS):
            original = line
            for emoji, replacement in REPLACEMENTS.items():
                line = line.replace(emoji, replacement)
            if line != original:
                lines[i] = line
                modified = True
                print(f"  Line {i+1}: {original.strip()[:60]}...")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return True
    return False

# Files to fix
files = [
    'utils/lut_manager.py',
    'ui/callbacks.py',
    'ui/layout_new.py',
    'core/converter.py',
    'core/extractor.py',
    'core/heightmap_loader.py',
    'core/calibration.py',
]

print("Fixing emojis in return statements...\n")
for filepath in files:
    if os.path.exists(filepath):
        print(f"Processing {filepath}...")
        if fix_file(filepath):
            print(f"  ✓ Modified\n")
        else:
            print(f"  - No changes\n")

print("Done!")
