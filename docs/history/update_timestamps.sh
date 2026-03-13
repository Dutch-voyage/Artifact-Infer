#!/bin/bash

# Script to update content.md with correct filesystem timestamps

echo "Updating content.md with correct timestamps..."

# Count total documents
total_docs=$(find /home/yyx/efficient_inference/xylo-infer/docs -name '*.md' -not -name 'content.md' | wc -l)

# Create the new content.md file
cat > /home/yyx/efficient_inference/xylo-infer/docs/content.md << EOF
# Documentation Index
*Last updated: $(date '+%Y-%m-%d %H:%M:%S %Z')*

## Analysis
*Requirements and architectural analysis documents*

| File | Created | Modified |
|------|---------|----------|
EOF

# Analysis files
for file in /home/yyx/efficient_inference/xylo-infer/docs/analysis/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        created=$(stat -c "%W" "$file" 2>/dev/null | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        modified=$(stat -c "%y" "$file" | cut -d'.' -f1)
        echo "| [$filename](./analysis/$filename) | $created | $modified |" >> /home/yyx/efficient_inference/xylo-infer/docs/content.md
    fi
done

cat >> /home/yyx/efficient_inference/xylo-infer/docs/content.md << EOF

## Architecture
*System architecture and planning documents*

| File | Created | Modified |
|------|---------|----------|
EOF

# Architecture files
for file in /home/yyx/efficient_inference/xylo-infer/docs/architecture/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        created=$(stat -c "%W" "$file" 2>/dev/null | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        modified=$(stat -c "%y" "$file" | cut -d'.' -f1)
        echo "| [$filename](./architecture/$filename) | $created | $modified |" >> /home/yyx/efficient_inference/xylo-infer/docs/content.md
    fi
done

cat >> /home/yyx/efficient_inference/xylo-infer/docs/content.md << EOF

## Design
*Design specifications and implementation plans*

| File | Created | Modified |
|------|---------|----------|
EOF

# Design files
for file in /home/yyx/efficient_inference/xylo-infer/docs/design/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        created=$(stat -c "%W" "$file" 2>/dev/null | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        modified=$(stat -c "%y" "$file" | cut -d'.' -f1)
        echo "| [$filename](./design/$filename) | $created | $modified |" >> /home/yyx/efficient_inference/xylo-infer/docs/content.md
    fi
done

cat >> /home/yyx/efficient_inference/xylo-infer/docs/content.md << EOF

## System
*Core system documentation*

| File | Created | Modified |
|------|---------|----------|
EOF

# System files
for file in /home/yyx/efficient_inference/xylo-infer/docs/system/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        created=$(stat -c "%W" "$file" 2>/dev/null | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        modified=$(stat -c "%y" "$file" | cut -d'.' -f1)
        echo "| [$filename](./system/$filename) | $created | $modified |" >> /home/yyx/efficient_inference/xylo-infer/docs/content.md
    fi
done

cat >> /home/yyx/efficient_inference/xylo-infer/docs/content.md << EOF

## Guides
*User guides and documentation*

| File | Created | Modified |
|------|---------|----------|
EOF

# Guides files
for file in /home/yyx/efficient_inference/xylo-infer/docs/guides/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        created=$(stat -c "%W" "$file" 2>/dev/null | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        modified=$(stat -c "%y" "$file" | cut -d'.' -f1)
        echo "| [$filename](./guides/$filename) | $created | $modified |" >> /home/yyx/efficient_inference/xylo-infer/docs/content.md
    fi
done

cat >> /home/yyx/efficient_inference/xylo-infer/docs/content.md << EOF

## Specifications
*Technical specifications and requirements*

| File | Created | Modified |
|------|---------|----------|
EOF

# Specifications files
for file in /home/yyx/efficient_inference/xylo-infer/docs/specifications/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        created=$(stat -c "%W" "$file" 2>/dev/null | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        modified=$(stat -c "%y" "$file" | cut -d'.' -f1)
        echo "| [$filename](./specifications/$filename) | $created | $modified |" >> /home/yyx/efficient_inference/xylo-infer/docs/content.md
    fi
done

cat >> /home/yyx/efficient_inference/xylo-infer/docs/content.md << EOF

---
*Total: $total_docs documents across 6 categories*
EOF

echo "Updated content.md with correct filesystem timestamps!"
echo "Total documents: $total_docs"