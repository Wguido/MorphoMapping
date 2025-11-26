#!/bin/bash
# Quick script to check which files still need translation

echo "Checking documentation files for German content..."
echo ""

files=(
  "USER_GUIDE.md"
  "GITHUB_ISSUES.md"
  "GITHUB_ISSUE_TEMPLATES.md"
  "DOCUMENTATION_INDEX.md"
)

for file in "${files[@]}"; do
  if grep -q "Übersicht\|Anleitung\|Schritt\|Problem\|Lösung\|Tipp\|Hinweis" "$file" 2>/dev/null; then
    echo "❌ $file: Still contains German"
  else
    echo "✅ $file: OK"
  fi
done

