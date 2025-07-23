#!/bin/bash

# Script to rename all instances of policycortex to policortex
echo "Renaming policycortex to policortex in all Bicep files..."

# Find all bicep files and replace policycortex with policortex (case-insensitive)
find ./infrastructure/bicep -name "*.bicep" -type f -exec sed -i 's/policycortex/policortex/gi' {} \;

# Also check for any PolicyCortex (capitalized) instances
find ./infrastructure/bicep -name "*.bicep" -type f -exec sed -i 's/PolicyCortex/Policortex/g' {} \;

echo "Replacement complete!"
echo ""
echo "Files modified:"
find ./infrastructure/bicep -name "*.bicep" -type f -exec grep -l "policortex" {} \;