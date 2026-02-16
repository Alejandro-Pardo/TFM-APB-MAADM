#!/bin/bash
# Script to push the clean-history branch and optionally replace main
# Usage: ./push_clean_history.sh

set -e  # Exit on error

echo "=========================================="
echo "Clean History Branch Setup Script"
echo "=========================================="
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Error: This script must be run from the root of the repository"
    exit 1
fi

# Check if clean-history branch exists
if ! git show-ref --verify --quiet refs/heads/clean-history; then
    echo "Error: clean-history branch does not exist locally"
    echo "Please fetch it first: git fetch origin clean-history"
    exit 1
fi

echo "Step 1: Checking out clean-history branch..."
git checkout clean-history

echo ""
echo "Step 2: Verifying clean history (should show only 1 commit)..."
git log --oneline
echo ""

# Count commits
commit_count=$(git rev-list --count HEAD)
echo "Total commits in clean-history: $commit_count"
echo ""

if [ "$commit_count" -ne 1 ]; then
    echo "Warning: Expected 1 commit, but found $commit_count commits"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted"
        exit 1
    fi
fi

echo "Step 3: Pushing clean-history branch to origin..."
git push -u origin clean-history
echo "✓ Successfully pushed clean-history branch"
echo ""

echo "=========================================="
echo "Next Steps (Manual):"
echo "=========================================="
echo ""
echo "1. Go to GitHub repository Settings → Branches"
echo "2. Change default branch from 'main' to 'clean-history'"
echo "3. Delete the old 'main' branch:"
echo "   - Via GitHub UI: Branches page → Delete main"
echo "   - Via command: git push origin --delete main"
echo ""
echo "4. (Optional) Rename clean-history to main:"
echo "   git branch -m clean-history main"
echo "   git push origin -u main"
echo ""
echo "For detailed instructions, see CLEAN_HISTORY_SETUP.md"
echo ""
