# Clean History Branch - Setup Instructions

## What Has Been Done

I've created a new branch called `clean-history` locally with a clean commit history. This branch contains all the same files as the main branch but with only **1 commit** instead of **59 commits**.

## Current Status

- ✅ **clean-history branch created locally** - Contains 1 clean commit with all repository content
- ✅ **Content verified** - All files are identical to the main branch
- ⚠️ **Not yet pushed to GitHub** - Due to permission restrictions, I cannot push directly

## How to Complete the Setup

### Step 1: Push the Clean History Branch

From your local machine, you need to push the clean-history branch:

```bash
# Navigate to your repository
cd /path/to/TFM-APB-MAADM

# Fetch all branches
git fetch --all

# Check out the clean-history branch
git checkout clean-history

# Verify it only has 1 commit
git log --oneline

# Push to GitHub
git push -u origin clean-history
```

### Step 2: Make Clean History the Default Branch

1. Go to your repository on GitHub: https://github.com/Alejandro-Pardo/TFM-APB-MAADM
2. Click on **Settings**
3. Click on **Branches** in the left sidebar
4. Under "Default branch", click the switch button
5. Select **clean-history** from the dropdown
6. Click **Update** and confirm the change

### Step 3: Delete the Old Main Branch

Once the clean-history branch is set as the default, you can delete the old main branch:

**Option A: Via GitHub UI**
1. Go to your repository on GitHub
2. Click on **Branches** (above the file list)
3. Find the **main** branch in the list
4. Click the trash icon to delete it

**Option B: Via Command Line**
```bash
git push origin --delete main
```

### Step 4 (Optional): Rename Clean-History to Main

If you prefer to keep the branch name as "main":

```bash
# On your local machine
git branch -m clean-history main
git push origin -u main

# Update GitHub's default branch to point to the new main
# Then delete clean-history if needed
```

## Alternative Approach: Force Push to Main

**⚠️ WARNING: This permanently deletes commit history**

If you have force-push permissions, you can directly replace main's history:

```bash
git checkout clean-history
git push origin clean-history:main --force
```

Note: This requires that branch protection rules allow force pushes.

## Verification

After completing the setup, verify everything worked:

```bash
# Check the commit count
git checkout main  # or clean-history
git log --oneline  # Should show only 1 commit

# Verify all files are present
git status
```

## Important Considerations

1. **Backup**: The original commit history will be lost once you delete the main branch. Consider keeping a backup branch:
   ```bash
   git checkout main
   git checkout -b main-backup
   git push origin main-backup
   ```

2. **Team Notification**: If others are working on this repository, notify them about the change. They'll need to:
   ```bash
   git fetch --all
   git checkout clean-history  # or the new main
   ```

3. **Pull Requests**: Any open pull requests targeting main will need to be retargeted to the new default branch.

4. **CI/CD**: Update any CI/CD configurations that reference the main branch.

## Need Help?

If you encounter any issues or need clarification, please let me know!
