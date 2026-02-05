# GitHub Pages Setup Guide

This guide will walk you through the steps to publish your MinexPy documentation to GitHub Pages.

## Prerequisites

- A GitHub account
- Your repository pushed to GitHub
- The `pagestest` branch (or `main` branch) ready

## Step-by-Step Instructions

### Step 1: Update Repository URLs

Before pushing, update the repository URLs in `docs/mkdocs.yml`:

1. Open `docs/mkdocs.yml`
2. Replace `yourusername` with your actual GitHub username in:
   - `site_url`: `https://yourusername.github.io/MinexPy/`
   - `repo_url`: `https://github.com/yourusername/MinexPy`
3. Update `edit_uri` if your default branch is not `pagestest`

### Step 2: Push Your Branch to GitHub

```bash
# Make sure you're on the pagestest branch
git checkout pagestest

# Add all files
git add .

# Commit your changes
git commit -m "Add statistical analysis module and GitHub Pages setup"

# Push to GitHub (first time)
git push -u origin pagestest

# Or if branch already exists
git push
```

### Step 3: Enable GitHub Pages

1. Go to your GitHub repository on GitHub.com
2. Click on **Settings** (top menu bar)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Source**: `GitHub Actions`
   - (Not "Deploy from a branch")
5. Save the settings

### Step 4: Grant Permissions (First Time Only)

If this is the first time setting up GitHub Pages:

1. Go to **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select:
   - **Read and write permissions**
   - Check **Allow GitHub Actions to create and approve pull requests**
3. Save the changes

### Step 5: Trigger the Workflow

The GitHub Actions workflow will automatically run when you:
- Push to the `pagestest` branch (or `main` if configured)
- Manually trigger it from the Actions tab

To manually trigger:

1. Go to the **Actions** tab in your repository
2. Select **Deploy Documentation to GitHub Pages** workflow
3. Click **Run workflow** → **Run workflow**

### Step 6: Wait for Deployment

1. Go to the **Actions** tab
2. You'll see the workflow running
3. Wait for it to complete (usually 2-5 minutes)
4. When it shows a green checkmark, it's deployed!

### Step 7: Access Your Documentation

Your documentation will be available at:
```
https://yourusername.github.io/MinexPy/
```

Replace `yourusername` with your GitHub username.

## Troubleshooting

### Workflow Fails

1. Check the **Actions** tab for error messages
2. Common issues:
   - Missing dependencies: Check `docs/requirements.txt`
   - Python version: Ensure Python 3.9+ is used
   - Path issues: Make sure `mkdocs.yml` is in the `docs/` directory

### Documentation Not Updating

1. Check that the workflow completed successfully
2. Clear your browser cache
3. Wait a few minutes (GitHub Pages can take time to update)
4. Check the workflow logs in the **Actions** tab

### Permission Errors

1. Go to **Settings** → **Actions** → **General**
2. Ensure **Workflow permissions** is set to **Read and write**
3. Re-run the workflow

## Testing Locally

Before pushing, test your documentation locally:

```bash
# Install dependencies
cd docs
pip install -r requirements.txt

# Build the site
mkdocs build

# Serve locally (optional, for preview)
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

## Updating Documentation

Every time you push changes to the `pagestest` branch (or `main`), the documentation will automatically rebuild and deploy. No manual steps needed!

## Next Steps

- Update the `site_url` and `repo_url` in `mkdocs.yml` with your actual GitHub username
- Add more content to your documentation
- Customize the theme in `mkdocs.yml`
- Add more examples and tutorials

