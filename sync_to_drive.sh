#!/bin/bash
# Sync local notebook and workspace to Google Drive using Google Workspace CLI (gws)

set -e

# Define files/folders to sync
NOTEBOOK="InCTRL_TextualAdapter_Ablation.ipynb"
WORKSPACE_DIR="."
ARCHIVE_NAME="InCTRL_Workspace_Backup.tar.gz"
CLOUD_FOLDER_NAME="InCTRL_Cloud_Sync"

# 1. Check authentication
echo "🔄 Checking Google Workspace CLI authentication..."
if ! gws drive about get --params '{"fields": "user(displayName,emailAddress)"}' > /dev/null 2>&1; then
    echo "❌ You are not authenticated with Google Workspace CLI."
    echo "Please run the following commands first:"
    echo "  1. gws auth setup  (Follow instructions to configure OAuth client)"
    echo "  2. gws auth login  (Log in with your Google account)"
    exit 1
fi

echo "✅ Authenticated successfully."

# 2. Package the workspace (excluding heavy/unnecessary items if you want)
echo "📦 Packaging workspace into $ARCHIVE_NAME..."
# Excluding .git and other potential massive artifacts like datasets if they shouldn't be synced every time
tar --exclude="./.git" \
    --exclude="./datasets/*" \
    --exclude="./$ARCHIVE_NAME" \
    -czf "$ARCHIVE_NAME" "$WORKSPACE_DIR"

# 3. Find or Create Cloud Folder
echo "🔍 Searching for cloud folder '$CLOUD_FOLDER_NAME'..."
FOLDER_JSON=$(gws drive files list \
    --params "{\"q\": \"name = '$CLOUD_FOLDER_NAME' and mimeType = 'application/vnd.google-apps.folder' and trashed = false\", \"fields\": \"files(id,name)\"}" 2>/dev/null)

FOLDER_ID=$(echo "$FOLDER_JSON" | grep -o '"id": "[^"]*' | cut -d'"' -f4 | head -n 1 || true)

if [ -z "$FOLDER_ID" ]; then
    echo "📁 Folder not found. Creating '$CLOUD_FOLDER_NAME'..."
    CREATE_JSON=$(gws drive files create \
        --json "{\"name\": \"$CLOUD_FOLDER_NAME\", \"mimeType\": \"application/vnd.google-apps.folder\"}" \
        --params '{"fields": "id"}')
    FOLDER_ID=$(echo "$CREATE_JSON" | grep -o '"id": "[^"]*' | cut -d'"' -f4 | head -n 1)
    echo "✅ Created folder with ID: $FOLDER_ID"
else
    echo "✅ Found existing folder with ID: $FOLDER_ID"
fi

# 4. Upload / Update Notebook
echo "☁️ Syncing $NOTEBOOK to cloud..."
# Check if notebook exists in the folder
NB_JSON=$(gws drive files list \
    --params "{\"q\": \"name = '$NOTEBOOK' and '$FOLDER_ID' in parents and trashed = false\", \"fields\": \"files(id)\"}")
NB_ID=$(echo "$NB_JSON" | grep -o '"id": "[^"]*' | cut -d'"' -f4 | head -n 1 || true)

if [ -z "$NB_ID" ]; then
    # Create new
    gws drive files create \
        --json "{\"name\": \"$NOTEBOOK\", \"parents\": [\"$FOLDER_ID\"]}" \
        --upload "$NOTEBOOK" \
        --params '{"fields": "id"}' > /dev/null
    echo "✅ Uploaded $NOTEBOOK (New)"
else
    # Update existing
    gws drive files update \
        --params "{\"fileId\": \"$NB_ID\", \"fields\": \"id\"}" \
        --upload "$NOTEBOOK" > /dev/null
    echo "✅ Updated $NOTEBOOK (Existing)"
fi

# 5. Upload / Update Workspace Archive
echo "☁️ Syncing $ARCHIVE_NAME to cloud..."
ARC_JSON=$(gws drive files list \
    --params "{\"q\": \"name = '$ARCHIVE_NAME' and '$FOLDER_ID' in parents and trashed = false\", \"fields\": \"files(id)\"}")
ARC_ID=$(echo "$ARC_JSON" | grep -o '"id": "[^"]*' | cut -d'"' -f4 | head -n 1 || true)

if [ -z "$ARC_ID" ]; then
    # Create new
    gws drive files create \
        --json "{\"name\": \"$ARCHIVE_NAME\", \"parents\": [\"$FOLDER_ID\"]}" \
        --upload "$ARCHIVE_NAME" \
        --params '{"fields": "id"}' > /dev/null
    echo "✅ Uploaded $ARCHIVE_NAME (New)"
else
    # Update existing
    gws drive files update \
        --params "{\"fileId\": \"$ARC_ID\", \"fields\": \"id\"}" \
        --upload "$ARCHIVE_NAME" > /dev/null
    echo "✅ Updated $ARCHIVE_NAME (Existing)"
fi

echo "🎉 Sync completed successfully!"
