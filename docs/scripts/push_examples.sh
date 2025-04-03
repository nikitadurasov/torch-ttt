#!/bin/bash

set -e

# Configuration
PAGES_REPO="https://github.com/torch-ttt/torch-ttt.github.io.git"
BRANCH="main"
TMP_DIR=$(mktemp -d)
BUILD_DIR="../_build/html"
EXAMPLES_DIR="$BUILD_DIR/auto_examples"
DOWNLOADS_DIR="$EXAMPLES_DIR/downloads"
IMAGES_DIR="$BUILD_DIR/_images"

# Check that auto_examples exists
if [ ! -d "$EXAMPLES_DIR" ]; then
  echo "❌ auto_examples not found. Did you build the docs? ($EXAMPLES_DIR)"
  exit 1
fi

echo "📦 Collecting all .ipynb files from $BUILD_DIR into $DOWNLOADS_DIR..."
mkdir -p "$DOWNLOADS_DIR"

# Find and copy all notebooks into the auto_examples/downloads/ folder
find "$BUILD_DIR" -name "*.ipynb" -exec cp {} "$DOWNLOADS_DIR" \;

echo "🚀 Cloning GitHub Pages repo using HTTPS..."
git clone --depth 1 --branch "$BRANCH" "$PAGES_REPO" "$TMP_DIR"

echo "🧹 Removing old auto_examples in GitHub Pages repo..."
rm -rf "$TMP_DIR/auto_examples"
rm -rf "$TMP_DIR/_images"

echo "📁 Copying new auto_examples (with notebooks)..."
cp -r "$EXAMPLES_DIR" "$TMP_DIR/auto_examples"

echo "🖼️ Copying _images directory..."
cp -r "$IMAGES_DIR" "$TMP_DIR/_images"

echo "📦 Committing and pushing changes..."
cd "$TMP_DIR"
git add auto_examples
git add _images
git commit -m "Update auto_examples + notebooks from local build on $(date)" || echo "🟡 Nothing to commit."
git push

echo "✅ Done. Cleaned up temp directory: $TMP_DIR"
rm -rf "$TMP_DIR"
