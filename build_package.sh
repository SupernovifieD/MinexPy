#!/bin/bash
# Quick build and test script for MinexPy

set -e  # Exit on error

echo "🔨 Building MinexPy package..."
python3 -m build

echo ""
echo "✅ Build complete! Package files created in dist/"
echo ""
echo "📦 Package contents:"
ls -lh dist/

echo ""
echo "================================================"
echo "✨ Package is ready for testing!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Create test environment: python3 -m venv test_env"
echo "2. Activate it: source test_env/bin/activate"
echo "3. Install package: pip install dist/minexpy-0.1.0-py3-none-any.whl"
echo "4. Run tests from TESTING_GUIDE.md"
echo ""
echo "Or run all tests automatically with:"
echo "  ./test_package.sh"
echo ""
