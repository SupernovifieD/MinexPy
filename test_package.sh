#!/bin/bash
# Automated testing script for MinexPy

set -e

echo "🧪 MinexPy Package Testing Suite"
echo "================================"
echo ""

# Check if package is built
if [ ! -d "dist" ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
    echo "❌ Package not built yet. Run ./build_package.sh first"
    exit 1
fi

# Create test environment
echo "📦 Creating test environment..."
python3 -m venv test_env_temp
source test_env_temp/bin/activate

echo "📥 Installing build tools..."
pip install --quiet --upgrade pip setuptools wheel

echo "📥 Installing minexpy from local build..."
pip install --quiet dist/minexpy-0.1.0-py3-none-any.whl

echo ""
echo "🧪 Running tests..."
echo "===================="
echo ""

# Test 1: Import
echo "Test 1: Package import..."
python3 -c "import minexpy; print('✅ Import successful')"

# Test 2: Version
echo "Test 2: Version check..."
python3 -c "import minexpy; print(f'✅ Version: {minexpy.__version__}')"

# Test 3: Stats module
echo "Test 3: Stats module functionality..."
python3 << 'EOF'
import numpy as np
import minexpy.stats as mstats

data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
mean = mstats.mean(data)
std = mstats.std(data)
skew = mstats.skewness(data)
print(f'✅ Mean: {mean:.2f}, Std: {std:.2f}, Skewness: {skew:.3f}')
EOF

# Test 4: StatisticalAnalyzer class
echo "Test 4: StatisticalAnalyzer class..."
python3 << 'EOF'
import numpy as np
from minexpy.stats import StatisticalAnalyzer

data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
analyzer = StatisticalAnalyzer(data)
summary = analyzer.summary()
print(f'✅ StatisticalAnalyzer working! Found {len(summary)} metrics')
EOF

# Test 5: Describe function
echo "Test 5: Describe function..."
python3 << 'EOF'
import numpy as np
from minexpy import describe

data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
summary = describe(data)
print(f'✅ Describe function working! Generated {len(summary)} statistics')
EOF

# Test 6: Package metadata
echo "Test 6: Package metadata..."
pip show minexpy | grep -E "^(Name|Version|Summary|Author|License):"

echo ""
echo "🎉 All tests passed!"
echo "===================="
echo ""

# Cleanup
deactivate
rm -rf test_env_temp

echo "✨ MinexPy is ready for publication!"
echo ""
echo "Next steps:"
echo "1. Test on Test PyPI: python3 -m twine upload --repository testpypi dist/*"
echo "2. If successful, publish to PyPI: python3 -m twine upload dist/*"
echo ""
