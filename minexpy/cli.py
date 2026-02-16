"""Command-line interface for MinexPy."""

import sys
import click
from . import __version__


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', is_flag=True, help='Show version information')
def cli(ctx, version):
    """
    MinexPy - A python toolkit for geoscience researchers.
    
    MinexPy provides analysis tools specifically designed for
    geochemical, geophysical, and  geological data analysis.
    
    Use 'minexpy COMMAND --help' for more information on a specific command.
    """
    if version:
        click.echo(f'MinexPy version {__version__}')
        return
    
    if ctx.invoked_subcommand is None:
        # Show enhanced help when no command is given
        show_enhanced_help()


def show_enhanced_help():
    """Display enhanced help with available modules and functions."""
    from . import stats
    
    click.echo(f"""
┌─────────────────────────────────────────────────────────────────┐
│                MinexPy v{__version__}                           │ 
│        A Practical Toolkit for Geoscience Researchers           │
└─────────────────────────────────────────────────────────────────┘

USAGE
  minexpy [COMMAND] [OPTIONS]

COMMANDS
  docs        Open documentation in web browser
  info        Show detailed package information
  demo        Show practical code examples you can copy and use
  help        Show this help message

OPTIONS
  --version   Show version information
  --help      Show this help message

AVAILABLE MODULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  minexpy.stats       Statistical analysis module for geoscience data
  minexpy.correlation Correlation analysis utilities
  minexpy.statviz     Statistical visualization utilities
  minexpy.mapping     Mapping data

MAIN CLASSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  StatisticalAnalyzer    Comprehensive statistical analyzer class

STATISTICAL FUNCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Basic Statistics:
    mean                 Calculate arithmetic mean
    median               Calculate median value
    std                  Calculate standard deviation
    variance             Calculate variance
    
  Distribution Metrics:
    skewness             Measure distribution asymmetry
    kurtosis             Measure distribution tailedness
    coefficient_of_variation  Normalized dispersion measure
    
  Robust Statistics:
    iqr                  Interquartile range
    percentile           Calculate specific percentile
    
  Outlier Detection:
    z_score              Calculate standardized scores
    
  Summary:
    describe             Comprehensive statistical summary

CORRELATION FUNCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pearson_correlation     Linear correlation with p-value
  spearman_correlation    Rank-based monotonic correlation
  kendall_correlation     Robust rank concordance correlation
  distance_correlation    Nonlinear dependence correlation
  biweight_midcorrelation Robust outlier-resistant correlation
  partial_correlation     Correlation with control variables
  correlation_matrix      Pairwise correlation matrix utility

VISUALIZATION FUNCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  plot_histogram          Histogram (linear/log)
  plot_box_violin         Box or violin plot
  plot_ecdf               Empirical CDF
  plot_qq                 Q-Q diagnostic plot
  plot_pp                 P-P diagnostic plot
  plot_scatter            Scatter plot (+ optional trend line)

QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  import numpy as np
  import minexpy.stats as mstats
  from minexpy import StatisticalAnalyzer, describe, pearson_correlation
  
  # Analyze your data
  data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
  summary = describe(data)

EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Run 'minexpy demo' to see practical examples you can copy and use!

DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📚 Full documentation: https://SupernovifieD.github.io/MinexPy/
  💻 GitHub repository: https://github.com/SupernovifieD/MinexPy
  🐛 Report issues: https://github.com/SupernovifieD/MinexPy/issues

For detailed function documentation, see the online documentation or use:
  python -c "import minexpy.stats; help(minexpy.stats.FUNCTION_NAME)"
    """)


@cli.command()
def docs():
    """Open documentation in web browser."""
    import webbrowser
    url = 'https://SupernovifieD.github.io/MinexPy/'
    webbrowser.open(url)
    click.echo(f'📚 Opening documentation in your browser...')
    click.echo(f'   {url}')


@cli.command()
def info():
    """Show detailed package information."""
    click.echo(f"""
┌─────────────────────────────────────────────────────────────────┐
│                MinexPy v{__version__}                           │
│        A Practical Toolkit for Geoscience Researchers           │
└─────────────────────────────────────────────────────────────────┘

ABOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MinexPy is a python toolkit for geoscience researchers and students
who are learning how to solve real geoscientific problems with Python.

It provides curated, beginner-friendly building blocks and sensible
combinations of widely-used libraries, so you can focus more on the
science and less on setup and guesswork.

PACKAGE DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📦 Package:      minexpy
  🔖 Version:      {__version__}
  🐍 Python:       >=3.8
  📝 License:      MIT
  👤 Author:       Yasin Ghasemi

DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • numpy        (>=1.24.0)  - Numerical computing
  • pandas       (>=2.0.0)   - Data manipulation
  • scipy        (>=1.10.0)  - Scientific computing
  • matplotlib   (>=3.7.0)   - Statistical visualization
  • openpyxl     (>=3.1.0)   - Excel file support

LINKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📚 Documentation: https://SupernovifieD.github.io/MinexPy/
  💻 Repository:    https://github.com/SupernovifieD/MinexPy
  🐛 Issues:        https://github.com/SupernovifieD/MinexPy/issues
  📦 PyPI:          https://pypi.org/project/minexpy/

INSTALLATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install minexpy

QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  import numpy as np
  from minexpy import describe
  
  data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
  summary = describe(data)
  print(summary)

Run 'minexpy demo' for more examples!
    """)


@cli.command()
@click.option(
    '--topic',
    type=click.Choice(
        ['basic', 'class', 'dataframe', 'outliers', 'percentiles', 'csv', 'correlation', 'visualization', 'all']
    ),
              default='all', help='Show examples for specific topic')
def demo(topic):
    """
    Practical code examples you can copy and use.
    
    Examples are categorized by topic:
    - basic: Basic statistical functions
    - class: Using StatisticalAnalyzer class
    - dataframe: Multi-column DataFrame analysis
    - outliers: Outlier detection with z-scores
    - percentiles: Percentile and IQR analysis
    - csv: CSV workflow example
    - correlation: Correlation analysis workflows
    - visualization: Statistical plotting workflows
    - all: Show all examples (default)
    """
    examples = {
        'basic': '''
┌─────────────────────────────────────────────────────────────────┐
│              EXAMPLE 1: Basic Statistical Analysis              │
└─────────────────────────────────────────────────────────────────┘

# Import the library
import numpy as np
import minexpy.stats as mstats

# Your geochemical data (e.g., zinc concentrations in ppm)
data = np.array([45.2, 52.3, 38.7, 61.2, 49.8, 55.1, 42.3, 58.9])

# Calculate individual statistics
mean_val = mstats.mean(data)
median_val = mstats.median(data)
std_val = mstats.std(data)
skew_val = mstats.skewness(data)
kurt_val = mstats.kurtosis(data)
cv_val = mstats.coefficient_of_variation(data)

print(f"Mean: {mean_val:.2f} ppm")
print(f"Median: {median_val:.2f} ppm")
print(f"Std Dev: {std_val:.2f}")
print(f"Skewness: {skew_val:.3f}")
print(f"Kurtosis: {kurt_val:.3f}")
print(f"CV: {cv_val:.3f} ({cv_val*100:.1f}%)")

# Or get everything at once with describe()
from minexpy import describe
summary = describe(data)
print("\\nComplete Summary:")
for key, value in summary.items():
    print(f"{key}: {value:.3f}")
''',
        'class': '''
┌─────────────────────────────────────────────────────────────────┐
│         EXAMPLE 2: Using StatisticalAnalyzer Class              │
└─────────────────────────────────────────────────────────────────┘

# Import the class
import numpy as np
from minexpy.stats import StatisticalAnalyzer

# Your geochemical data
copper_data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 25.0])

# Create analyzer instance
analyzer = StatisticalAnalyzer(copper_data)

# Get comprehensive summary
summary = analyzer.summary()
print("Copper (Cu) Analysis:")
print(summary)

# Or access individual metrics
print(f"\\nMean: {analyzer.mean():.2f}")
print(f"Median: {analyzer.median():.2f}")
print(f"Skewness: {analyzer.skewness():.3f}")
print(f"Kurtosis: {analyzer.kurtosis():.3f}")
print(f"IQR: {analyzer.iqr():.2f}")
print(f"CV: {analyzer.coefficient_of_variation():.3f}")
''',
        'dataframe': '''
┌─────────────────────────────────────────────────────────────────┐
│        EXAMPLE 3: Multi-Element DataFrame Analysis              │
└─────────────────────────────────────────────────────────────────┘

# Import libraries
import pandas as pd
from minexpy.stats import StatisticalAnalyzer

# Create sample geochemical dataset
data = {
    'Zn': [45.2, 52.3, 38.7, 61.2, 49.8, 55.1, 42.3, 58.9],
    'Cu': [12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 25.0, 14.2],
    'Pb': [8.3, 10.1, 9.5, 11.2, 9.8, 10.5, 8.9, 9.7],
    'Ag': [0.5, 0.7, 0.6, 0.9, 0.8, 0.7, 1.2, 0.6]
}
df = pd.DataFrame(data)

# Analyze all elements at once
analyzer = StatisticalAnalyzer(df)
summary_df = analyzer.summary()

print("Multi-Element Statistical Summary:")
print(summary_df)

# Compare variability across elements
print("\\nCoefficient of Variation (CV) Comparison:")
cv_values = analyzer.coefficient_of_variation()
for element, cv in cv_values.items():
    print(f"{element}: {cv:.3f} ({cv*100:.1f}%)")

# Find most variable element
most_variable = cv_values.idxmax()
print(f"\\nMost variable element: {most_variable}")

# Analyze specific columns only
selected_elements = df[['Zn', 'Cu']]
analyzer_selected = StatisticalAnalyzer(selected_elements)
print("\\nSelected Elements Analysis:")
print(analyzer_selected.summary())
''',
        'outliers': '''
┌─────────────────────────────────────────────────────────────────┐
│           EXAMPLE 4: Outlier Detection with Z-Scores            │
└─────────────────────────────────────────────────────────────────┘

# Import libraries
import numpy as np
from minexpy.stats import z_score, StatisticalAnalyzer

# Geochemical data with potential outliers
data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 45.2, 14.2, 
                 17.9, 21.3, 18.1, 19.7, 16.2])

# Calculate z-scores for all data points
z_scores = z_score(data)

# Identify outliers (|z| > 2 is common threshold)
outlier_mask = np.abs(z_scores) > 2
outliers = data[outlier_mask]
outlier_indices = np.where(outlier_mask)[0]

print("Outlier Detection Results:")
print(f"Total samples: {len(data)}")
print(f"Outliers found: {len(outliers)}")
print(f"Outlier values: {outliers}")
print(f"At indices: {outlier_indices}")
print(f"Z-scores: {z_scores[outlier_mask]}")

# More stringent threshold (|z| > 3)
severe_outliers = data[np.abs(z_scores) > 3]
print(f"\\nSevere outliers (|z| > 3): {len(severe_outliers)}")

# Check specific value
test_value = 50.0
z_test = z_score(data, value=test_value)
print(f"\\nZ-score for {test_value}: {z_test:.3f}")
if abs(z_test) > 2:
    print("  → This would be an outlier!")

# Statistical summary excluding outliers
clean_data = data[~outlier_mask]
analyzer = StatisticalAnalyzer(clean_data)
print("\\nStatistics (outliers removed):")
summary = analyzer.summary()
print(f"Mean: {summary['mean']:.2f}")
print(f"Std: {summary['std']:.2f}")
print(f"Skewness: {summary['skewness']:.3f}")
''',
        'percentiles': '''
┌─────────────────────────────────────────────────────────────────┐
│        EXAMPLE 5: Percentile Analysis & Distribution            │
└─────────────────────────────────────────────────────────────────┘

# Import libraries
import numpy as np
from minexpy.stats import percentile, iqr, describe

# Geochemical dataset
gold_data = np.array([0.5, 0.7, 0.6, 0.9, 0.8, 0.7, 1.2, 0.6, 
                      0.9, 1.1, 0.8, 1.5, 0.7, 0.9, 1.0])

# Calculate specific percentiles
p25 = percentile(gold_data, 25)
p50 = percentile(gold_data, 50)  # median
p75 = percentile(gold_data, 75)
p90 = percentile(gold_data, 90)
p95 = percentile(gold_data, 95)

print("Gold (Au) Concentration Percentiles (ppm):")
print(f"25th percentile (Q1): {p25:.2f}")
print(f"50th percentile (Median): {p50:.2f}")
print(f"75th percentile (Q3): {p75:.2f}")
print(f"90th percentile: {p90:.2f}")
print(f"95th percentile: {p95:.2f}")

# Interquartile range
iqr_val = iqr(gold_data)
print(f"\\nInterquartile Range (IQR): {iqr_val:.2f}")

# Use describe with custom percentiles
summary = describe(gold_data, percentiles=[10, 25, 50, 75, 90, 95, 99])
print("\\nComplete Distribution Summary:")
print(f"Min: {summary['min']:.2f}")
print(f"10th: {summary['percentile_10']:.2f}")
print(f"25th (Q1): {summary['percentile_25']:.2f}")
print(f"50th (Median): {summary['percentile_50']:.2f}")
print(f"75th (Q3): {summary['percentile_75']:.2f}")
print(f"90th: {summary['percentile_90']:.2f}")
print(f"95th: {summary['percentile_95']:.2f}")
print(f"99th: {summary['percentile_99']:.2f}")
print(f"Max: {summary['max']:.2f}")

# Identify high-grade samples (>90th percentile)
high_grade = gold_data[gold_data > p90]
print(f"\\nHigh-grade samples (>P90): {len(high_grade)}")
print(f"Values: {high_grade}")
''',
        'csv': '''
┌─────────────────────────────────────────────────────────────────┐
│          EXAMPLE 6: Working with Real CSV Data                  │
└─────────────────────────────────────────────────────────────────┘

# Import libraries
import pandas as pd
from minexpy.stats import StatisticalAnalyzer

# Load your geochemical data from CSV
# df = pd.read_csv('your_geochemical_data.csv')

# For this example, let's create sample data
data = {
    'Sample_ID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006'],
    'Zn_ppm': [45.2, 52.3, 38.7, 61.2, 49.8, 55.1],
    'Cu_ppm': [12.5, 15.3, 18.7, 22.1, 19.4, 16.8],
    'Pb_ppm': [8.3, 10.1, 9.5, 11.2, 9.8, 10.5],
    'Au_ppb': [5.2, 7.3, 6.1, 9.2, 8.1, 7.5],
    'Location': ['North', 'North', 'South', 'South', 'East', 'West']
}
df = pd.DataFrame(data)

# Select numeric columns for analysis
numeric_cols = ['Zn_ppm', 'Cu_ppm', 'Pb_ppm', 'Au_ppb']

# Analyze all elements
analyzer = StatisticalAnalyzer(df[numeric_cols])
summary = analyzer.summary()

print("Geochemical Data Summary:")
print(summary)
print()

# Analyze by location (group by)
print("\\nAnalysis by Location:")
for location in df['Location'].unique():
    location_data = df[df['Location'] == location][numeric_cols]
    if len(location_data) > 1:
        print(f"\\n{location}:")
        loc_analyzer = StatisticalAnalyzer(location_data)
        loc_summary = loc_analyzer.summary()
        print(f"  Zn mean: {loc_summary.loc['Zn_ppm', 'mean']:.2f}")
        print(f"  Cu mean: {loc_summary.loc['Cu_ppm', 'mean']:.2f}")

# Export summary to CSV
summary.to_csv('statistical_summary.csv')
print("\\nSummary exported to 'statistical_summary.csv'")

# Individual element analysis
print("\\n" + "="*60)
print("Zinc (Zn) Detailed Analysis:")
print("="*60)
zn_analyzer = StatisticalAnalyzer(df['Zn_ppm'])
zn_summary = zn_analyzer.summary()
for stat, value in zn_summary.items():
    print(f"{stat:25s}: {value:.3f}")
''',
        'correlation': '''
┌─────────────────────────────────────────────────────────────────┐
│          EXAMPLE 7: Correlation Analysis Workflows              │
└─────────────────────────────────────────────────────────────────┘

# Import libraries
import numpy as np
import pandas as pd
from minexpy.correlation import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
    distance_correlation,
    biweight_midcorrelation,
    partial_correlation,
    correlation_matrix,
)

rng = np.random.default_rng(42)
depth = np.linspace(20, 300, 80)
zn = 0.15 * depth + rng.normal(0, 4, size=80)
cu = 0.10 * depth + 0.45 * zn + rng.normal(0, 3, size=80)
pb = 0.05 * depth + rng.normal(0, 2, size=80)

print("Pairwise correlation between Zn and Cu:")
print("  Pearson :", pearson_correlation(zn, cu))
print("  Spearman:", spearman_correlation(zn, cu))
print("  Kendall :", kendall_correlation(zn, cu))
print("  Distance:", distance_correlation(zn, cu))
print("  Biweight:", biweight_midcorrelation(zn, cu))
print("  Partial (control depth):", partial_correlation(zn, cu, controls=depth))

df = pd.DataFrame({"Zn": zn, "Cu": cu, "Pb": pb})
print("\\nPearson correlation matrix:")
print(correlation_matrix(df, method="pearson"))

print("\\nDistance correlation matrix:")
print(correlation_matrix(df, method="distance"))
''',
        'visualization': '''
┌─────────────────────────────────────────────────────────────────┐
│      EXAMPLE 8: Statistical Visualization Diagnostics           │
└─────────────────────────────────────────────────────────────────┘

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from minexpy.statviz import (
    plot_histogram,
    plot_box_violin,
    plot_ecdf,
    plot_qq,
    plot_pp,
    plot_scatter,
)

rng = np.random.default_rng(42)
zn = rng.lognormal(mean=2.2, sigma=0.35, size=250)
cu = 0.35 * zn + rng.normal(0, 1.5, size=250)

# Histogram (linear and log)
plot_histogram(zn, bins=30, scale="linear", xlabel="Zn (ppm)", title="Histogram (Linear)")
plot_histogram(zn, bins=30, scale="log", xlabel="Zn (ppm, log scale)", title="Histogram (Log)")

# Box / violin
plot_box_violin({"Zn": zn, "Cu": cu}, kind="box", ylabel="Concentration (ppm)", title="Box Plot")
plot_box_violin({"Zn": zn, "Cu": cu}, kind="violin", ylabel="Concentration (ppm)", title="Violin Plot")

# ECDF, Q-Q, P-P, scatter
plot_ecdf({"Zn": zn, "Cu": cu}, xlabel="Concentration (ppm)", title="ECDF")
plot_qq(zn, distribution="norm", ylabel="Zn Sample Quantiles", title="Q-Q Plot")
plot_pp(zn, distribution="norm", title="P-P Plot")
plot_scatter(zn, cu, add_trendline=True, xlabel="Zn (ppm)", ylabel="Cu (ppm)", title="Scatter Plot")

plt.show()
'''
    }
    
    if topic == 'all':
        click.echo("\n" + "="*65)
        click.echo("         MinexPy - Practical Code Examples")
        click.echo("="*65)
        click.echo("\nCopy and paste these examples into your Python scripts!\n")
        
        for example_name, example_code in examples.items():
            click.echo(example_code)
            click.echo()
    else:
        if topic in examples:
            click.echo(examples[topic])
        else:
            click.echo(f"Topic '{topic}' not found.")
            return
    
    click.echo("="*65)
    click.echo("📚 For more examples and detailed documentation, visit:")
    click.echo("   https://SupernovifieD.github.io/MinexPy/")
    click.echo()
    click.echo("💡 Tips:")
    click.echo("   - Use 'minexpy demo --topic=basic' to see specific examples")
    click.echo("   - Use 'minexpy demo --topic=correlation' for correlation workflows")
    click.echo("   - Use 'minexpy demo --topic=visualization' for plotting workflows")
    click.echo("   - Replace sample data with your own geochemical datasets")
    click.echo("   - Check help for any function: help(minexpy.stats.FUNCTION)")
    click.echo("="*65)


@cli.command()
def help():
    """Show detailed help information."""
    show_enhanced_help()


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
