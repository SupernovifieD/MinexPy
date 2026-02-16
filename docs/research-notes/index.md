# Research Notes

The research notes capture the scientific background behind each module. They summarize the statistical and spatial assumptions, formulas, and limitations that inform how the library is implemented.

## Available notes

| Topic | What it covers | Related APIs |
| --- | --- | --- |
| [Statistics](statistical-foundations.md) | Sample moments, quantiles, shape metrics, correlation methods, visualization diagnostics, and how MinexPy implements them | `minexpy.stats`, `minexpy.correlation`, `minexpy.statviz`, `StatisticalAnalyzer` |
| [Mapping](mapping-foundations.md) | End-to-end 4-step mapping workflow: preparation, gridding, interpolation mathematics, and map composition principles | `minexpy.mapping.prepare`, `minexpy.mapping.create_grid`, `minexpy.mapping.interpolate*`, `minexpy.mapping.plot_map` |
