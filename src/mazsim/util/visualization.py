from pathlib import Path
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, save
from bokeh.resources import INLINE


def save_scatter_html(df, geography, variable, total_col, obs_col, year, output_dir, base_year):
    """Interactive observed vs. simulated change scatterplot saved as a standalone html file."""
    plot_df = df[[total_col, obs_col]].dropna()
    source = ColumnDataSource(
        data={
            'geography_id': plot_df.index.astype(str),
            'observed': plot_df[obs_col],
            'simulated': plot_df[total_col],
            'difference': plot_df[total_col] - plot_df[obs_col],
        }
    )
    r_squared = plot_df[obs_col].corr(plot_df[total_col]) ** 2
    # Change can be negative, so the axes are squared off around both tails and zero.
    span = float(max(plot_df[[obs_col, total_col]].abs().max().max(), 1)) * 1.05
    axis_min, axis_max = -span, span

    p = figure(
        title=(
            f'change in {variable} by {geography}, {base_year}\u2013{year} '
            f'(n={len(plot_df)}, r\u00b2={r_squared:.3f})'
        ),
        x_axis_label=f'observed change ({obs_col})',
        y_axis_label=f'simulated change ({total_col})',
        width=800,
        height=800,
        x_range=(axis_min, axis_max),
        y_range=(axis_min, axis_max),
        tools='pan,box_zoom,wheel_zoom,reset,save',
    )
    p.line([axis_min, axis_max], [axis_min, axis_max], color='red', line_dash='dashed', legend_label='1:1 line')
    points = p.scatter('observed', 'simulated', source=source, size=7, alpha=0.5)
    p.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=[
                (geography, '@geography_id'),
                ('observed change', '@observed{0,0}'),
                ('simulated change', '@simulated{0,0}'),
                ('difference', '@difference{0,0}'),
            ],
        )
    )
    p.legend.location = 'top_left'

    html_path = Path.joinpath(output_dir, f'{geography}_{variable}_change_{year}.html')
    save(p, filename=str(html_path), resources=INLINE, title=f'{variable} change validation {base_year}\u2013{year}')
    print(f'Saved validation scatterplot to {html_path}')