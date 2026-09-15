from pathlib import Path
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, save
from bokeh.resources import INLINE


def save_scatter_html(df, geography, variable, total_col, obs_col, year, output_dir):
    """Interactive observed vs. simulated scatterplot saved as a standalone html file."""
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
    axis_max = float(max(plot_df[obs_col].max(), plot_df[total_col].max(), 1)) * 1.05

    p = figure(
        title=f'{variable} by {geography}, {year} (n={len(plot_df)}, r\u00b2={r_squared:.3f})',
        x_axis_label=f'observed ({obs_col})',
        y_axis_label=f'simulated ({total_col})',
        width=800,
        height=800,
        x_range=(0, axis_max),
        y_range=(0, axis_max),
        tools='pan,box_zoom,wheel_zoom,reset,save',
    )
    p.line([0, axis_max], [0, axis_max], color='red', line_dash='dashed', legend_label='1:1 line')
    points = p.scatter('observed', 'simulated', source=source, size=7, alpha=0.5)
    p.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=[
                (geography, '@geography_id'),
                ('observed', '@observed{0,0}'),
                ('simulated', '@simulated{0,0}'),
                ('difference', '@difference{0,0}'),
            ],
        )
    )
    p.legend.location = 'top_left'

    html_path = Path.joinpath(output_dir, f'{geography}_{variable}_{year}.html')
    save(p, filename=str(html_path), resources=INLINE, title=f'{variable} validation {year}')
    print(f'Saved validation scatterplot to {html_path}')