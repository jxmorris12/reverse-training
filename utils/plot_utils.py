from typing import Optional, Union

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


font_name = "Helvetica Neue LT Pro"

# colors = [
#     (0.5, 0.0, 0.13),  # Maroon
#     (0.6, 0.1, 0.2),   # Dark red
#     (0.7, 0.2, 0.3),   # Deep rose
#     (0.7, 0.8, 0.9),   # Pastel blue
#     (0.6, 0.7, 0.85),  # Light pastel blue
#     (0.5, 0.6, 0.8),   # Medium pastel blue
#     (0.4, 0.5, 0.7)    # Soft indigo
# ]
# def _get_capacity_cmap() -> matplotlib.colors.LinearSegmentedColormap:
#     return matplotlib.colors.LinearSegmentedColormap.from_list("PinkBlue", colors)

# def _get_capacity_palette(n_colors: int) -> sns.color_palette:
#     pink_blue_cmap = _get_capacity_cmap()
#     if n_colors == 1:
#         return [pink_blue_cmap(0.5)]
#     else:
#         palette = [pink_blue_cmap(i / (n_colors - 1)) for i in range(n_colors)]
#         return sns.color_palette(palette)


grid_color = "#A8A29E"
def _init_plot():
    # Set font and style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.figure(figsize=(10, 8))
    # sns.set_theme(style="white", font_scale=1.9, font=font_name, context="notebook")
    sns.set_theme(style="whitegrid", 
        font_scale=1.8, font=font_name, 
        context="notebook", 
        rc={
            "axes.facecolor": (250/255, 243/255, 235/255, 1.0),
            "grid.color": grid_color, 
            "grid.linewidth": 1.5
        },
    )

def _format_label(label: Union[str, list, tuple]) -> str:
    label_str_0 = label[0].replace(" ", "\\ ")
    label_str_0 = r'$\bf{' + label_str_0 + '}$'
    label_str_1 = label[1]
    label_str_1 = f'\n({label_str_1})'
    return label_str_0 + label_str_1

def plot_df(
        df: pd.DataFrame,
        x: str,
        save_path: str,
        logx: bool = True,
        logy: bool = True,
        y: Optional[str] = None,
        yticks: Optional[list] = None,
        yticklabels: Optional[list] = None,
        ylabel: Optional[str] = None,
        xlabel: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        title: Optional[str] = None,
        legend_title: str = "",
        legend_loc = "upper right",
        legend_hide: bool = False,
        kind: str = "line",
        custom_legend_labels: Optional[dict] = None,
        augment_ax: Optional[callable] = None,
        **plot_kwargs
    ):
    _init_plot()
    # Plot using Seaborn
    if kind == "line":
        plot_func = sns.lineplot
        if "lw" not in plot_kwargs:
            plot_kwargs["lw"] = 3.2
    elif kind == "scatter":
        plot_func = sns.scatterplot
    elif kind == "hist":
        plot_func = sns.histplot
    else:
        raise ValueError(f"Unknown plot kind: {kind}")
    
    plot_kwargs["rasterized"] = True
    # Replace markers
    if "marker" in plot_kwargs:
        plot_kwargs["marker"] = "h"

    ax = plot_func(data=df, x=x, y=y, **plot_kwargs)

    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if yticks is not None: ax.set_yticks([0.001, 0.01, 0.1, 1])
    if yticklabels is not None: ax.set_yticklabels(["0.1%", "1%", "10%", "100%"])
    if ylabel is not None: 
        if isinstance(ylabel, tuple) or isinstance(ylabel, list):
            ax.set_ylabel(_format_label(ylabel), fontweight="light", labelpad=4)
        else:
            ax.set_ylabel(ylabel, fontweight="bold", labelpad=4)
    if xlabel is not None:
        if isinstance(xlabel, tuple) or isinstance(xlabel, list):
            ax.set_xlabel(_format_label(xlabel), fontweight="light", labelpad=4)
        else:
            ax.set_xlabel(xlabel, fontweight="bold", labelpad=4)
        
    if title is not None: 
        title_size = ax.xaxis.label.get_fontsize()
        ax.set_title(title, fontweight="bold", fontsize=title_size, pad=8)
    
    # Customize xlim and ylim
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # Run custom functionality
    if augment_ax is not None:
        augment_ax(ax)
    
    # Some more customization
    # Hide top and right spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Customize ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.tick_params(axis='both', which='major', labelweight='light')
    # tick_font_color = (75/255, 58/255, 40/255)
    tick_font_color = "#444"
    for tick in ax.get_xticklabels():
        tick.set_fontweight(300)
        tick.set_fontsize(tick.get_fontsize() * 0.85)
        tick.set_color(tick_font_color)
    for tick in ax.get_yticklabels():
        tick.set_fontweight(300)
        tick.set_fontsize(tick.get_fontsize() * 0.85)
        tick.set_color(tick_font_color)

    # Customize legend
    legend = ax.get_legend()
    if legend is not None:
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.texts]
        if custom_legend_labels is not None:
            new_labels = [custom_legend_labels.get(label, str(label)) for label in labels]
            legend = ax.legend(handles, new_labels, handletextpad=0.5, labelspacing=0.2, loc=legend_loc, framealpha=1, facecolor="white")
        else:
            legend = ax.legend(handles, labels, handletextpad=0.5, labelspacing=0.2, loc=legend_loc, framealpha=1, facecolor="white")
        legend.set_title(legend_title)
        legend_font_size = ax.xaxis.label.get_fontsize() * 0.75
        legend.get_title().set_fontsize(legend_font_size)
        legend.get_title().set_weight('bold')
        for text in legend.get_texts():
            text.set_fontsize(legend_font_size)
        legend.get_frame().set_linewidth(2)
        legend.get_frame().set_edgecolor('gray')

        if legend_hide:
            legend.remove()

    # Customize plot borders
    border_width = 3.0
    border_color = 'black'
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['top'].set_edgecolor(border_color)
    ax.spines['right'].set_linewidth(border_width)
    ax.spines['right'].set_edgecolor(border_color)
    ax.spines['left'].set_linewidth(border_width)
    ax.spines['left'].set_edgecolor(border_color)
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['bottom'].set_edgecolor(border_color)

    plt.tight_layout()
    plt.savefig(
        save_path, 
        format='pdf', 
        bbox_inches='tight', 
        dpi=300,
    )
    print(f"Saved plot to {save_path}.")
    plt.show()
    return ax