# %%

import pandas as pd

from opentak import TakBuilder, TakVisualizer
from opentak.generation_cohort_tak import GenerateCohortTAK
from opentak.tak_theme import set_style
from opentak.visualization import add_grid_on_tak_fig

set_style()

# %%
RANDOM_STATE = 42

NB_PATIENTS = 216
NB_JOURS_END = 350

# Event log generation
base = GenerateCohortTAK(
    nb_patients=NB_PATIENTS, nb_days_end=NB_JOURS_END, random_state=RANDOM_STATE
)
base.initialisation_dataframe(
    treatment_name="A",
    dose_mean=int(NB_JOURS_END / 10),
    dose_std=int(NB_JOURS_END / 25),
)
base.add_switch_gaussien("B")
base.add_switch_gaussien("C")

# Formatting
base = base.add_in_out()
base = base.sort_values(by=["ID_PATIENT", "TIMESTAMP"])

# %%
# Event log generation 2
NB_JOURS_END_LONG = int(365.25 * 5)

base_long = GenerateCohortTAK(
    nb_patients=NB_PATIENTS, nb_days_end=NB_JOURS_END_LONG, random_state=RANDOM_STATE
)
base_long.initialisation_dataframe(
    treatment_name="A",
    dose_mean=int(NB_JOURS_END_LONG / 10),
    dose_std=int(NB_JOURS_END_LONG / 25),
)
base_long.add_switch_gaussien("B")
base_long.add_switch_gaussien("C")

# Formatting
base_long = base_long.add_in_out()
base_long = base_long.sort_values(by=["ID_PATIENT", "TIMESTAMP"])

# %%
# Objects used to build visualization
tak = TakBuilder(base).build()
tak.fit()

# %%
# Visualization

# Initializing visualizer
tak_viz = TakVisualizer(tak)

# Processing of the array
tak_viz.process_visualization()

# Generating Plotly figure
figplotly = tak_viz.get_plot()

figplotly.write_image("docs/assets/tak_hca_example.svg")

# %%
# Generating Plotly figure
fig_events_on_tak_percent = tak_viz.graph_events_rep_on_tak(
    events_not_shown=["start", "in", "out", "end"],
    events_not_in_percent=["start", "in", "out", "end"],
    threshold_percent=0,
)

fig_events_on_tak_percent.write_image("docs/assets/tak_repartition.svg")

# %%
# Usage avancé

# Personnaliser couleurs

tak_viz.update_colors(dict_new_colors={"A": "rgb(255, 114, 64)"}, B="#5E5A85")
tak_viz.process_visualization()
tak_viz.get_plot().write_image("docs/assets/tak_update_colors.svg")

# Personnaliser les légendes

tak_viz_legend = TakVisualizer(tak, dico_evt_for_legend={"A": "Nouveau nom pour A"})
tak_viz_legend.process_visualization()
tak_viz_legend.get_plot().write_image("docs/assets/tak_update_legend_names.svg")

# Axe calendaire

# 2 months
tak_viz_legend.process_visualization()
tak_viz_legend.get_plot(unit_as_months=True, nb_months=2).write_image(
    "docs/assets/tak_xaxes_2months.svg"
)

# years
tak = TakBuilder(base_long).build()
tak.fit()
tak_viz = TakVisualizer(tak)
tak_viz.process_visualization()
tak_viz.get_plot(unit_as_years=True).write_image("docs/assets/tak_xaxes_years.svg")

# calendaire years
tak_viz.process_visualization(base_date=pd.to_datetime("2014-01-01"))
tak_viz.get_plot(unit_as_years=False).write_image(
    "docs/assets/tak_xaxes_years_2014.svg"
)

tak_viz.graph_events_rep_on_tak(
    events_not_shown=["start", "in", "out", "end"],
    events_not_in_percent=["start", "in", "out", "end"],
    threshold_percent=0,
).show()

# %%

# Add grid to the tak
tak_viz_grid = TakVisualizer(tak)
tak_viz_grid.process_visualization()
fig = tak_viz_grid.get_plot()
fig = add_grid_on_tak_fig(
    fig, grid="xy", params={"x": {"opacity": 1}, "y": {"opacity": 1}}
)
fig.write_image("docs/assets/tak_grid.svg")

# %%
# Trouver les clusters

n_clusters = 3

tak = TakBuilder(base).build()
tak.fit(n_clusters=n_clusters)

tak_viz = TakVisualizer(tak)
tak_viz.process_visualization()
tak_viz.get_plot(add_sep=True).write_image("docs/assets/tak_clusters.svg")

# %%
# Trouver les clusters : changer taille et couleur annotations

figplotly_with_changed_annotation = tak_viz.get_plot(
    add_sep=True, size_annotation=20, color_annotation="#0000FF"
)
figplotly_with_changed_annotation.write_image("docs/assets/tak_clusters_annotation.svg")

# %%
# Trouver les clusters : changer longueur et épaisseurs des barres horizontales

figplotly_with_changed_line_extension_and_width = tak_viz.get_plot(
    add_sep=True, coef_line_extension=0.3, width_line=10
)
figplotly_with_changed_line_extension_and_width.write_image(
    "docs/assets/tak_clusters_lines.svg"
)

# %%
# n clusters

n_clusters = 3
list_clusters = list(range(n_clusters))

# Objects used to build visualization
tak = TakBuilder(base).build()
tak.fit(n_clusters=n_clusters)

# Initializing visualizer
tak_viz = TakVisualizer(tak)

for num_cluster in list_clusters:
    # Processing of the array
    tak_viz.process_visualization(num_cluster=num_cluster)

    n_pat_cluster = len(tak.list_ids_clusters[num_cluster])
    # Generating Plotly figure
    fig = tak_viz.get_plot().update_layout(
        title_text=f"TAK cluster {num_cluster}, {n_pat_cluster} patients"
    )
    fig.write_image(f"docs/assets/tak_n_clusters_{num_cluster}.svg")

# %%
#############################################
# TAK split doc: How to only visualize specific IDs

tak = TakBuilder(base).build()
tak.fit()

tak_viz = TakVisualizer(tak)
tak_viz.process_visualization()
tak_viz.get_plot().write_image("docs/assets/tak_split_init.svg")

# Split : Only select patients between ID 0 and ID 100
tak_viz.split(list(range(100)))
tak_viz.process_visualization()
tak_viz.get_plot().write_image("docs/assets/tak_split_sub.svg")

# %%
#############################################
# Example with 3 clusters

tak = TakBuilder(base).build()
tak.fit(n_clusters=3)

tak_viz = TakVisualizer(tak)

tak_viz.process_visualization()
tak_viz.get_plot(add_sep=True).write_image("docs/assets/tak_split_clusters_init.svg")

tak_viz.split(list(range(100)))

tak_viz.process_visualization()
tak_viz.get_plot(add_sep=True).write_image("docs/assets/tak_split_clusters_sub.svg")

# %%
##############################################
# Dendrogram

# Visualization without add_sep

tak = TakBuilder(base).build()
tak.fit()

# Initializing visualizer
tak_viz = TakVisualizer(tak)
tak_viz.process_visualization()

# Generating Plotly figure
tak_viz.get_plot(dendrogram=True).write_image("docs/assets/tak_dendrogram.svg")

# Visualization with add_sep

tak = TakBuilder(base).build()
tak.fit(n_clusters=3)

# Initializing visualizer
tak_viz = TakVisualizer(tak)
tak_viz.process_visualization()

# Generating Plotly figure
tak_viz.get_plot(add_sep=True, dendrogram=True).write_image(
    "docs/assets/tak_dendrogram_with_sep.svg"
)

# %%
