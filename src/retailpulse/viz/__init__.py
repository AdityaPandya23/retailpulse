"""Plotly figure builders for the app. Every function takes DataFrames / dicts and returns a
go.Figure; pages only lay them out. Importing this package registers the shared template."""

from retailpulse.viz.palette import register_template

TEMPLATE = register_template()
