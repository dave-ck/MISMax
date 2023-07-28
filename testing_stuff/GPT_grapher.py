# content of this file was (largely) generated using GPT
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import networkx as nx
import numpy as np


def calculate_positions(graph, layout):
    ''' Calculates node positions according to selected layout '''
    pos = None

    if layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'random':
        pos = nx.random_layout(graph)
    elif layout == 'shell':
        pos = nx.shell_layout(graph)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)

    if pos:
        return {node: {'x': p[0] * 1e3, 'y': p[1] * 1e3} for node, p in pos.items()}

    return None


def convert_graph_to_cyto_elements(graph):
    ''' Converts a networkx graph into elements for cytoscape '''
    elements = []

    # Add nodes
    for node in graph.nodes:
        elements.append({
            'data': {'id': str(node), 'label': str(node)},
        })

    # Add edges
    for source, target in graph.edges:
        elements.append({
            'data': {
                'source': str(source),
                'target': str(target)
            }
        })

    return elements


# Create a random graph
G = nx.erdos_renyi_graph(10, 0.5)

# App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='layout-dropdown',
            options=[
                {'label': 'Spring Layout', 'value': 'spring'},
                {'label': 'Circular Layout', 'value': 'circular'},
                {'label': 'Random Layout', 'value': 'random'},
                {'label': 'Shell Layout', 'value': 'shell'},
                {'label': 'Spectral Layout', 'value': 'spectral'},
                {'label': 'Kamada Kawai Layout', 'value': 'kamada_kawai'}
            ],
            value='spring',
            clearable=False,
            style={'width': '200px'}
        )
    ], style={'padding': '10px'}),
    cyto.Cytoscape(
        id='cytoscape-graph',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '800px'},
        elements=convert_graph_to_cyto_elements(G),
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)'
                }
            }
        ],
        userZoomingEnabled=True,
        userPanningEnabled=True,
        boxSelectionEnabled=True,
        autoungrabify=False,
    )
])


@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('layout-dropdown', 'value')
)
def update_node_positions(layout):
    new_elements = convert_graph_to_cyto_elements(G)
    positions = calculate_positions(G, layout)
    for elem in new_elements:
        if 'source' not in elem['data']:
            elem['position'] = positions[int(elem['data']['id'])]  # convert id to int
    return new_elements


if __name__ == '__main__':
    app.run_server(debug=True)
