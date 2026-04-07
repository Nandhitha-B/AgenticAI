import sqlite3
import pandas as pd
import plotly.express as px

DB_PATH = "backend/database/app.db"


def load_metrics_dataframe():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT m.version, 
               m.clean_acc, 
               m.worst_acc, 
               m.gap
        FROM models m
        ORDER BY m.id
    """, conn)

    conn.close()
    return df


def plot_gap_trend():
    df = load_metrics_dataframe()

    fig = px.line(
        df,
        x="version",
        y="gap",
        title="Robustness Gap Over Time",
        markers=True
    )

    return fig


def plot_accuracy_trends():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT model_version, metric_name, value
        FROM metrics
    """, conn)

    conn.close()

    fig = px.line(
        df,
        x="model_version",
        y="value",
        color="metric_name",
        title="Accuracy Across Attacks"
    )

    return fig


def plot_worst_case():
    df = load_metrics_dataframe()

    fig = px.bar(
        df,
        x="version",
        y="worst_acc",
        title="Worst-case Accuracy per Model"
    )

    return fig


def get_model_table():
    df = load_metrics_dataframe()
    return df
