from backend.database.db import get_connection


# -------------------------
# SAVE MODEL
# -------------------------
def save_model(version, path, clean_acc, worst_acc, gap):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO models (version, path, clean_acc, worst_acc, gap)
    VALUES (?, ?, ?, ?, ?)
    """, (version, path, clean_acc, worst_acc, gap))

    conn.commit()
    conn.close()


# -------------------------
# SAVE IMAGE
# -------------------------
def save_image(path, prediction, confidence):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO images (path, prediction, confidence)
    VALUES (?, ?, ?)
    """, (path, prediction, confidence))

    conn.commit()
    conn.close()


# -------------------------
# SAVE METRICS
# -------------------------
def save_metrics(model_version, metrics_dict):

    conn = get_connection()
    cursor = conn.cursor()

    for attack, acc in metrics_dict.items():
        cursor.execute("""
        INSERT INTO metrics (model_version, attack, accuracy)
        VALUES (?, ?, ?)
        """, (model_version, attack, acc))

    conn.commit()
    conn.close()
