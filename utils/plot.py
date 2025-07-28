import matplotlib.pyplot as plt
import io
import base64

def create_scatterplot(df):
    fig, ax = plt.subplots()
    ax.scatter(df["Rank"], df["Peak"], label="Data")
    m, b = np.polyfit(df["Rank"], df["Peak"], 1)
    ax.plot(df["Rank"], m * df["Rank"] + b, linestyle='dotted', color='red', label="Regression")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
