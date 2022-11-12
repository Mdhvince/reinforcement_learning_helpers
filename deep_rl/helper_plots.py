import matplotlib.pyplot as plt


def basis_plotting_style(title, xlabel, ylabel, rotation_xlabel=0, rotation_ylabel=0):
    plt.style.use('seaborn-paper')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation_xlabel)
    plt.yticks(rotation=rotation_ylabel)
    plt.grid()


if __name__ == "__main__":
    pass