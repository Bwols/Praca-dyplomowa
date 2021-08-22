import matplotlib.pyplot as plt

import matplotlib.ticker as mtick
gen = [3,3,3,5,6.5,6,7,7,10.5]
dis = [3,3,5,6,7,7,8,6,7]


def draw_gen_dis_plot(gen_loss_v, dis_loss_v, title, output_plot_path = None):
    epochs = [e+1 for e in range(len(gen_loss_v))]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

    plt.plot(epochs, gen_loss_v, label="Błąd generatora")
    plt.plot(epochs, dis_loss_v, label="Błąd dyskryminatora")

    plt.xlabel('Epoka')
    plt.ylabel('Suma błędów epoki')
    plt.title(title)
    plt.legend()
    if output_plot_path != None:
        plt.savefig(output_plot_path)
    else:
        plt.show()

    plt.clf()


def draw_vae_loss_plot(vae_loss_v, title, output_plot_path = None):
    epochs = [e + 1 for e in range(len(vae_loss_v))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

    plt.plot(epochs, vae_loss_v, label="Błąd modelu")

    plt.xlabel('Epoka')
    plt.ylabel('Suma błędów epoki')
    plt.title(title)
    plt.legend()
    if output_plot_path != None:
        plt.savefig(output_plot_path)
    else:
        plt.show()

    plt.clf()


def save_model_loss_data(loss_vector, file_output_path):
    with open(file_output_path,'w') as file:
        file.write(' '.join(str(el)for el in loss_vector)+ '\n')  # el epoch loss



