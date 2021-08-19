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
    plt.ylabel('Błąd MSE')
    plt.title(title)

    if output_plot_path != None:
        plt.savefig(output_plot_path)
    plt.show()



draw_gen_dis_plot(gen,dis,"Błędy generatora i dyskryminaotra przy uczeniu\n optymalizatorem ADAM i funckcją strany BCE")