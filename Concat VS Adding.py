import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __name__=="__main__":
    with open ("train_metrices_withAdding.json","r")as f:
        train_metrices_add=json.load(f)
    with open ("train_metrices_withConcat.json","r")as f:
        train_metrices_con=json.load(f)

    epochsa = train_metrices_add["epochs"]
    epochsb = train_metrices_con["epochs"]
    # Create a figure with two subplots: one for path loss, one for velocity loss
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12, 6))

    # Plot Path MSE for Training and Validation
    line1,=ax1.plot(epochsa, train_metrices_add["train_path_loss"], label='Train Path MSE_add', marker='o')
    line2=ax1.plot(epochsb, train_metrices_con["train_path_loss"], label='Train Path MSE_con', marker='x')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Position Loss Across Epochs')
    ax1.legend()
    ax1.grid(True)
        



        # Plot Velocity MSE for Training and Validation
    line3,=ax2.plot(epochsa, train_metrices_add["train_vel_loss"], label='Train Vel MSE_add', marker='o')
    line3,=ax2.plot(epochsb, train_metrices_con["train_vel_loss"], label='Train Vel MSE_con', marker='x')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Velocity Loss Across Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
