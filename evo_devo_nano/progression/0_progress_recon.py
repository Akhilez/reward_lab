
import torch
from torch.nn import functional as F

from evo_devo_nano.interaction import Interaction, Memory
from evo_devo_nano.model.model import CraftaxModel


def do_it_again():

    interaction = Interaction()
    model = CraftaxModel(h=interaction.h, w=interaction.w, n_actions=interaction.n_actions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ----------------- Recon obs backprop -----------------
    model.train()
    obs = torch.tensor(interaction.obs)  # [64, 64, 3] (0-1)
    obs = obs.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 64, 64]
    obs_embeddings, reconstruction = model.obs_encoder(obs, return_reconstruction=True)  # [1, 36, 64], [1, 3, 64, 64]
    loss_recon = F.mse_loss(reconstruction, obs)

    # Backprop
    optimizer.zero_grad()
    loss_recon.backward()
    optimizer.step()

    for i in range(100):
        action = torch.randint(0, interaction.n_actions, (1,))  # [1,]
        # ------------ Step ---------------
        interaction.step(action[0].item())


        # ---------------- Recon next obs backprop --------------------
        model.train()
        obs_next = torch.tensor(interaction.obs)  # [64, 64, 3]
        obs_next = obs_next.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 63, 63]
        obs_next_embeddings, reconstruction_next = model.obs_encoder(obs_next, return_reconstruction=True)


        loss_recon = F.mse_loss(reconstruction_next, obs_next)


        loss = loss_recon
        print({
            "action": action,
            "loss_recon": loss_recon,
        })

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    # do_it()
    do_it_again()

