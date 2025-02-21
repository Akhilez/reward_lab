import torch
from torch.nn import functional as F

from evo_devo_nano.interaction import Interaction, Memory
from evo_devo_nano.model.model import CraftaxModel


def do_it_again():

    interaction = Interaction()
    memory = Memory(5)
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

    obs_embeddings = obs_embeddings.detach()
    obs_embeddings = model.time_encoder(obs_embeddings, timestep=0)  # [1, 38, 64]

    memory.observations.append(obs_embeddings)

    for i in range(999):

        action = torch.randint(0, interaction.n_actions, (1,))  # [1,]
        # ------------ Step ---------------
        interaction.step(action[0].item())


        # ---------------- Recon next obs --------------------
        model.train()
        obs_next = torch.tensor(interaction.obs)  # [64, 64, 3]
        obs_next = obs_next.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 63, 63]
        obs_next_embeddings, reconstruction_next = model.obs_encoder(obs_next, return_reconstruction=True)

        # -------------- Gather embeddings --------------
        obs_embeddings = memory.observations[-1]  # [1, 37, 64]

        obs_next_embeddings = obs_next_embeddings.detach()
        obs_next_embeddings = model.time_encoder(obs_next_embeddings, timestep=i + 1)
        memory.observations.append(obs_next_embeddings)

        action_embedding = model.embeddings(model.vocab.action_indices[action])  # [1, 64]
        action_embedding = action_embedding.unsqueeze(1)  # [1, 1, 64]
        memory.actions.append(action_embedding)

        next_state_key = model.vocab["next_state_key"]  # int
        next_state_key = model.embeddings(torch.tensor(next_state_key).unsqueeze(0))  # [1, 64]
        next_state_key = next_state_key.unsqueeze(0)  # [1, 1, 64]

        action_key = model.vocab["action_key"]  # int
        action_key = model.embeddings(torch.tensor(action_key).unsqueeze(0))  # [1, 64]
        action_key = action_key.view(1, 1, -1)  # [1, 1, 64]

        # --------------- Training -----------------

        seq_forward_dynamics = torch.cat([
            next_state_key,
            obs_embeddings,
            action_key,
            action_embedding,
            next_state_key,
            obs_next_embeddings[:, :-1],  # --> next emb
        ], dim=1)  # [1, 76, 64]
        # TODO: Create mask

        outputs = model.transformer(seq_forward_dynamics)  # [1, 76, 64]

        # -------------- Losses -------------

        loss_recon = F.mse_loss(reconstruction_next, obs_next)

        obs2_emb_with_grad = outputs[:, -obs_next_embeddings.shape[1]:]  # [1, 37, 64]
        loss_forward_dynamics = F.mse_loss(obs2_emb_with_grad, obs_next_embeddings)

        loss = loss_recon + loss_forward_dynamics

        print({
            "action": action,
            # "action_greedy": action_logits.argmax(-1),
            # "action_pred_latent": action_logits_latent.argmax(-1),
            # "action_logits": action_logits,
            # "action_logits_latent": action_logits_latent,
            # "reward_full_state": reward_full_state,
            # "reward_latent": reward_latent,
            "loss_recon": loss_recon,
            "loss_forward_dynamics": loss_forward_dynamics,
            # "loss_inverse_dynamics": loss_inverse_dynamics,
            # "loss_action_latent": loss_action_latent,
            # "loss_policy": loss_policy,
            # "loss_latent_obs": loss_latent_obs,
            "loss": loss,
        })

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




if __name__ == "__main__":
    # do_it()
    do_it_again()

